"""
SAM 2 tracking driven by automatic object proposals.

This joins the two halves of the pipeline. object_proposals decides WHAT is
worth tracking (regions that move independently of the camera and are large
enough to deserve a discrete Atmos object); SAM 2 then does the actual
frame-to-frame tracking, which it is very good at and the old hand-rolled
Kalman/blob pipeline was not.

The division of labour matters, because each part fails in a way the other
cannot fix. SAM 2 will track literally anything it is pointed at -- in an
earlier test it followed a brick building for an entire shot at full
confidence and never signalled that anything was wrong. It has no notion of
"worth panning". Conversely the proposal step localises objects only roughly,
because a motion-residual blob sits on an object's moving edges rather than
its body. Proposals choose the target; SAM 2 refines and holds it.

Usage:
    python sam2_track.py <video> <shot_start> <shot_end> [n_objects]
"""

import os
import shutil
import tempfile

import cv2
import numpy as np
import torch

import sam2.sam2_video_predictor as svp

from object_proposals import propose_for_shot, read_shot_frames

MODEL_CKPT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "models", "sam2.1_hiera_base_plus.pt")
MODEL_CFG = "configs/sam2.1/sam2.1_hiera_b+.yaml"


class _Float32Torch:
    """Stands in for `torch` inside sam2_video_predictor, reporting bfloat16
    as float32 and passing everything else through.

    SAM 2 hardcodes `.to(torch.bfloat16)` on its memory-bank features, a
    CUDA-oriented default. The weights load as float32, so the prompt frame
    succeeds (nothing to attend to yet) and the first propagation step then
    hits a bfloat16-vs-float32 matmul in memory attention: MPS aborts inside a
    Metal assertion, CPU raises outright.

    The obvious fix -- bfloat16 autocast, as the official CUDA demo uses -- is
    a trap on Apple Silicon, which has no accelerated CPU bfloat16 path. Every
    matmul falls back to software emulation: measured at over 12 minutes at
    493% CPU for 32 frames, without finishing. Making the cache match the
    weights instead keeps everything float32, needs no autocast, and runs on
    the GPU at roughly 1.2s/frame.
    """

    def __getattr__(self, name):
        return torch.float32 if name == "bfloat16" else getattr(torch, name)


svp.torch = _Float32Torch()

from sam2.build_sam import build_sam2_video_predictor  # noqa: E402  (after patch)


def _write_frames(frames, directory):
    for i, f in enumerate(frames):
        cv2.imwrite(os.path.join(directory, f"{i:05d}.jpg"), f,
                    [cv2.IMWRITE_JPEG_QUALITY, 95])


def build_predictor(device=None):
    """Build the SAM 2 predictor once so batch runs can reuse it.

    Loading the 309MB checkpoint per shot is pure waste in a batch, and on a
    thermally-constrained laptop the wasted work is heat as well as time.
    """
    if device is None:
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    return build_sam2_video_predictor(MODEL_CFG, MODEL_CKPT, device=device)


def track_proposals(frames, proposals, device=None, verbose=True, predictor=None):
    """Track each proposal through the shot. Returns {obj_id: {frame: (area, cx, cy)}}.

    Propagation runs in BOTH directions. Each proposal is seeded at the frame
    where it is largest -- that is where the object is clearest and least
    likely to be a partial edge, but it is usually late in the shot, so
    forward-only propagation would cover almost none of it.
    """
    if not proposals:
        return {}

    if device is None:
        device = "mps" if torch.backends.mps.is_available() else "cpu"

    workdir = tempfile.mkdtemp(prefix="sam2_frames_")
    try:
        _write_frames(frames, workdir)
        if predictor is None:
            predictor = build_sam2_video_predictor(MODEL_CFG, MODEL_CKPT, device=device)
        state = predictor.init_state(video_path=workdir)

        seed_frames = []
        for oid, p in enumerate(proposals, start=1):
            sf, (sx, sy), area = p.seed()
            seed_frames.append(sf)
            predictor.add_new_points_or_box(
                inference_state=state,
                frame_idx=sf,
                obj_id=oid,
                points=np.array([[sx, sy]], dtype=np.float32),
                labels=np.array([1], dtype=np.int32),
            )
            if verbose:
                print(f"  seeded obj {oid} at frame {sf} ({sx:.0f},{sy:.0f}) "
                      f"area={area:.0f} score={p.score():.0f}")

        tracks = {oid: {} for oid in range(1, len(proposals) + 1)}

        sharp = {}

        def frame_sharpness(frame_idx):
            """Cached per-pixel high-frequency energy for one frame."""
            if frame_idx not in sharp:
                g = cv2.cvtColor(frames[frame_idx], cv2.COLOR_BGR2GRAY)
                lap = np.abs(cv2.Laplacian(g.astype(np.float32), cv2.CV_32F))
                sharp[frame_idx] = (lap, float(lap.mean()) + 1e-6)
            return sharp[frame_idx]

        def harvest(frame_idx, obj_ids, mask_logits):
            for i, oid in enumerate(obj_ids):
                m = (mask_logits[i] > 0.0).float().cpu().numpy().squeeze()
                area = int(m.sum())
                if area == 0:
                    tracks[oid].setdefault(frame_idx,
                                           (0, float("nan"), float("nan"), 0.0))
                    continue
                ys, xs = np.nonzero(m)
                prev = tracks[oid].get(frame_idx)
                # Both passes cover the frames between the earliest and latest
                # seed; keep whichever produced an actual mask.
                if prev is None or prev[0] == 0:
                    lap, lap_mean = frame_sharpness(frame_idx)
                    # Focus over the object's own pixels. This is the whole
                    # point of measuring it here rather than at proposal time:
                    # the mask is the object, so nothing is averaged in from
                    # the background behind it.
                    focus = float(lap[m > 0].mean() / lap_mean)
                    tracks[oid][frame_idx] = (area, float(xs.mean()),
                                              float(ys.mean()), focus)

        for out in predictor.propagate_in_video(state):
            harvest(*out)
        if max(seed_frames) > 0:
            for out in predictor.propagate_in_video(
                    state, start_frame_idx=max(seed_frames), reverse=True):
                harvest(*out)

        return tracks
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def deduplicate_tracks(tracks, radii_apart=0.75, min_shared=4):
    """Merge tracked objects that turned out to be the same thing.

    Proposal-level dedup cannot catch these. A single figure yields several
    residual blobs -- torso and legs came out 66px apart on the Spider-Man
    shot, correctly judged distinct at proposal time -- and they only collapse
    into one object once SAM 2 expands each seed to the whole figure. On that
    shot the result was the same character tracked twice, as objects 1 and 3,
    with normalised tracks differing by a median of 0.0014. Panning that would
    put two Atmos objects on one prop.

    Judged on the tracked masks: the segmented area gives a real size to
    measure separation against, which the edge-only proposal blobs did not.
    Input order is the ranking; earlier keys win.
    """
    order = sorted(tracks)
    kept = {}
    for oid in order:
        live = {f: v for f, v in tracks[oid].items() if v[0] > 0}
        if not live:
            continue
        duplicate = False
        for other in kept.values():
            shared = set(live) & set(other)
            if len(shared) < min_shared:
                continue
            sep = np.median([np.hypot(live[f][1] - other[f][1],
                                      live[f][2] - other[f][2]) for f in shared])
            size = np.median([np.sqrt(max(live[f][0], other[f][0]) / np.pi)
                              for f in shared])
            if sep < radii_apart * size:
                duplicate = True
                break
        if not duplicate:
            kept[oid] = live
    return kept


FOCUS_MIN_RATIO = 1.25


def focus_filter(tracks, min_ratio=FOCUS_MIN_RATIO):
    """Drop tracked objects that are not in focus -- i.e. background.

    Focus is the cinematographer's own statement of what matters. A DP pulls
    focus onto the subject the audience is meant to follow, so sharpness is a
    direct read of intended attention, and intended attention is exactly what
    earns a discrete Atmos object. Distant buildings and background detail are
    soft by design, and neither motion nor screen area rejects them: a far-off
    building moves across frame during a pan and can be large.

    Measured against the frame average on the last automatic run:
        Spider-Man (real object)         1.74
        background vanishing point       1.16
        frame-edge parallax artifact     0.64

    Deep-focus footage is the hard case for this, since everything in it is
    fairly sharp. Shallow depth of field separates much harder.
    """
    kept, dropped = {}, {}
    for oid, frames in tracks.items():
        live = [v for v in frames.values() if v[0] > 0]
        if not live:
            continue
        median_focus = float(np.median([v[3] for v in live]))
        (kept if median_focus >= min_ratio else dropped)[oid] = frames
    return kept, dropped


def tracks_to_csv(tracks, names, fps, picture, out_path, time_offset=0.0):
    """Write tracks as the CSV export_tracked_adm.py consumes.

    `picture` is (y0, y1, x0, x1) of real image content -- positions are
    normalised against the picture, not the letterboxed frame, or every object
    reads as vertically compressed toward centre.
    """
    y0, y1, x0, x1 = picture
    width = max(1, x1 - x0)
    height = max(1, y1 - y0)

    rows = ["object,time_sec,x,y_depth,z_height"]
    for oid, frames in sorted(tracks.items()):
        live = {f: v for f, v in frames.items() if v[0] > 0}
        if not live:
            continue
        peak = max(v[0] for v in live.values())
        for f in sorted(live):
            area, cx, cy = live[f][0], live[f][1], live[f][2]
            x = np.clip(((cx - x0) / width) * 2 - 1, -1, 1)
            z = np.clip(1 - ((cy - y0) / height) * 2, -1, 1)
            # Apparent size as a proxy for distance. sqrt(area) is linear in
            # distance, so this is at least the right shape -- but it is the
            # weakest of the three axes and bloom or occlusion wreck it.
            depth = np.clip(1 - np.sqrt(area) / np.sqrt(peak), -1, 1)
            t = time_offset + f / fps
            rows.append(f"{names[oid]},{t:.3f},{x:.3f},{depth:.3f},{z:.3f}")

    with open(out_path, "w") as fh:
        fh.write("\n".join(rows) + "\n")
    return len(rows) - 1


if __name__ == "__main__":
    import sys

    from object_proposals import picture_bounds

    video = sys.argv[1]
    start, end = int(sys.argv[2]), int(sys.argv[3])
    top_n = int(sys.argv[4]) if len(sys.argv) > 4 else 3

    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    cap.release()

    print(f"reading shot {start}-{end} ...")
    frames = read_shot_frames(video, start, end)

    print("proposing objects ...")
    proposals = propose_for_shot(frames)[:top_n]
    print(f"  {len(proposals)} proposal(s) kept\n")

    print("tracking with SAM 2 ...")
    tracks = track_proposals(frames, proposals)

    before = len(tracks)
    tracks = deduplicate_tracks(tracks)
    if len(tracks) < before:
        print(f"  merged {before - len(tracks)} duplicate object(s)")

    tracks, dropped = focus_filter(tracks)
    for oid, fr in sorted(dropped.items()):
        live = [v for v in fr.values() if v[0] > 0]
        print(f"  dropped obj {oid}: out of focus "
              f"(focus={np.median([v[3] for v in live]):.2f}) -- background")

    print()
    names = {oid: f"Object {oid}" for oid in tracks}
    n = tracks_to_csv(tracks, names, fps, picture_bounds(frames),
                      "auto_track.csv", time_offset=start / fps)
    for oid, fr in sorted(tracks.items()):
        live = [v for v in fr.values() if v[0] > 0]
        if live:
            print(f"  obj {oid}: {len(live)}/{len(frames)} frames, "
                  f"area {min(v[0] for v in live)}-{max(v[0] for v in live)}px, "
                  f"focus={np.median([v[3] for v in live]):.2f}")
    print(f"\nwrote auto_track.csv ({n} keyframes, "
          f"timeline offset {start/fps:.3f}s)")
