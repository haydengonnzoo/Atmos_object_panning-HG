"""
Shot boundary detection.

Tracking cannot cross a cut: a tracker handed a cut will happily follow
whatever happens to sit near the old position in the new shot, and report
full confidence while doing it. So every object proposal and every track has
to be scoped to a single shot, which makes this a prerequisite for the rest
of the pipeline rather than a nicety.

This replaces the cut rejection inside tracking_pipeline.estimate_global_motion,
which tested only whether the estimated camera translation was implausibly
large. That catches a cut between two very different frames and misses a cut
between two similar ones -- on the Spider-Man trailer it passed three obvious
hard cuts that colour-histogram comparison caught immediately.

Two boundary types are detected:

  hard cut -- one frame to the next, the whole colour distribution changes.
  dissolve -- a gradual blend over several frames, which never spikes but
              stays elevated for its duration. Trailers and title sequences
              are full of these; the Spider-Man logo sequence is one.
"""

import cv2
import numpy as np

# Frames are compared as coarse colour histograms rather than pixels, so that
# ordinary motion inside a shot (a person walking, the camera panning) barely
# registers while a genuine scene change moves most of the mass at once.
HIST_BINS = 8
HIST_SIZE = (160, 90)

# A cut is scored against the local median difference rather than a fixed
# number, because "normal" frame-to-frame change is wildly different between a
# locked-off dialogue shot and a handheld action shot. ABS_FLOOR stops quiet
# footage from turning its own noise into cuts.
ADAPTIVE_WINDOW = 25
ADAPTIVE_MULT = 6.0
ABS_FLOOR = 0.30

# A dissolve never spikes, so it is caught by sustained elevation instead.
DISSOLVE_MULT = 2.0
DISSOLVE_MIN_LEN = 4

MIN_SHOT_FRAMES = 8


def frame_signature(frame):
    """Coarse normalised colour histogram -- the comparison unit for cuts."""
    small = cv2.resize(frame, HIST_SIZE, interpolation=cv2.INTER_AREA)
    hist = cv2.calcHist([small], [0, 1, 2], None,
                        [HIST_BINS] * 3, [0, 256] * 3).flatten()
    total = hist.sum()
    return hist / total if total > 0 else hist


def frame_differences(video_path, max_frames=None, progress=None):
    """L1 distance between consecutive frame signatures, plus frame count."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"cannot open video: {video_path}")

    diffs, prev, count = [], None, 0
    while True:
        ok, frame = cap.read()
        if not ok or (max_frames is not None and count >= max_frames):
            break
        sig = frame_signature(frame)
        if prev is not None:
            diffs.append(float(np.abs(sig - prev).sum()))
        prev = sig
        count += 1
        if progress and count % progress == 0:
            print(f"    scanned {count} frames", flush=True)

    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    cap.release()
    return np.array(diffs), count, fps


def _local_baseline(diffs, i):
    """Median difference around frame i, excluding i itself.

    Excluding the candidate matters: a cut is large enough to drag a short
    window's median up toward itself and mask its own detection.
    """
    lo = max(0, i - ADAPTIVE_WINDOW)
    hi = min(len(diffs), i + ADAPTIVE_WINDOW + 1)
    window = np.concatenate([diffs[lo:i], diffs[i + 1:hi]])
    return float(np.median(window)) if len(window) else 0.0


def find_boundaries(diffs):
    """Frame indices where a new shot begins."""
    cuts = set()

    for i, d in enumerate(diffs):
        if d < ABS_FLOOR:
            continue
        base = _local_baseline(diffs, i)
        if d > max(ABS_FLOOR, base * ADAPTIVE_MULT):
            cuts.add(i + 1)  # diffs[i] compares frame i to i+1

    # Dissolves: a run of frames all moderately above baseline. Only the start
    # of the run is reported -- the blend belongs to whichever shot it leads
    # into, and reporting every frame of it would shatter the timeline.
    global_base = float(np.median(diffs)) if len(diffs) else 0.0
    run_start = None
    for i, d in enumerate(diffs):
        elevated = d > max(ABS_FLOOR * 0.5, global_base * DISSOLVE_MULT)
        if elevated and run_start is None:
            run_start = i
        elif not elevated and run_start is not None:
            if i - run_start >= DISSOLVE_MIN_LEN:
                cuts.add(run_start + 1)
            run_start = None

    return sorted(cuts)


def shots_from_boundaries(cuts, n_frames, min_len=MIN_SHOT_FRAMES):
    """Turn boundary frames into (start, end) inclusive spans."""
    edges = [0] + [c for c in cuts if 0 < c < n_frames] + [n_frames]
    spans = [(edges[i], edges[i + 1] - 1) for i in range(len(edges) - 1)]
    return [(a, b) for a, b in spans if b - a + 1 >= min_len]


def detect_shots(video_path, max_frames=None, min_len=MIN_SHOT_FRAMES, progress=None):
    """Segment a video into shots. Returns (shots, fps, n_frames)."""
    diffs, n_frames, fps = frame_differences(video_path, max_frames, progress)
    cuts = find_boundaries(diffs)
    return shots_from_boundaries(cuts, n_frames, min_len), fps, n_frames


if __name__ == "__main__":
    import sys

    path = sys.argv[1]
    shots, fps, n = detect_shots(path, progress=500)
    print(f"\n{n} frames @ {fps:.3f}fps -> {len(shots)} shots\n")
    for a, b in shots:
        print(f"  {a:>5}-{b:<5} {a/fps:7.2f}s-{b/fps:7.2f}s  ({b-a+1:>4} frames)")
