"""
Object proposals: deciding what in a shot is worth panning.

This is the part every earlier approach failed at. Brightness, saturation,
blob size and local contrast were all tried as a way to tell "an object" from
"background", and all of them failed for the same reason: they describe how a
region *looks*, and how a region looks is not correlated with whether it is a
sound source. A glowing spark and a helicopter are both bright.

The premise here is different, and comes from how the mix actually works. An
Atmos object exists for something the listener localises and follows -- a
vehicle, a projectile, a figure crossing frame. Ambience, sparks, ash and rain
live in the bed. So the question is not "is this region bright" but:

    does this region move independently of the camera,
    and is it big enough on screen to be worth a discrete object?

Both halves matter, and each kills a different false positive:

  * motion disagreement kills static background. A brick building travels
    across frame during a pan, but it travels exactly as the camera predicts.
    A previous SAM 2 run tracked such a building for a full shot at complete
    confidence, which is what motivated this test.
  * the area floor kills sparks, ash and motes -- the things that prompted
    this whole reframing. They move independently, but they are tiny.

What this deliberately does NOT claim is that surviving both filters makes
something a sound source. Traffic, pedestrians and wind-blown foliage all
move independently and can be large. The honest goal is a short ranked list
of candidates worth a human glance, not a final answer.

Proposals are consumed as SAM 2 prompt points -- see sam2_track.py.
"""

import cv2
import numpy as np

from shot_detection import detect_shots
from camera_motion import estimate_global_motion

# Analysis runs downscaled: camera-motion estimation and residuals do not need
# full resolution, and this keeps a whole-shot pass cheap on a laptop GPU-less
# code path. Coordinates are scaled back to full res on output.
ANALYSIS_WIDTH = 480

# A pixel counts as independently moving when it disagrees with the
# camera-predicted frame by more than this.
#
# The threshold is anchored to the MEDIAN residual, not a high percentile.
# A high percentile seems like the natural choice and is actively wrong here:
# residual is dominated by one-to-two-pixel misregistration along building and
# foliage edges, so p99 lands at 108-212 and tracks edge sharpness rather than
# anything moving. On the Spider-Man shot that filtered out the flying object
# (residual 55-125) entirely, and morphological opening then deleted the thin
# edge lines that had passed, leaving nothing at all.
#
# The median instead reflects sensor grain and compression noise -- the actual
# floor we need to clear -- and stays stable when a shot happens to be full of
# hard edges. A fixed threshold of 25 recovered the object in 6 of 6 sampled
# frames where p99 recovered it in 0.
RESIDUAL_MEDIAN_MULT = 5.0
RESIDUAL_FLOOR = 22.0
RESIDUAL_CEILING = 45.0

# Warping leaves a rim of invalid pixels at the frame edge, and parallax makes
# near-edge background disagree with a single global affine. Both produce
# false positives that hug the border.
EDGE_MARGIN = 12

# Letterbox bars must be cropped BEFORE the residual is computed, not merely
# ignored afterwards. Warping the previous frame slides the black bar by the
# camera translation, and the resulting bar-vs-picture edge is the highest
# residual anywhere in frame -- it set the adaptive threshold to 108-212 on
# the Spider-Man shot while the actual flying object measured only 49-116,
# so every real object was filtered out as sub-threshold. The bars also drag
# the percentile down by filling the frame with identical black pixels.
LETTERBOX_BLACK = 22

# Minimum share of the picture a candidate must occupy. This is the spark
# filter, expressed the way the user framed it: would this get its own object?
MIN_AREA_FRAC = 0.0004
MAX_AREA_FRAC = 0.35

MORPH_KERNEL = 5
LINK_MAX_DIST_FRAC = 0.12
MIN_PERSISTENCE = 4

# How many frames back to compare against. Comparing t-1 to t only detects
# objects that move a meaningful distance in a single frame, which quietly
# tunes the whole detector for fast action and misses anything slower. On a
# Ghost in the Shell shot of bodies falling through frame -- unambiguously
# objects worth panning -- a one-frame baseline found 1 region across the
# whole shot.
#
# Widening the baseline amplifies slow motion, but too far and static content
# starts drifting enough to register. Measured region counts:
#
#     shot                        t-1    t-3    t-6
#     bodies falling (missed)       1     82    253
#     figure moving  (missed)       3     88    126
#     static face    (correct 0)    0      2    136   <- false positives
#
# t-3 recovers both misses while leaving genuinely static footage alone.
#
# But t-3 alone is not a replacement, because it breaks the opposite case. A
# fast object travels far enough in three frames that its old and new
# positions stop overlapping and register as separate blobs. Re-running the
# Spider-Man flyby at t-3: the best proposal landed 161px from the verified
# track (38px at t-1) and the score gap between first and second place
# collapsed from 5.7x to 1.1x -- no discrimination left at all.
#
# So both baselines run and their masks are unioned. Fast objects register at
# t-1, slow ones at t-3, and an object seen by both simply merges into one
# region rather than competing.
MOTION_BASELINES = (1, 3)

# Focus: how sharp a region is relative to the average sharpness of the frame.
#
# This reads the cinematographer's own decision rather than trying to re-derive
# it. Focus is where the DP put the viewer's attention, so an in-focus subject
# is by construction the thing the audience is following -- and therefore the
# thing that wants a discrete Atmos object. Distant buildings, background
# detail and defocused bokeh all fall away, which motion and area alone do not
# achieve: a far-off building tracks perfectly well and is large on screen.
#
# Measured on the last automatic run, where 1.0 is the frame average:
#     Spider-Man (real object)         1.74
#     background vanishing point       1.16
#     frame-edge parallax artifact     0.64
# Deep-focus footage like that street shot is the hard case, since everything
# in it is fairly sharp; shallow depth of field separates far harder.
#
# Sampled in a window on the image, NOT inside the residual mask -- the mask
# sits on moving edges, which are high-frequency by definition and would score
# as "in focus" no matter what they belong to.
#
# Only a permissive floor is applied HERE, because at proposal time we do not
# yet know where the object actually is: a residual blob sits on the object's
# moving edges, roughly 38px off the body on the Spider-Man shot, so the focus
# window lands partly on background. Measured that way the real object scored
# 1.13 while a clean measurement on its true position scored 1.74 -- a strict
# threshold at this stage rejected the one correct object in the shot.
#
# The decisive focus filter runs after SAM 2 has produced an actual mask, in
# sam2_track.focus_filter. This mirrors deduplication, which failed at
# proposal level for the same reason and works on the segmented masks.
FOCUS_FLOOR = 0.85
FOCUS_WINDOW_MIN = 10


class Proposal:
    """A region that moved independently, linked across frames of one shot."""

    def __init__(self, frame_idx, centroid, area, focus):
        self.frames = [frame_idx]
        self.centroids = [centroid]
        self.areas = [area]
        self.focuses = [focus]

    def add(self, frame_idx, centroid, area, focus):
        self.frames.append(frame_idx)
        self.centroids.append(centroid)
        self.areas.append(area)
        self.focuses.append(focus)

    @property
    def median_focus(self):
        return float(np.median(self.focuses))

    @property
    def last(self):
        return self.centroids[-1]

    @property
    def persistence(self):
        return len(self.frames)

    @property
    def median_area(self):
        return float(np.median(self.areas))

    @property
    def independent_displacement(self):
        """Total camera-compensated path length.

        Path length rather than start-to-end distance, so an object that
        crosses frame and comes back still scores as having moved.
        """
        pts = np.array(self.centroids, dtype=float)
        if len(pts) < 2:
            return 0.0
        return float(np.hypot(*np.diff(pts, axis=0).T).sum())

    def seed(self):
        """Frame and point to prompt SAM 2 with -- the largest observation,
        where the object is clearest and least likely to be a partial edge."""
        i = int(np.argmax(self.areas))
        return self.frames[i], self.centroids[i], self.areas[i]

    def score(self):
        """Rank candidates. Deliberately simple and inspectable: each factor
        is reported alongside the score so a bad ranking can be diagnosed."""
        # Focus is deliberately absent: it is too noisy at this stage to rank
        # with (see FOCUS_FLOOR). It decides inclusion after segmentation.
        return (self.persistence
                * np.sqrt(self.median_area)
                * (1.0 + self.independent_displacement))


# Dense optical flow, reused rather than rebuilt per frame. FAST preset costs
# about 5ms at analysis resolution, so this is not the expensive part.
#
# MEDIUM rather than FAST: the FAST preset's search range cannot follow a fast
# object. On the Spider-Man flyby, where the figure moves ~30px per frame, FAST
# put the real object 112px from truth while MEDIUM found it at 38px, for an
# extra 0.2s across the whole shot.
_dis_flow = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)

# Optical flow is undefined where there is no texture to track: in a clear sky
# or a flat wall, every candidate match looks equally good and the returned
# vector is arbitrary. Those pixels then "disagree" with the camera model for
# no physical reason. On the Spider-Man shot this put a motionless patch of
# distant sky at rank 1, scoring 50x the real object, purely because the sky
# sits at effective infinity and moves less than the affine fitted to the
# nearer buildings predicts.
#
# Gating on local texture SHOULD remove that whole class of artifact -- no
# texture means no trustworthy flow -- but measured against ground truth it
# only ever made things worse, so it is disabled. Raising it fragments regions
# and buries the real object rather than cleaning up the sky:
#
#     texture   proposals   Spider-Man rank   distance   bodies-shot proposals
#       0.0         2             #2            38px            28
#       2.0         4             #3            37px            23
#       4.0         7             #5            38px            34
#       6.0         7             #5            38px            33
#
# The distance never improves, so the gate is not helping localisation; it is
# only stripping texture off the real objects along with the sky. Left in place
# and set to 0 because the reasoning is sound and a better formulation (gating
# on flow confidence rather than raw texture) may yet work.
FLOW_MIN_TEXTURE = 0.0

# Unexplained motion, in PIXELS of displacement, needed to call a pixel
# independently moving. Anchored to the median so it adapts to how well the
# camera model happens to fit a given shot, with a floor for clean footage.
FLOW_RESIDUAL_FLOOR = 1.0
FLOW_MEDIAN_MULT = 4.0
FLOW_RESIDUAL_CEILING = 6.0


def flow_residual_mask(prev_gray, curr_gray):
    """Pixels whose MOTION the camera model cannot explain.

    This replaces comparing warped pixel intensities, which was the single
    largest defect in the detector. Intensity difference is only a proxy for
    motion, and a poor one: a one-pixel misalignment along a high-contrast
    building edge produces a huge intensity difference while representing zero
    independent motion. That is why the adaptive threshold settled at 108-212
    and tracked edge sharpness, why a real flying object measuring 49-125 fell
    below its own background, and why morphological opening then deleted the
    thin edge lines that had passed, leaving nothing at all.

    Measuring in motion space removes the failure at its source. A correctly
    tracked edge has flow matching the camera prediction and residual near
    zero, however bright the edge is. The threshold is also physically
    meaningful -- pixels of unexplained displacement -- rather than an
    intensity number that means something different in every shot.
    """
    M = estimate_global_motion(prev_gray, curr_gray)
    if M is None:
        return None

    flow = _dis_flow.calc(prev_gray, curr_gray, None)
    h, w = curr_gray.shape
    xs, ys = np.meshgrid(np.arange(w, dtype=np.float32),
                         np.arange(h, dtype=np.float32))

    # Flow a purely static point would show under the estimated camera motion:
    # where the affine sends it, minus where it started.
    pred_x = M[0, 0] * xs + M[0, 1] * ys + M[0, 2] - xs
    pred_y = M[1, 0] * xs + M[1, 1] * ys + M[1, 2] - ys

    resid = np.hypot(flow[..., 0] - pred_x, flow[..., 1] - pred_y)

    noise = float(np.median(resid))
    thresh = min(FLOW_RESIDUAL_CEILING,
                 max(FLOW_RESIDUAL_FLOOR, FLOW_MEDIAN_MULT * noise))
    mask = (resid > thresh).astype(np.uint8)

    # Discard pixels with too little local texture for flow to mean anything.
    # Local standard deviation via the mean-of-squares identity, which is one
    # pass of box filtering rather than a per-pixel window.
    g32 = curr_gray.astype(np.float32)
    mean = cv2.boxFilter(g32, -1, (7, 7))
    mean_sq = cv2.boxFilter(g32 * g32, -1, (7, 7))
    texture = np.sqrt(np.maximum(mean_sq - mean * mean, 0.0))
    mask[texture < FLOW_MIN_TEXTURE] = 0

    mask[:EDGE_MARGIN, :] = 0
    mask[-EDGE_MARGIN:, :] = 0
    mask[:, :EDGE_MARGIN] = 0
    mask[:, -EDGE_MARGIN:] = 0

    k = np.ones((MORPH_KERNEL, MORPH_KERNEL), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)


def picture_bounds(frames, sample=8):
    """(y0, y1, x0, x1) of real picture content, excluding letterbox bars.

    Measured from the shot's own frames rather than the whole video, so a
    shot with different framing is handled on its own terms. A row is treated
    as bar only if it stays black across every sampled frame -- content that
    is merely dark in one frame must not be cropped away.
    """
    idx = np.linspace(0, len(frames) - 1, min(sample, len(frames)), dtype=int)
    brightest = np.maximum.reduce([frames[i].max(axis=2) for i in idx])

    def span(profile):
        lit = np.where(profile >= LETTERBOX_BLACK)[0]
        return (int(lit[0]), int(lit[-1])) if len(lit) else (0, len(profile) - 1)

    y0, y1 = span(brightest.max(axis=1))
    x0, x1 = span(brightest.max(axis=0))
    return y0, y1, x0, x1


def independent_motion_mask(prev_gray, curr_gray):
    """Pixels that moved in a way the camera motion does not explain.

    Returns None when camera motion cannot be estimated -- better to skip a
    frame pair than to subtract a fabricated transform and call the resulting
    garbage an object.
    """
    M = estimate_global_motion(prev_gray, curr_gray)
    if M is None:
        return None

    h, w = curr_gray.shape
    warped = cv2.warpAffine(prev_gray, M, (w, h))
    resid = cv2.absdiff(curr_gray, warped)

    noise = float(np.median(resid))
    thresh = min(RESIDUAL_CEILING, max(RESIDUAL_FLOOR, RESIDUAL_MEDIAN_MULT * noise))
    mask = (resid > thresh).astype(np.uint8)

    # Warping invalidates the border, and parallax breaks the single-affine
    # assumption worst near frame edges.
    mask[:EDGE_MARGIN, :] = 0
    mask[-EDGE_MARGIN:, :] = 0
    mask[:, :EDGE_MARGIN] = 0
    mask[:, -EDGE_MARGIN:] = 0

    k = np.ones((MORPH_KERNEL, MORPH_KERNEL), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)


def sharpness_map(gray):
    """Per-pixel high-frequency energy, and the frame's average.

    Focus is reported as a ratio against the frame average so it stays
    comparable across shots of wildly different contrast and exposure.
    """
    lap = np.abs(cv2.Laplacian(gray.astype(np.float32), cv2.CV_32F))
    return lap, float(lap.mean()) + 1e-6


def regions_in_mask(mask, picture_area, lap, lap_mean):
    """Connected components surviving the screen-area filter, with focus."""
    n, _, stats, centroids = cv2.connectedComponentsWithStats(mask)
    lo = MIN_AREA_FRAC * picture_area
    hi = MAX_AREA_FRAC * picture_area
    h, w = lap.shape
    out = []
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if not (lo <= area <= hi):
            continue
        cx, cy = float(centroids[i][0]), float(centroids[i][1])
        # Window scaled to the region, sampled on the IMAGE rather than inside
        # the residual mask -- moving edges are high-frequency whatever they
        # belong to, so measuring within the mask would score everything sharp.
        r = max(FOCUS_WINDOW_MIN, int(np.sqrt(area / np.pi)))
        win = lap[max(0, int(cy) - r):min(h, int(cy) + r),
                  max(0, int(cx) - r):min(w, int(cx) + r)]
        focus = float(win.mean() / lap_mean) if win.size else 0.0
        out.append(((cx, cy), area, focus))
    return out


def propose_for_shot(frames, verbose=False):
    """Rank independently-moving regions across one shot's frames.

    `frames` is a list of BGR images, already scoped to a single shot -- this
    must never be handed footage spanning a cut.
    """
    if len(frames) < 2:
        return []

    y0, y1, x0, x1 = picture_bounds(frames)
    cropped = [f[y0:y1 + 1, x0:x1 + 1] for f in frames]

    h, w = cropped[0].shape[:2]
    scale = ANALYSIS_WIDTH / w
    size = (ANALYSIS_WIDTH, int(round(h * scale)))
    grays = [cv2.cvtColor(cv2.resize(f, size, interpolation=cv2.INTER_AREA),
                          cv2.COLOR_BGR2GRAY) for f in cropped]

    picture_area = float(size[0] * size[1])
    link_dist = LINK_MAX_DIST_FRAC * size[0]
    if verbose:
        print(f"    picture rows {y0}-{y1}, cols {x0}-{x1}; "
              f"analysis {size[0]}x{size[1]}, area floor "
              f"{MIN_AREA_FRAC * picture_area:.0f}px")

    active, done = [], []
    baselines = sorted({min(k, max(1, len(grays) - 1)) for k in MOTION_BASELINES})
    for i in range(max(baselines), len(grays)):
        mask = None
        for k in baselines:
            m = MOTION_METHOD(grays[i - k], grays[i])
            if m is None:
                continue
            mask = m if mask is None else cv2.bitwise_or(mask, m)
        if mask is None:
            if verbose:
                print(f"    frame {i}: no camera estimate, skipped")
            continue

        lap, lap_mean = sharpness_map(grays[i])
        regions = regions_in_mask(mask, picture_area, lap, lap_mean)
        if verbose:
            print(f"    frame {i}: {len(regions)} region(s) pass area filter")

        # Greedy nearest-centroid linking, largest regions claimed first so a
        # big object is not stolen by an adjacent scrap of noise.
        unclaimed = list(active)
        for centroid, area, focus in sorted(regions, key=lambda r: -r[1]):
            best, best_d = None, link_dist
            for p in unclaimed:
                d = float(np.hypot(centroid[0] - p.last[0], centroid[1] - p.last[1]))
                if d < best_d:
                    best, best_d = p, d
            if best is not None:
                best.add(i, centroid, area, focus)
                unclaimed.remove(best)
            else:
                active.append(Proposal(i, centroid, area, focus))

        # A proposal that misses a frame is closed rather than coasted. Within
        # a single shot an object worth an Atmos object should be visible
        # continuously; gap-filling is the tracker's job, not the detector's.
        for p in unclaimed:
            active.remove(p)
            done.append(p)

    done.extend(active)
    survivors = [p for p in done
                 if p.persistence >= MIN_PERSISTENCE
                 and p.median_focus >= FOCUS_FLOOR]
    if verbose:
        rejected = [p for p in done if p.persistence >= MIN_PERSISTENCE
                    and p.median_focus < FOCUS_FLOOR]
        print(f"    focus floor: kept {len(survivors)}, "
              f"rejected {len(rejected)} clearly-defocused region(s)")

    # Back to full-frame coordinates: undo the downscale, then undo the crop.
    inv = 1.0 / scale
    for p in survivors:
        p.centroids = [(x * inv + x0, y * inv + y0) for x, y in p.centroids]
        p.areas = [a * inv * inv for a in p.areas]

    return deduplicate(sorted(survivors, key=lambda p: -p.score()))


def deduplicate(proposals, radii_apart=1.5, min_shared=3):
    """Drop proposals that describe the same physical object.

    One object routinely generates several proposals: the residual mask sits
    on its moving edges, and leading and trailing edges can survive as
    separate components that never merge. On the Spider-Man shot this put the
    same figure at ranks 1 and 3, and both were then tracked as independent
    Atmos objects -- their normalised tracks differed by a median of 0.0014.

    Separation is judged against object size rather than a fixed distance, so
    a large near object is not split while two genuinely distinct distant
    ones are kept apart. Input must already be ranked best-first.
    """
    kept = []
    for p in proposals:
        p_radius = np.sqrt(max(p.median_area, 1.0) / np.pi)
        duplicate = False
        for q in kept:
            shared = set(p.frames) & set(q.frames)
            if len(shared) < min_shared:
                continue
            pi = {f: c for f, c in zip(p.frames, p.centroids)}
            qi = {f: c for f, c in zip(q.frames, q.centroids)}
            sep = np.median([np.hypot(pi[f][0] - qi[f][0], pi[f][1] - qi[f][1])
                             for f in shared])
            q_radius = np.sqrt(max(q.median_area, 1.0) / np.pi)
            if sep < radii_apart * max(p_radius, q_radius):
                duplicate = True
                break
        if not duplicate:
            kept.append(p)
    return kept


# Which motion measure the proposal step uses. flow_residual_mask measures
# motion directly; independent_motion_mask compares warped pixel intensities
# and is kept only so the two can be compared on the same footage.
MOTION_METHOD = flow_residual_mask


def read_shot_frames(video_path, start, end):
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    frames = []
    for _ in range(end - start + 1):
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    cap.release()
    return frames


if __name__ == "__main__":
    import sys

    video = sys.argv[1]
    only = (int(sys.argv[2]), int(sys.argv[3])) if len(sys.argv) > 3 else None

    if only:
        shots = [only]
        fps = cv2.VideoCapture(video).get(cv2.CAP_PROP_FPS) or 24.0
    else:
        shots, fps, _ = detect_shots(video)
        print(f"{len(shots)} shots\n")

    for a, b in shots:
        frames = read_shot_frames(video, a, b)
        props = propose_for_shot(frames, verbose=bool(only))
        print(f"\nshot {a}-{b} ({a/fps:.2f}s, {len(frames)} frames): "
              f"{len(props)} proposal(s)")
        for rank, p in enumerate(props[:5], 1):
            sf, (sx, sy), sa = p.seed()
            print(f"  #{rank} score={p.score():10.0f}  seen={p.persistence:>3}fr  "
                  f"med_area={p.median_area:>7.0f}px  indep_disp={p.independent_displacement:>6.0f}px  "
                  f"focus={p.median_focus:4.2f}  "
                  f"seed=frame{a+sf} ({sx:.0f},{sy:.0f}) area={sa:.0f}")
