"""
Camera (global) motion estimation.

Extracted verbatim from tracking_pipeline.py, which was the original
blob-and-Kalman tracker. That file is 874 lines and this is the only part of
it the current pipeline still uses -- everything else in it belongs to
approaches that were tried and abandoned (glow detection, saturation filters,
per-track Kalman prediction, orbiter/strobe/bulb classification). Pulling
these 40 lines out lets the rest be archived without the live code depending
on a large file that is 95% dead.

Behaviour is unchanged from the original.
"""

import cv2
import numpy as np

GLOBAL_MOTION_MAX_FEATURES = 400
GLOBAL_MOTION_QUALITY = 0.01
GLOBAL_MOTION_MIN_DISTANCE = 20
GLOBAL_MOTION_MIN_POINTS = 6
GLOBAL_MOTION_RANSAC_THRESH = 3.0

# Reject implausibly large translations, which a shot cut produces.
#
# NOTE: this is now largely redundant. It predates shot_detection.py, and it
# is a weak test -- it only catches a cut when the two shots differ enough to
# produce a big bogus translation, and it passed three obvious hard cuts on the
# Spider-Man trailer that colour-histogram detection caught immediately. Since
# callers now segment shots first and never hand this footage spanning a cut,
# its main remaining effect is the unwanted one: rejecting a genuine fast whip
# pan as if it were a cut. Left in place so behaviour matches the original;
# worth revisiting deliberately rather than as a side effect of cleanup.
GLOBAL_MOTION_CUT_FRACTION = 0.15


def estimate_global_motion(prev_gray, curr_gray):
    """
    Frame-to-frame camera/global motion as a 2x3 affine transform, from
    sparse feature tracking + RANSAC. Returns None if it can't get a
    confident estimate (too few trackable features -- e.g. a near-black or
    heavily blurred frame), in which case callers should fall back to
    treating raw motion as-is rather than trusting a bad transform.

    Operates in whatever coordinate space the images are given in; callers
    are responsible for scaling positions to match.
    """
    pts_prev = cv2.goodFeaturesToTrack(
        prev_gray, maxCorners=GLOBAL_MOTION_MAX_FEATURES,
        qualityLevel=GLOBAL_MOTION_QUALITY, minDistance=GLOBAL_MOTION_MIN_DISTANCE)
    if pts_prev is None or len(pts_prev) < GLOBAL_MOTION_MIN_POINTS:
        return None

    pts_curr, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, pts_prev, None)
    if pts_curr is None or status is None:
        return None

    ok = status.flatten() == 1
    good_prev, good_curr = pts_prev[ok], pts_curr[ok]
    if len(good_prev) < GLOBAL_MOTION_MIN_POINTS:
        return None

    M, _ = cv2.estimateAffinePartial2D(
        good_prev, good_curr, method=cv2.RANSAC,
        ransacReprojThreshold=GLOBAL_MOTION_RANSAC_THRESH)
    if M is None:
        return None

    if np.hypot(M[0, 2], M[1, 2]) > GLOBAL_MOTION_CUT_FRACTION * prev_gray.shape[1]:
        return None
    return M


def apply_affine(M, x, y):
    """Where a purely-static point at (x, y) would land under transform M."""
    return (M[0, 0] * x + M[0, 1] * y + M[0, 2],
            M[1, 0] * x + M[1, 1] * y + M[1, 2])
