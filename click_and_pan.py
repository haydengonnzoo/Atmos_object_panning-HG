"""
Point-and-click object panning.

Watch a shot, click the object, name its track. SAM 2 tracks it through the
shot; the automatic detector never has to be right, because you are choosing
the object -- this sidesteps the two problems that kept breaking the fully
automatic path:

  * RECALL. An automatic detector can miss an object entirely (Iron Man, in
    the exact shot where Spider-Man tracked perfectly) and nothing downstream
    can recover it -- a classifier only re-sorts candidates that already
    exist. A click has no recall problem: you are not detecting the object,
    you are pointing at it.

  * CORRESPONDENCE. Nothing in pixels says a tracked blob belongs on
    "Fx Object 17" rather than "Fx Object 23" -- that mapping lives in how you
    built the session. Naming the track at click time answers this by
    construction, and if the name matches a real Pro Tools track exactly,
    Import Session Data can auto-match it instead of asking you to map by hand.

SAM 2 itself is unchanged from the automatic pipeline -- it is not the part
that was failing. Only the FOCUS FILTER is skipped for manual clicks: if you
clicked it, you already decided it is worth tracking, and overriding that
with a heuristic that exists to guess what a human would want would be
perverse when a human already said so.

The automatic detector still runs, quietly, on every shot you touch. It logs
whether it would have proposed the same region you clicked -- nothing is
shown here, this is purely so a labelled dataset accumulates for later
(see DETECTOR_LOG). That log is the raw material for training_data_review.py
style follow-up work, not something this tool consumes itself.

Session state is saved after every shot, so you can quit ('q') and resume
later with the same command -- already-handled shots are skipped.

Keys, per shot:
    click video      mark an object
    s                next shot (keep whatever's marked here)
    w                go back and redo the PREVIOUS shot -- undoes its old
                     objects from the export first, so redoing it never
                     leaves duplicates behind
    Esc              discard everything marked in THIS shot, move on
    u                undo the last click
    a / d            step one frame back / forward
    space            jump to the middle of the shot
    q                quit and save -- resume any time with the same command
'w' and Esc used to be one key ('s' doubled as both skip and advance) --
split apart because 's' could silently discard clicks you had already
named, with no warning that anything had been lost.

Usage:
    python click_and_pan.py [--reset] <video.mp4> [track_names.txt]

--reset clears any saved session for this video and starts over from the
first shot. Use it if a resumed session looks stuck or wrong -- e.g. every
shot already marked 'skipped' with no objects ever added, which is what a
stale test session looks like, not real progress.
"""

import json
import os
import queue
import subprocess
import sys
import threading
import time

import cv2
import numpy as np

from object_proposals import propose_for_shot, read_shot_frames, picture_bounds
from sam2_track import build_predictor, track_proposals
from shot_detection import detect_shots

# Shots this short are near-certainly a countdown leader, a title card, or a
# fragment too brief to be worth a click -- same floor batch_validate.py uses.
MIN_CONTENT_FRAMES = 27

DISPLAY_MAX_W = 1280
DISPLAY_MAX_H = 800

ESC_KEY = 27

DETECTOR_LOG = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "detector_agreement_log.jsonl")


class ManualSeed:
    """Adapts a manual click to the interface track_proposals expects.

    track_proposals calls .seed() -> (frame_idx, (x, y), area) and, when
    verbose, .score() for a progress line. area/score are display-only for a
    manual seed, so they are stubbed rather than computed.
    """

    def __init__(self, frame_idx, x, y):
        self._seed = (frame_idx, (x, y), 0)

    def seed(self):
        return self._seed

    def score(self):
        return 0.0


# ---------------------------------------------------------------------------
# session state (resume support)
# ---------------------------------------------------------------------------

def session_path(video):
    return os.path.splitext(video)[0] + "_click_session.json"


def load_session(video, shots, fps):
    path = session_path(video)
    if os.path.exists(path):
        with open(path) as fh:
            state = json.load(fh)
        if state.get("shots") == [list(s) for s in shots]:
            # shot_csv_lines/shot_objects are newer fields -- default them in
            # for a session saved before "go back" existed, rather than
            # requiring a --reset just to pick up the new capability.
            state.setdefault("shot_csv_lines", {})
            state.setdefault("shot_objects", {})
            return state
        print("  (video's shot list changed since last session -- starting fresh)")
    return {"video": video, "fps": fps, "shots": [list(s) for s in shots],
            "done": {}, "objects": [],           # done: {"a-b": "done"|"skipped"}
            "shot_csv_lines": {}, "shot_objects": {}}


def save_session(video, state):
    with open(session_path(video), "w") as fh:
        json.dump(state, fh, indent=2)


# ---------------------------------------------------------------------------
# track-name list (optional autocomplete)
# ---------------------------------------------------------------------------

def load_track_names(path):
    if not path or not os.path.exists(path):
        return []
    with open(path) as fh:
        return [line.strip() for line in fh if line.strip()]


def prompt_track_name(known_names):
    """Terminal prompt for a track name, with a numbered pick-list.

    Runs alongside the OpenCV window rather than inside it -- cv2 has no text
    input widget, and shelling out to the terminal for this one interaction is
    far simpler than building one.
    """
    if known_names:
        print("  known tracks:", "  ".join(f"[{i}] {n}" for i, n in enumerate(known_names)))
        print("  type a number to reuse a name, or type a new track name:")
    else:
        print("  type the destination track name (e.g. 'Fx Object 17'):")
    while True:
        raw = input("  > ").strip()
        if not raw:
            print("  (empty -- try again)")
            continue
        if raw.isdigit() and int(raw) < len(known_names):
            return known_names[int(raw)]
        if raw not in known_names:
            known_names.append(raw)
        return raw


# ---------------------------------------------------------------------------
# shared visual theme -- one look across the clicking screen and the review
# screen, rather than two windows that feel like different tools. Colours are
# BGR (OpenCV order), chosen for a dark, low-glare palette that won't fight
# with whatever's on screen behind it during a long session.
# ---------------------------------------------------------------------------

BG = (24, 22, 20)             # canvas background, deep warm charcoal
PANEL = (46, 42, 38)           # scrubber track background
TRACK_LINE = (92, 86, 80)      # scrubber's static line
DIVIDER = (70, 65, 60)         # hairline between video and chrome
ACCENT = (235, 220, 80)        # playhead / primary readout -- soft cyan-teal
GOOD = (120, 210, 70)          # confirmed / actively-tracked
WARN = (50, 165, 255)          # needs-your-attention amber
LOST = (80, 70, 230)           # lost-track red
MUTED = (150, 148, 145)        # secondary / help text
SHADOW = (0, 0, 0)
FONT = cv2.FONT_HERSHEY_SIMPLEX

# Extra chrome drawn BELOW the video image, in pixels. DISPLAY_MAX_H bounds
# only the video portion; the window ends up this much taller.
SCRUBBER_H = 28
TIMESTAMP_H = 28
INSTRUCTIONS_H = 100


def _text(canvas, text, pos, color, scale=0.5, weight=1):
    """Anti-aliased text with a dark outline, legible over any frame content."""
    cv2.putText(canvas, text, pos, FONT, scale, SHADOW, weight + 2, cv2.LINE_AA)
    cv2.putText(canvas, text, pos, FONT, scale, color, weight, cv2.LINE_AA)


def _draw_scrubber(canvas, y0, width, cur_frac, ticks=()):
    """A clean seek bar: track line, optional tick marks, current-position dot."""
    cv2.rectangle(canvas, (0, y0), (width, y0 + SCRUBBER_H), PANEL, -1)
    cv2.line(canvas, (4, y0 + SCRUBBER_H // 2), (width - 4, y0 + SCRUBBER_H // 2),
             TRACK_LINE, 2, cv2.LINE_AA)
    for frac in ticks:
        tx = int(4 + frac * (width - 8))
        cv2.line(canvas, (tx, y0 + 4), (tx, y0 + SCRUBBER_H - 4), GOOD, 3, cv2.LINE_AA)
    px = int(4 + cur_frac * (width - 8))
    cv2.circle(canvas, (px, y0 + SCRUBBER_H // 2), 7, ACCENT, -1, cv2.LINE_AA)
    cv2.circle(canvas, (px, y0 + SCRUBBER_H // 2), 7, SHADOW, 1, cv2.LINE_AA)


def _drain_stale_keys():
    """Discard any keypresses queued before this window's input loop starts.

    SAM 2 tracking is a blocking call with no window open at all while it
    runs -- the ShotEditor window is destroyed first, and nothing replaces it
    until TrackReview opens afterward. A key pressed during that gap (very
    plausible: nothing visibly responds, so waiting feels like nothing is
    happening) can sit queued by the OS and then be consumed instantly the
    moment the NEXT window calls cv2.waitKey -- silently firing 'c' (accept)
    the instant review opens, or 's' (advance) the instant the next shot's
    editor opens, cascading through several shots before anything is drawn.
    Draining here is the fix: each new window starts from a clean queue.
    """
    for _ in range(10):
        if cv2.waitKey(1) == -1:
            break


# ---------------------------------------------------------------------------
# the click window
# ---------------------------------------------------------------------------


class ShotEditor:
    """One shot's clicking UI: scrub frames, click objects, name them.

    Naming happens in the terminal (cv2 has no text-entry widget), but that
    MUST NOT be a blocking input() call on the main thread -- that was the
    freeze reported on first use. It wasn't a crash: the moment the main
    thread stops pumping cv2's event loop, macOS marks the window "Not
    Responding" (spinning wheel) even though the process is fine, just
    waiting on a terminal the user may not have been looking at. Typing runs
    on a background thread instead, so cv2.waitKey keeps being called every
    ~20ms and the window stays visibly alive the whole time.
    """

    def __init__(self, frames, shot_label, fps, shot_start_frame):
        self.frames = frames
        self.shot_label = shot_label
        self.fps = fps
        self.shot_start_frame = shot_start_frame
        self.idx = len(frames) // 2          # start mid-shot: least likely to
        self.clicks = []                     # be a dissolve/blur at the cut
        self.pending = None
        self.awaiting_name = None            # (x, y, result_queue) while typing

        h, w = frames[0].shape[:2]
        self.scale = min(DISPLAY_MAX_W / w, DISPLAY_MAX_H / h, 1.0)
        self.disp_w = int(w * self.scale)
        self.disp_h = int(h * self.scale)
        self.win = f"click_and_pan: {shot_label}"
        cv2.namedWindow(self.win)
        cv2.setMouseCallback(self.win, self._on_mouse)

    def _on_mouse(self, event, x, y, flags, userdata):
        if event != cv2.EVENT_LBUTTONDOWN or self.awaiting_name is not None:
            return                            # ignore clicks while a name is pending
        if y < self.disp_h:
            self.pending = (x / self.scale, y / self.scale)
        elif self.disp_h <= y < self.disp_h + SCRUBBER_H:
            self._seek_to(x)

    def _seek_to(self, disp_x):
        frac = float(np.clip(disp_x / max(1, self.disp_w), 0.0, 1.0))
        self.idx = int(round(frac * (len(self.frames) - 1)))

    def _compose_canvas(self):
        """Build the full display image. No GUI calls -- kept separate from
        _draw() so this can be tested by inspecting pixel values directly,
        without a real display."""
        video = cv2.resize(self.frames[self.idx], (self.disp_w, self.disp_h))
        for fr, x, y, name in self.clicks:
            if fr == self.idx:
                cv2.circle(video, (int(x * self.scale), int(y * self.scale)), 8, GOOD, 2, cv2.LINE_AA)
                _text(video, name, (int(x * self.scale) + 12, int(y * self.scale) + 4), GOOD)

        canvas = np.full((self.disp_h + SCRUBBER_H + TIMESTAMP_H + INSTRUCTIONS_H,
                          self.disp_w, 3), BG, dtype=np.uint8)
        canvas[:self.disp_h, :] = video
        cv2.line(canvas, (0, self.disp_h), (self.disp_w, self.disp_h), DIVIDER, 1)

        # No per-click tick marks here -- a click has no real per-frame position
        # yet (SAM 2 hasn't run), so a marker on the scrubber can only ever mean
        # "you clicked somewhere in this shot", not "the object is here". That
        # distinction wasn't obvious from a tick alone and read as the object
        # having vanished when scrubbed away from its exact click frame. The
        # persistent status line below is the real fix; a genuine per-frame
        # position only exists once SAM 2 has tracked it, in TrackReview.
        span = max(1, len(self.frames) - 1)
        _draw_scrubber(canvas, self.disp_h, self.disp_w, self.idx / span)

        shot_t = self.idx / self.fps
        shot_dur = span / self.fps
        video_t = (self.shot_start_frame + self.idx) / self.fps
        ts = (f"shot time {shot_t:5.2f}s / {shot_dur:5.2f}s    "
              f"video time {video_t:7.2f}s    frame {self.idx+1}/{len(self.frames)}")
        _text(canvas, ts, (8, self.disp_h + SCRUBBER_H + 20), ACCENT, 0.55)

        # Marked-objects line: ALWAYS visible, independent of the current scrub
        # position. The on-frame circle above only appears on its own exact
        # frame -- scrub away and it vanishes, which read as "the object got
        # lost" even though nothing was lost; the click was safe the whole
        # time. This line is what actually can't disappear from scrubbing.
        if self.clicks:
            marked = "  ".join(f"{name}@f{fr}" for fr, _, _, name in self.clicks)
            marked_line = f"marked: {marked}"
        else:
            marked_line = "marked: (none yet -- click the video to mark an object)"

        if self.awaiting_name is not None:
            lines = [f"{self.shot_label}   ({len(self.clicks)} object(s) marked)",
                     marked_line,
                     ">>> TYPE THE TRACK NAME IN THE TERMINAL WINDOW NOW <<<",
                     "(this window stays open -- switch to Terminal to finish naming it)",
                     ""]
            colors = [ACCENT, GOOD, WARN, MUTED, MUTED]
        else:
            lines = [f"{self.shot_label}   ({len(self.clicks)} object(s) marked)",
                     marked_line,
                     "click video=mark object   click/drag scrubber=seek   a/d=step frame   space=jump to middle",
                     "s=next shot   w=go back & redo previous shot   u=undo last click",
                     "Esc=discard everything marked THIS shot   q=quit & save (resume later)"]
            colors = [ACCENT, GOOD, MUTED, MUTED, WARN]
        base_y = self.disp_h + SCRUBBER_H + TIMESTAMP_H
        for i, (line, col) in enumerate(zip(lines, colors)):
            _text(canvas, line, (8, base_y + 20 + i * 20), col)
        return canvas

    def _draw(self):
        cv2.imshow(self.win, self._compose_canvas())

    def run(self, known_names):
        """Returns ('done', clicks) | ('skip', []) | ('back', clicks so far)
        | ('quit', clicks so far)."""
        _drain_stale_keys()
        while True:
            self._draw()
            key = cv2.waitKey(20) & 0xFF

            if self.awaiting_name is not None:
                x, y, result_q = self.awaiting_name
                try:
                    name = result_q.get_nowait()
                except queue.Empty:
                    continue                  # keep pumping the window, don't block
                self.clicks.append((self.idx, x, y, name))
                self.awaiting_name = None
                # Best-effort: typing in Terminal can leave IT holding
                # keyboard focus, so a keypress meant for this window (like
                # 'n' to move on) can silently go nowhere until you click
                # back on the video. This nudges the window forward; it is
                # not guaranteed to reclaim focus on every macOS/OpenCV
                # combination, so the printed instruction is the real fix.
                try:
                    cv2.setWindowProperty(self.win, cv2.WND_PROP_TOPMOST, 1)
                    cv2.setWindowProperty(self.win, cv2.WND_PROP_TOPMOST, 0)
                except cv2.error:
                    pass
                print(f"  -> '{name}' added ({len(self.clicks)} object(s) this shot)")
                print(f"  click back on the video window before pressing "
                      f"n/s/a/d -- Terminal may still have keyboard focus\n")
                continue

            if self.pending is not None:
                x, y = self.pending
                self.pending = None
                print(f"\n  marked object at frame {self.idx} ({x:.0f},{y:.0f})")
                result_q = queue.Queue()
                threading.Thread(
                    target=lambda: result_q.put(prompt_track_name(known_names)),
                    daemon=True).start()
                self.awaiting_name = (x, y, result_q)
                continue

            # 's' (advance) and Esc (discard) used to be the SAME key -- 's'
            # meant "skip" but silently won even over clicks you'd already
            # made and named, which is the most likely reason a full run
            # through this trailer ended with every shot marked "skipped"
            # despite names having been typed along the way. Splitting them
            # means the two actions can no longer be confused for each other.
            if key == ord('q'):
                cv2.destroyWindow(self.win)
                return "quit", self.clicks
            if key in (ord('s'), ord('n')):        # advance -- 'n' kept as a
                cv2.destroyWindow(self.win)          # quiet alias, same action
                return "done", self.clicks
            if key == ord('w'):
                cv2.destroyWindow(self.win)
                return "back", self.clicks
            if key == ESC_KEY:
                if self.clicks:
                    print(f"  discarding {len(self.clicks)} marked object(s) "
                          f"in this shot: {', '.join(c[3] for c in self.clicks)}")
                else:
                    print("  nothing marked -- skipping this shot")
                cv2.destroyWindow(self.win)
                return "skip", []
            if key == ord('u') and self.clicks:
                removed = self.clicks.pop()
                print(f"  undid: '{removed[3]}'")
            elif key == ord('a'):
                self.idx = max(0, self.idx - 1)
            elif key == ord('d'):
                self.idx = min(len(self.frames) - 1, self.idx + 1)
            elif key == ord(' '):
                self.idx = len(self.frames) // 2


class _ProcessingWindow:
    """A static 'please wait' window held open for a slow blocking call.

    ShotEditor destroys its own window before returning, and SAM 2 tracking
    -- the slow step -- runs as one blocking call with no window open at all
    while it works. That gap is exactly where a keypress is most likely: the
    screen goes quiet and nothing visibly responds, so pressing something
    again feels reasonable even though it does nothing useful and, before the
    stale-key drain existed, could fire the instant the NEXT window opened.
    This keeps a window on screen the whole time so it never looks like
    nothing is happening, and drains on close so nothing queued during the
    wait leaks into whatever opens next.
    """

    def __init__(self, message):
        self.win = "click_and_pan: working"
        cv2.namedWindow(self.win)
        canvas = np.full((160, 640, 3), BG, dtype=np.uint8)
        _text(canvas, message, (16, 70), ACCENT, 0.6)
        _text(canvas, "this can take a while for a long or fast-moving shot",
              (16, 100), MUTED)
        cv2.imshow(self.win, canvas)
        cv2.waitKey(1)                  # force the frame to actually paint

    def close(self):
        cv2.destroyWindow(self.win)
        _drain_stale_keys()


# ---------------------------------------------------------------------------
# tracking review -- watch SAM 2's actual result, not the click
# ---------------------------------------------------------------------------

class TrackReview:
    """After SAM 2 tracks a shot's marked objects, scrub or play through and
    watch each one's REAL per-frame position. The circle rides the actual
    tracked mask centroid, not the point you originally clicked -- so drift,
    or a lock lost partway through (the real risk on something like Spider-Man
    sliding behind a window frame), is visible immediately instead of being
    discovered later staring at an automation curve in Pro Tools.

    'p' auto-plays through the shot at its own frame rate, so the object's
    motion actually reads as motion rather than a manual frame-by-frame step.
    """

    def __init__(self, frames, shot_label, fps, shot_start_frame, tracks, names):
        self.frames = frames
        self.shot_label = shot_label
        self.fps = fps
        self.shot_start_frame = shot_start_frame
        self.tracks = tracks              # {oid: {frame: (area, cx, cy, focus)}}
        self.names = names                # {oid: name}
        self.idx = 0
        self.playing = True               # start playing -- that's the point
        self._last_advance = time.time()

        h, w = frames[0].shape[:2]
        self.scale = min(DISPLAY_MAX_W / w, DISPLAY_MAX_H / h, 1.0)
        self.disp_w = int(w * self.scale)
        self.disp_h = int(h * self.scale)
        self.win = f"review: {shot_label}"
        cv2.namedWindow(self.win)
        cv2.setMouseCallback(self.win, self._on_mouse)

    def _on_mouse(self, event, x, y, flags, userdata):
        if event == cv2.EVENT_LBUTTONDOWN and self.disp_h <= y < self.disp_h + SCRUBBER_H:
            frac = float(np.clip(x / max(1, self.disp_w), 0.0, 1.0))
            self.idx = int(round(frac * (len(self.frames) - 1)))
            self.playing = False

    def _compose_canvas(self):
        video = cv2.resize(self.frames[self.idx], (self.disp_w, self.disp_h))
        lost = []
        for oid, name in self.names.items():
            entry = self.tracks.get(oid, {}).get(self.idx)
            if entry and entry[0] > 0:
                _, cx, cy, _focus = entry
                x, y = int(cx * self.scale), int(cy * self.scale)
                cv2.circle(video, (x, y), 9, GOOD, 2, cv2.LINE_AA)
                _text(video, name, (x + 13, y + 4), GOOD)
            else:
                lost.append(name)

        span = max(1, len(self.frames) - 1)
        canvas = np.full((self.disp_h + SCRUBBER_H + TIMESTAMP_H + INSTRUCTIONS_H,
                          self.disp_w, 3), BG, dtype=np.uint8)
        canvas[:self.disp_h, :] = video
        cv2.line(canvas, (0, self.disp_h), (self.disp_w, self.disp_h), DIVIDER, 1)
        _draw_scrubber(canvas, self.disp_h, self.disp_w, self.idx / span)

        video_t = (self.shot_start_frame + self.idx) / self.fps
        ts = (f"frame {self.idx+1}/{len(self.frames)}    video time {video_t:7.2f}s    "
              f"{'PLAYING' if self.playing else 'paused'}")
        _text(canvas, ts, (8, self.disp_h + SCRUBBER_H + 20), ACCENT, 0.55)

        status = (f"lost track this frame: {', '.join(lost)}" if lost
                  else "all objects tracked on this frame")
        _text(canvas, status, (8, self.disp_h + SCRUBBER_H + TIMESTAMP_H + 20),
              LOST if lost else GOOD)

        help_lines = ["p=play/pause   a/d=step frame   click/drag scrubber=seek",
                     "c=accept & continue   r=redo this shot's clicks"]
        for i, line in enumerate(help_lines):
            _text(canvas, line, (8, self.disp_h + SCRUBBER_H + TIMESTAMP_H + 40 + i * 20), MUTED)
        return canvas

    def run(self):
        """Returns 'accept' or 'redo'."""
        _drain_stale_keys()
        while True:
            if self.playing:
                now = time.time()
                if now - self._last_advance >= 1.0 / self.fps:
                    self.idx = (self.idx + 1) % len(self.frames)
                    self._last_advance = now
            cv2.imshow(self.win, self._compose_canvas())
            key = cv2.waitKey(15) & 0xFF

            if key == ord('p'):
                self.playing = not self.playing
                self._last_advance = time.time()
            elif key == ord('a'):
                self.idx = max(0, self.idx - 1)
                self.playing = False
            elif key == ord('d'):
                self.idx = min(len(self.frames) - 1, self.idx + 1)
                self.playing = False
            elif key == ord('c'):
                cv2.destroyWindow(self.win)
                return "accept"
            elif key == ord('r'):
                cv2.destroyWindow(self.win)
                return "redo"


# ---------------------------------------------------------------------------
# background detector-agreement logging (never shown to the user)
# ---------------------------------------------------------------------------

MATCH_TOLERANCE_FRAC = 0.08


def log_detector_agreement(video, shot, clicks, tracks, proposals, frames):
    """Would the automatic detector have proposed what you just clicked?

    For each manually tracked object, checks whether any automatic proposal's
    centroid ever came within matching distance of it. Logged with the
    proposal's features (persistence, area, focus, displacement) when there is
    a match, or flagged as a clean miss when there is none. This is the raw
    material for eventually training a classifier on proposal features, and
    for measuring the automatic detector's true recall against real
    selections -- neither of which this tool acts on itself.

    Distance tolerance is a fraction of FRAME width, not the tracked object's
    own mask radius. Object radius was the first thing tried and it was wrong:
    a mask is a few pixels across in an object's first tracked frames, and a
    proposal sits on the object's moving EDGE rather than its centre -- a
    known correct match runs 31-38px off regardless of object size (measured
    against a hand-verified track earlier in this project). Scaling to the
    object's own tiny early radius made that same correct match register as
    zero-overlap on a live check of this exact function.
    """
    frame_w = frames[0].shape[1] if frames else 640
    tolerance = MATCH_TOLERANCE_FRAC * frame_w

    records = []
    for oid, (frame0, x0, y0, name) in enumerate(clicks, start=1):
        live = {f: v for f, v in tracks.get(oid, {}).items() if v[0] > 0}
        if not live:
            continue

        best = None
        for p in proposals:
            p_pts = dict(zip(p.frames, p.centroids))
            shared = set(live) & set(p_pts)
            if not shared:
                continue
            dists = [np.hypot(p_pts[f][0] - live[f][1], p_pts[f][1] - live[f][2])
                     for f in shared]
            hit_frac = float(np.mean([d < tolerance for d in dists]))
            if hit_frac > 0.3 and (best is None or hit_frac > best[0]):
                best = (hit_frac, p)

        record = {
            "video": os.path.basename(video), "shot": list(shot),
            "track_name": name, "seed_frame": frame0,
            "n_tracked_frames": len(live), "timestamp": time.time(),
        }
        if best:
            hit_frac, p = best
            record.update({
                "detector_found_it": True, "hit_fraction": round(hit_frac, 2),
                "proposal_persistence": p.persistence,
                "proposal_median_area": round(p.median_area, 1),
                "proposal_displacement": round(p.independent_displacement, 1),
            })
        else:
            record["detector_found_it"] = False
        records.append(record)

    if records:
        with open(DETECTOR_LOG, "a") as fh:
            for r in records:
                fh.write(json.dumps(r) + "\n")


# ---------------------------------------------------------------------------
# output CSV (real track names, timeline-aligned across the whole video)
# ---------------------------------------------------------------------------

def build_shot_csv_lines(clicks, tracks, shot_start, fps, picture):
    """CSV data rows for one shot's tracked objects -- no file I/O.

    Pure function returning the lines rather than writing them directly, so a
    shot's contribution to the export can be stored, and later removed and
    the whole file rebuilt without it -- see rebuild_csv_from_shots and
    revert_shot. Append-only writing (the original design) cannot support
    that: once a row is on disk there is no clean way to take it back.
    """
    y0, y1, x0, x1 = picture
    width, height = max(1, x1 - x0), max(1, y1 - y0)
    rows = []
    for oid, (frame0, cx0, cy0, name) in enumerate(clicks, start=1):
        live = {f: v for f, v in tracks.get(oid, {}).items() if v[0] > 0}
        if not live:
            print(f"  ! '{name}': SAM 2 produced no mask anywhere -- dropped, click again")
            continue
        peak = max(v[0] for v in live.values())
        for f in sorted(live):
            area, cx, cy = live[f][0], live[f][1], live[f][2]
            x = float(np.clip(((cx - x0) / width) * 2 - 1, -1, 1))
            z = float(np.clip(1 - ((cy - y0) / height) * 2, -1, 1))
            depth = float(np.clip(1 - np.sqrt(area) / np.sqrt(peak), -1, 1))
            t = (shot_start + f) / fps
            rows.append(f"{name},{t:.3f},{x:.3f},{depth:.3f},{z:.3f}")
    return rows


def rebuild_csv_from_shots(out_csv, content, shot_csv_lines):
    """Rewrite the whole CSV from each shot's stored lines.

    Order doesn't affect correctness -- export_tracked_adm.py groups by
    object and sorts by time internally regardless of row order in the file
    -- but writing in the video's own shot order keeps the file itself
    readable if anyone opens it directly.

    If nothing is committed anywhere, the file is removed rather than left
    behind holding just a header: downstream code treats "no CSV" as "no
    objects marked", and a header-only file would silently break that check.
    """
    rows = []
    for a, b in content:
        rows.extend(shot_csv_lines.get(f"{a}-{b}", []))

    if not rows:
        if os.path.exists(out_csv):
            os.remove(out_csv)
        return 0

    with open(out_csv, "w") as fh:
        fh.write("object,time_sec,x,y_depth,z_height\n")
        fh.write("\n".join(rows) + "\n")
    return len(rows)


def revert_shot(state, content, out_csv, out_wav, shot_key):
    """Undo a previously committed shot: forget it was ever handled, remove
    its exact CSV contribution, and rebuild the export from what remains.

    Used by the 'go back' action -- redoing a shot must not leave its old
    objects sitting in the file alongside the new ones, which is exactly
    what would happen if the old append-only writer were re-run for it.
    """
    state["done"].pop(shot_key, None)
    removed = state.setdefault("shot_objects", {}).pop(shot_key, [])
    state.setdefault("shot_csv_lines", {}).pop(shot_key, None)
    state["objects"] = [n for names in state["shot_objects"].values() for n in names]

    rebuild_csv_from_shots(out_csv, content, state["shot_csv_lines"])
    if os.path.exists(out_csv):
        export_wav(out_csv, out_wav, quiet=True)
    elif os.path.exists(out_wav):
        os.remove(out_wav)          # nothing left to export -- don't leave a stale WAV
    return removed


def export_wav(out_csv, out_wav, quiet=False):
    """Rebuild the ADM WAV from the CSV as it stands right now.

    Called after every shot, not only once every remaining shot is handled --
    on a 42-shot trailer, requiring a full pass before anything is importable
    left the first real session with a tracked object and no way to get it
    into Pro Tools without asking for help. Re-exporting is cheap (under a
    second for a handful of objects) and overwrites cleanly each time, so
    there's no cost to keeping the WAV current after every commit.
    """
    if not os.path.exists(out_csv):
        return False
    result = subprocess.run(
        [sys.executable, "export_tracked_adm.py", out_csv, out_wav],
        cwd=os.path.dirname(os.path.abspath(__file__)),
        stdout=subprocess.DEVNULL if quiet else None,
        stderr=subprocess.DEVNULL if quiet else None)
    if result.returncode == 0:
        if not quiet:
            print(f"  -> {out_wav} updated, ready to import into Pro Tools")
        return True
    print(f"  ! export_tracked_adm.py failed (exit {result.returncode}) -- "
          f"the CSV is still at {out_csv}, safe to export by hand")
    return False


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    # --reset can appear anywhere on the command line, ahead of the two
    # positional arguments.
    argv = [a for a in sys.argv[1:] if a != "--reset"]
    reset = len(argv) != len(sys.argv) - 1
    video = argv[0]
    names_file = argv[1] if len(argv) > 1 else None
    known_names = load_track_names(names_file)

    if reset and os.path.exists(session_path(video)):
        os.remove(session_path(video))
        print(f"  --reset: cleared the previous session for this video\n")

    out_csv = os.path.splitext(video)[0] + "_manual_track.csv"
    out_wav = os.path.splitext(video)[0] + "_manual_track.wav"

    print(f"segmenting {os.path.basename(video)} ...")
    shots, fps, n_frames = detect_shots(video)
    content = [s for s in shots if s[1] - s[0] + 1 >= MIN_CONTENT_FRAMES]
    print(f"  {len(shots)} shots, {len(content)} after dropping leader/title-card fragments\n")

    state = load_session(video, content, fps)
    done_map = state["done"]
    remaining = [s for s in content if f"{s[0]}-{s[1]}" not in done_map]

    # A real, honest summary instead of a terse "already handled" -- this is
    # the thing that was missing when every shot came back marked skipped
    # with zero objects ever recorded: nothing in the tool could tell you
    # that at a glance, so "process is complete" looked like a dead end
    # instead of what it actually was, a stale test session.
    n_done = sum(1 for v in done_map.values() if v == "done")
    n_skipped = sum(1 for v in done_map.values() if v == "skipped")
    if done_map:
        print(f"  session so far: {n_done} shot(s) with objects marked, "
              f"{n_skipped} skipped, {len(state['objects'])} object(s) total")
    if len(remaining) < len(content):
        print(f"  {len(content) - len(remaining)} shot(s) already handled, "
              f"{len(remaining)} remaining\n")

    if not remaining:
        if not state["objects"]:
            print("\nEvery shot in this video is already marked 'skipped' and no "
                  "objects were ever added -- likely a stale session from testing, "
                  "not real work. Rerun with --reset to start over:")
            print(f'  python click_and_pan.py --reset "{video}"'
                  + (f' "{names_file}"' if names_file else ""))
            return
        print("all shots already handled. Exporting ...")
    else:
        print("loading SAM 2 (once, reused for every shot) ...")
        predictor = build_predictor()

        # Walking an INDEX into the full `content` list, not the pre-filtered
        # `remaining` list, is what makes 'go back' possible: it can just move
        # the pointer backward and let the loop re-enter an earlier shot,
        # rather than needing a separate mechanism to jump around.
        idx = 0
        while idx < len(content) and f"{content[idx][0]}-{content[idx][1]}" in done_map:
            idx += 1                                # skip what's already handled

        while idx < len(content):
            a, b = content[idx]
            key = f"{a}-{b}"
            if key in done_map:                     # already handled -- move on
                idx += 1
                continue

            n_done_now = sum(1 for v in done_map.values() if v in ("done", "skipped"))
            label = f"[{n_done_now+1}/{len(content)}] shot {a}-{b} ({a/fps:.1f}s, {b-a+1}fr)"
            print(f"\n{label}")
            frames = read_shot_frames(video, a, b)

            quitting = False
            advance = True                          # False only on 'back'
            while True:                              # 'redo' loops back, same shot
                editor = ShotEditor(frames, label, fps, a)
                action, clicks = editor.run(known_names)

                if action == "back":
                    if idx == 0:
                        print("  already at the first shot -- nothing to go back to")
                        continue                     # fresh ShotEditor, same shot
                    prev_a, prev_b = content[idx - 1]
                    prev_key = f"{prev_a}-{prev_b}"
                    removed = revert_shot(state, content, out_csv, out_wav, prev_key)
                    save_session(video, state)
                    if removed:
                        print(f"  reverted shot {prev_key}: removed {', '.join(removed)} "
                              f"-- reopening it")
                    else:
                        print(f"  reopening shot {prev_key} (was skipped, nothing to remove)")
                    idx -= 1
                    advance = False
                    break

                if action == "skip":
                    done_map[key] = "skipped"
                    state["shot_objects"][key] = []
                    save_session(video, state)
                    break

                if clicks:
                    print(f"  tracking {len(clicks)} object(s) with SAM 2 ...")
                    working = _ProcessingWindow(f"Tracking {len(clicks)} object(s) with SAM 2...")
                    seeds = [ManualSeed(f, x, y) for f, x, y, _ in clicks]
                    tracks = track_proposals(frames, seeds, predictor=predictor, verbose=False)
                    working.close()

                    names = {oid: c[3] for oid, c in enumerate(clicks, start=1)}
                    print("  reviewing -- watch the real tracked position; "
                          "'c' to accept, 'r' to redo this shot's clicks")
                    verdict = TrackReview(frames, label, fps, a, tracks, names).run()

                    if verdict == "redo":
                        print("  redoing this shot's clicks\n")
                        continue                    # fresh ShotEditor, same shot

                    picture = picture_bounds(frames)
                    lines = build_shot_csv_lines(clicks, tracks, a, fps, picture)
                    state["shot_csv_lines"][key] = lines
                    state["shot_objects"][key] = [c[3] for c in clicks]
                    state["objects"] = [n for names_ in state["shot_objects"].values()
                                        for n in names_]
                    rebuild_csv_from_shots(out_csv, content, state["shot_csv_lines"])
                    print(f"  wrote {len(lines)} keyframes for: "
                          f"{', '.join(c[3] for c in clicks)}")

                    # Rebuild the WAV now, not only at the very end -- this is
                    # what makes the tool importable after any single shot
                    # instead of forcing a full pass through everything first.
                    export_wav(out_csv, out_wav)

                    # Quiet background check -- never shown, just logged.
                    proposals = propose_for_shot(frames)
                    log_detector_agreement(video, (a, b), clicks, tracks, proposals, frames)

                if action == "quit":
                    done_map[key] = "done" if clicks else "skipped"
                    if clicks:
                        state["shot_objects"][key] = [c[3] for c in clicks]
                    save_session(video, state)
                    quitting = True
                    break

                done_map[key] = "done"
                save_session(video, state)
                break

            if quitting:
                print(f"\nsaved. resume any time with the same command:\n"
                      f"  python click_and_pan.py \"{video}\""
                      + (f" \"{names_file}\"" if names_file else ""))
                return

            if advance:
                idx += 1
                if idx >= len(content):
                    # Loud and impossible to miss, since reaching this point
                    # silently is exactly what "finished one shot and it
                    # acted like the whole project was done" looks like from
                    # the outside -- whether that was really the last shot,
                    # or several were skipped past faster than expected.
                    n_obj = sum(1 for v in done_map.values() if v == "done")
                    n_skip = sum(1 for v in done_map.values() if v == "skipped")
                    print(f"\n{'='*60}")
                    print(f"  THAT WAS THE LAST SHOT ({n_obj} with objects, "
                          f"{n_skip} skipped, {len(state['objects'])} object(s) total)")
                    print(f"  -> exporting now")
                    print(f"{'='*60}")

    if not os.path.exists(out_csv):
        print("\nno objects were marked in this video -- nothing to export.")
        return

    print(f"\nall shots handled. exporting {out_csv} -> {out_wav} ...")
    if export_wav(out_csv, out_wav, quiet=True):
        print(f"\nready to import into Pro Tools: {out_wav}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    main()
