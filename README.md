
# Video-Driven Dolby Atmos Object Panning

> A pipeline that tracks moving objects in video and generates Dolby Atmos object-panning automation — then hands final placement judgment back to the mixer.

---

## What this is

In a Dolby Atmos mix, every object needs an X/Y/Z position and a movement path over time. When sound follows something visible on screen — a car crossing the frame, a character walking, a helicopter arcing overhead — that path is dictated by the picture, but a mixer still authors it by hand, keyframe by keyframe.

This project asks: how much of that authoring can be derived directly from the video, and how much *should* be?

It tracks a chosen object across a shot, converts its on-screen motion into Atmos object coordinates, and writes real ADM/BWF panning automation that imports cleanly into an existing Pro Tools session. Crucially, it does **not** try to be fully autonomous — see [Design decision](#design-decision-assisted-not-autonomous).

## How it works

The pipeline runs in stages:

1. **Shot / cut detection** — segments the video so tracking never bleeds across a hard cut (where a "moving object" is really two unrelated shots).
2. **Object tracking** — uses [SAM 2](https://ai.meta.com/sam2/) (Meta's Segment Anything Model 2), combined with depth and focus cues, to isolate the intended subject from the background and other moving characters and follow it through the shot, producing a per-frame position track.
3. **Coordinate mapping** — maps normalized on-screen position to Atmos object coordinates (screen X/Y -> room position, with configurable depth handling).
4. **ADM/BWF authoring** — writes the movement as real object-panning automation in an ADM BWF file (see below).
5. **Pro Tools import** — the resulting file imports into an existing session with the panning automation landing correctly on the target track.

## The hard part: hand-authored ADM/BWF metadata

The panning automation isn't produced through a convenience library — it's written at the metadata level. That meant:

- Working directly with the **`axml`** and **`chna`** chunk structures inside the BWF/WAV container, at the byte level.
- **Reverse-engineering the format against a real Pro Tools Atmos bounce** — bouncing a previous session, inspecting exactly how Pro Tools encodes object positions and movement, and matching that structure rather than guessing from spec alone.
- Producing files that **validate on import**: confirmed that the authored automation lands on existing tracks with the intended positions and paths, not just that the file opens without error.

## Design decision: assisted, not autonomous

The project originally aimed for *fully automatic* panning — video in, finished automation out. After building it, I concluded that full automation isn't trustworthy enough for production: tracking drifts, occlusion and re-identification are imperfect, and a wrong-but-confident pan is worse than no pan at all because it costs the mixer time to find and undo.

So it pivoted to a **point-and-click assisted tool** (`click_and_pan.py`): the operator picks the object and stays in the loop, and the tool does the tedious part — converting a confirmed track into valid Atmos automation. The mixer keeps authority over *what* moves and *how it should sound*; the tool removes the manual keyframing labor.

That trade-off — where automated spatial tooling genuinely helps versus where it has to stay firmly assistive — is the actual point of the project, more than the tracking itself.

## Tech

- **Object tracking:** SAM 2 (Meta), with depth and focus filters for subject isolation
- **Metadata:** hand-authored ADM/BWF (`axml` / `chna` chunks)
- **Target DAW:** Pro Tools + Dolby Atmos Renderer
- **Language:** Python

## Status

Working end to end: video -> tracked object -> validated Atmos automation -> confirmed Pro Tools import. Current focus is the assisted `click_and_pan.py` workflow rather than full automation.

## Getting started

The tool runs on Python 3.

```bash
pip install -r requirements.txt
pip install "git+https://github.com/facebookresearch/sam2.git"
```

Then download the SAM 2.1 checkpoint (`sam2.1_hiera_base_plus.pt`) from the
[SAM 2 repo](https://github.com/facebookresearch/sam2) and place it at
`models/sam2.1_hiera_base_plus.pt`.

Run the tool against a video file:

```bash
python click_and_pan.py path/to/your_video.mp4
```

It steps through the video shot by shot — click the object you want to track
in each one, and SAM 2 tracks it, you confirm or revert, and it writes the
Atmos object-panning automation to `Tracked Automation/`.

*Built by Hayden Gonzales.*
