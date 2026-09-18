"""
Export tracked object motion (X / depth / height) for one or more objects as
a BWF .wav file carrying real ADM (Audio Definition Model, ITU-R BS.2076)
object metadata -- the same structure the Dolby Atmos Renderer itself writes
when you bounce a "WAV (Dolby Atmos)" print master from Pro Tools.

VERIFIED WORKING: this structure has been round-tripped through a real Pro
Tools import -- position automation for a single object showed up correctly
in both the Dolby Atmos Renderer's 3D view and Pro Tools' own automation
lanes (all three axes confirmed). The RIFF chunk layout (fmt/data/axml/chna)
and the ADM XML element chain were reverse-engineered from a real Pro
Tools ADM bounce read back byte-for-byte.

INPUT: a CSV file in "long" format, one row per keyframe -- matching the
tracking pipeline's own output shape (see 35_reconstructed_3d.csv):
    object,time_sec,x,y_depth,z_height[,match_confidence]
    TrackedObject,30.542,-0.81,0.0,-0.46
    TrackedObject,30.792,-0.86,0.0,-0.41
    ...
Multiple objects are supported -- just use a different `object` name per
row, in any order (rows are grouped and time-sorted per object automatically).
Any extra columns (e.g. match_confidence) are ignored. Files from earlier
pipeline stages with no `object` column are treated as a single object,
named after the input filename.
Each distinct object becomes its own mono PCM channel plus its own
audioObject/audioPackFormat/audioChannelFormat/audioTrackUID chain in the
ADM metadata, all packed into one interleaved multichannel WAV.

OUTPUT: a BWF .wav with N audible test-tone mono PCM channels (a different
tone per object so they're distinguishable by ear -- deliberately audible,
not silent, so a "no signal" result on import means something instead of
being indistinguishable from working-as-designed silence) and a real `axml`
ADM metadata chunk describing each object's X/Y/Z trajectory over time, plus
the `chna` chunk tying the metadata to each embedded audio channel.

TRACK MAPPING: the tracking pipeline names objects generically (e.g.
obj_0_(top182_rear42)), which has no relationship to the real Pro Tools
track names an existing production session already uses (e.g. "Fx Object
17"). An optional mapping CSV bridges the two:
    tracked_object,track_name
    obj_0_(top182_rear42),Fx Object 17
    obj_1_(top182_rear63),Fx Object 18
Only tracked objects present in the mapping are exported (with their name
replaced by the real track name), and anything not listed is skipped with a
warning -- this is deliberate, since an unmapped/misnamed object would just
create a stray "New Track" on import instead of landing on the audio it's
meant to pan. This mapping is inherently a human decision (which tracked
object in the footage corresponds to which already-recorded audio track)
and has to be supplied, not inferred.

USAGE:
    python export_tracked_adm.py [input.csv] [output.wav] [mapping.csv]
    (defaults to tracked_data.csv -> tracked_object_atmos.wav, no mapping)
(no third-party libraries needed -- this writes RIFF/ADM bytes directly)
"""

import csv
import math
import os
import struct
import sys

SAMPLE_RATE = 48000
BIT_DEPTH = 24
DEFAULT_INPUT = "tracked_data.csv"
DEFAULT_OUTPUT = "tracked_object_atmos.wav"

# Real tracked data is sampled once per source-video frame (confirmed
# uniform 1/24s spacing against the real pipeline output), so raw
# frame-to-frame jitter turns directly into sharp, jagged automation.
# This is a time-based moving-average window (not a frame count), so it
# behaves consistently regardless of the source clip's frame rate.
SMOOTHING_WINDOW_SECONDS = 0.3


def load_curve_data(csv_path):
    """
    Reads the tracking pipeline's long-format CSV (matching
    35_reconstructed_3d.csv: object,time_sec,x,y_depth,z_height[,match_confidence])
    and groups rows by object, sorted by time. Preserves first-seen object
    order so channel/track indices are stable from run to run. Files with
    no `object` column (single-object stages like 4_automation_data.csv)
    fall back to naming the one object after the input filename.
    """
    objects = {}
    order = []

    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        has_object_col = "object" in (reader.fieldnames or [])
        fallback_name = os.path.splitext(os.path.basename(csv_path))[0]

        for row in reader:
            name = row["object"].strip() if has_object_col else fallback_name
            if name not in objects:
                objects[name] = []
                order.append(name)
            objects[name].append((
                float(row["time_sec"]),
                float(row["x"]),
                float(row["y_depth"]),
                float(row["z_height"]),
            ))

    curves = {}
    for name in order:
        rows = sorted(objects[name], key=lambda r: r[0])
        curves[name] = {
            "times": [r[0] for r in rows],
            "xs": [r[1] for r in rows],
            "ys": [r[2] for r in rows],
            "zs": [r[3] for r in rows],
        }
    return curves


def smooth_curve(times, values, window_seconds):
    """
    Symmetric time-windowed moving average. Two-pointer sliding window, so
    it's O(n) even though it's re-run per axis per object. `times` must
    already be sorted ascending (load_curve_data guarantees this).
    """
    if window_seconds <= 0:
        return list(values)

    n = len(times)
    half = window_seconds / 2.0
    smoothed = [0.0] * n
    lo = 0
    hi = 0
    running_sum = 0.0
    count = 0

    for i in range(n):
        t = times[i]
        while hi < n and times[hi] <= t + half:
            running_sum += values[hi]
            count += 1
            hi += 1
        while lo < n and times[lo] < t - half:
            running_sum -= values[lo]
            count -= 1
            lo += 1
        smoothed[i] = running_sum / count

    return smoothed


def smooth_curves(curves, window_seconds):
    for name, c in curves.items():
        c["xs"] = smooth_curve(c["times"], c["xs"], window_seconds)
        c["ys"] = smooth_curve(c["times"], c["ys"], window_seconds)
        c["zs"] = smooth_curve(c["times"], c["zs"], window_seconds)
    return curves


def load_track_mapping(mapping_path):
    """
    Reads the tracked_object -> track_name mapping CSV. Returns a dict
    preserving file order.
    """
    mapping = {}
    with open(mapping_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            mapping[row["tracked_object"].strip()] = row["track_name"].strip()
    return mapping


def apply_track_mapping(curves, mapping):
    """
    Renames curves from generic tracked-object names to the real track
    names they should land on, in mapping-file order. Anything tracked but
    not listed in the mapping is dropped with a warning rather than passed
    through -- an unmapped name would just create a stray "New Track" on
    import instead of landing on the audio it's meant to pan.
    """
    mapped = {}
    for tracked_name, track_name in mapping.items():
        if tracked_name in curves:
            mapped[track_name] = curves[tracked_name]
        else:
            print(f"  warning: mapping lists '{tracked_name}' but it wasn't "
                  f"found in the tracking data, skipping")

    for tracked_name in curves:
        if tracked_name not in mapping:
            print(f"  warning: tracked object '{tracked_name}' has no entry "
                  f"in the mapping file, skipping (won't be exported)")

    return mapped


def clamp_axis(object_name, label, value):
    if -1.0 <= value <= 1.0:
        return value
    clamped = max(-1.0, min(1.0, value))
    print(f"  warning: [{object_name}] {label}={value:.3f} outside ADM's "
          f"[-1, 1] room-centric range, clamping to {clamped:.3f}")
    return clamped


def adm_timecode(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:08.5f}"


# Gap between consecutive keyframes of the SAME object beyond which they are
# treated as two separate appearances (e.g. a recurring character re-clicked
# in a later, non-adjacent shot) rather than one continuous move. Real
# per-frame keyframe spacing runs ~10-40ms; a gap in whole seconds means the
# object was off-screen or untracked in between, not gliding across the room.
GAP_HOLD_THRESHOLD = 1.0

# Near-instant snap once an object reappears after a gap, rather than a
# literal zero-duration block -- matches how a real Pro Tools bounce
# represents an abrupt position change (block durations as short as 0.00015s
# were observed reading a hand-mixed reference file earlier in this project).
GAP_JUMP_DURATION = 0.01


def build_block_formats(object_name, times, xs, ys, zs, total_duration):
    """
    Holds the object still at its first tracked position for any pre-roll
    before its first keyframe, ramps linearly (interpolationLength == the
    full segment duration) between tracked points, and holds at its last
    position for any post-roll after its last keyframe -- so every object's
    channel covers the full shared `total_duration`, even if its own tracked
    data spans a shorter window.

    A gap longer than GAP_HOLD_THRESHOLD between two consecutive keyframes is
    NOT ramped across. The original version ramped every internal segment
    unconditionally, which is correct for real per-frame motion but wrong for
    two separate appearances of the same named object -- clicking "glass" in
    one shot ending at 19s and again in a later shot starting at 55s produced
    a single 36-second linear pan sliding between those positions, sweeping
    through time when the object was not on screen or moving at all. A long
    gap instead holds at the prior position, then jumps to the next one.
    """
    blocks = []

    x0 = clamp_axis(object_name, "X", xs[0])
    y0 = clamp_axis(object_name, "Y", ys[0])
    z0 = clamp_axis(object_name, "Z", zs[0])

    if times[0] > 0:
        blocks.append({"rtime": 0.0, "duration": times[0],
                        "x": x0, "y": y0, "z": z0, "interp": 0.0})

    for i in range(1, len(times)):
        seg_duration = times[i] - times[i - 1]
        cur_x = clamp_axis(object_name, "X", xs[i])
        cur_y = clamp_axis(object_name, "Y", ys[i])
        cur_z = clamp_axis(object_name, "Z", zs[i])

        if seg_duration > GAP_HOLD_THRESHOLD:
            prev_x = clamp_axis(object_name, "X", xs[i - 1])
            prev_y = clamp_axis(object_name, "Y", ys[i - 1])
            prev_z = clamp_axis(object_name, "Z", zs[i - 1])
            hold_dur = seg_duration - GAP_JUMP_DURATION
            blocks.append({"rtime": times[i - 1], "duration": hold_dur,
                            "x": prev_x, "y": prev_y, "z": prev_z, "interp": 0.0})
            blocks.append({"rtime": times[i - 1] + hold_dur, "duration": GAP_JUMP_DURATION,
                            "x": cur_x, "y": cur_y, "z": cur_z, "interp": 0.0})
        else:
            blocks.append({
                "rtime": times[i - 1],
                "duration": seg_duration,
                "x": cur_x, "y": cur_y, "z": cur_z,
                "interp": seg_duration,
            })

    if times[-1] < total_duration:
        blocks.append({
            "rtime": times[-1],
            "duration": total_duration - times[-1],
            "x": clamp_axis(object_name, "X", xs[-1]),
            "y": clamp_axis(object_name, "Y", ys[-1]),
            "z": clamp_axis(object_name, "Z", zs[-1]),
            "interp": 0.0,
        })

    return blocks


class ObjectIds:
    """
    ID scheme verified against a real Pro Tools ADM bounce: AO_ uses a
    0x1000-based hex offset, AP_/AC_/AT_/AS_ share a "0003" (Objects
    typeLabel) prefix plus the same hex offset, and ATU_ is an 8-hex-digit
    track index -- all of which fit the chna chunk's fixed field widths
    (12/14/11 bytes) exactly, confirmed by round-tripping the real file.
    """
    def __init__(self, index):
        self.index = index
        suffix = f"{0x1000 + index:04x}"
        self.ao = f"AO_{suffix}"
        self.ap = f"AP_0003{suffix}"
        self.ac = f"AC_0003{suffix}"
        self.at = f"AT_0003{suffix}_01"
        self.as_ = f"AS_0003{suffix}"
        self.atu = f"ATU_{index:08x}"


def build_axml(objects_data, total_duration):
    total_tc = adm_timecode(total_duration)

    programme_refs = "\n".join(
        f'\t\t\t\t\t<audioObjectIDRef>{ids.ao}</audioObjectIDRef>'
        for _, ids, _ in objects_data
    )

    sections = []
    for name, ids, blocks in objects_data:
        block_xml = "\n".join(
            f'\t\t\t\t\t<audioBlockFormat audioBlockFormatID="AB_{ids.index:04x}_{i:08d}" '
            f'rtime="{adm_timecode(b["rtime"])}" duration="{adm_timecode(b["duration"])}">\n'
            f'\t\t\t\t\t\t<cartesian>1</cartesian>\n'
            f'\t\t\t\t\t\t<position coordinate="X">{b["x"]:.10f}</position>\n'
            f'\t\t\t\t\t\t<position coordinate="Y">{b["y"]:.10f}</position>\n'
            f'\t\t\t\t\t\t<position coordinate="Z">{b["z"]:.10f}</position>\n'
            f'\t\t\t\t\t\t<jumpPosition interpolationLength="{b["interp"]:.6f}">1</jumpPosition>\n'
            f'\t\t\t\t\t</audioBlockFormat>'
            for i, b in enumerate(blocks, start=1)
        )

        sections.append(f'''\t\t\t\t<audioObject audioObjectID="{ids.ao}" audioObjectName="{name}" start="00:00:00.00000" duration="{total_tc}">
\t\t\t\t\t<audioPackFormatIDRef>{ids.ap}</audioPackFormatIDRef>
\t\t\t\t\t<audioTrackUIDRef>{ids.atu}</audioTrackUIDRef>
\t\t\t\t</audioObject>
\t\t\t\t<audioPackFormat audioPackFormatID="{ids.ap}" audioPackFormatName="{name}" typeDefinition="Objects" typeLabel="0003">
\t\t\t\t\t<audioChannelFormatIDRef>{ids.ac}</audioChannelFormatIDRef>
\t\t\t\t</audioPackFormat>
\t\t\t\t<audioChannelFormat audioChannelFormatID="{ids.ac}" audioChannelFormatName="{name}" typeDefinition="Objects" typeLabel="0003">
{block_xml}
\t\t\t\t</audioChannelFormat>
\t\t\t\t<audioTrackUID UID="{ids.atu}" bitDepth="{BIT_DEPTH}" sampleRate="{SAMPLE_RATE}">
\t\t\t\t\t<audioTrackFormatIDRef>{ids.at}</audioTrackFormatIDRef>
\t\t\t\t\t<audioPackFormatIDRef>{ids.ap}</audioPackFormatIDRef>
\t\t\t\t</audioTrackUID>
\t\t\t\t<audioTrackFormat audioTrackFormatID="{ids.at}" audioTrackFormatName="PCM_{name}" formatDefinition="PCM" formatLabel="0001">
\t\t\t\t\t<audioStreamFormatIDRef>{ids.as_}</audioStreamFormatIDRef>
\t\t\t\t</audioTrackFormat>
\t\t\t\t<audioStreamFormat audioStreamFormatID="{ids.as_}" audioStreamFormatName="PCM_{name}" formatDefinition="PCM" formatLabel="0001">
\t\t\t\t\t<audioChannelFormatIDRef>{ids.ac}</audioChannelFormatIDRef>
\t\t\t\t\t<audioPackFormatIDRef>{ids.ap}</audioPackFormatIDRef>
\t\t\t\t\t<audioTrackFormatIDRef>{ids.at}</audioTrackFormatIDRef>
\t\t\t\t</audioStreamFormat>''')

    all_sections = "\n".join(sections)

    return f'''<?xml version="1.0" encoding="UTF-8"?>
<ebuCoreMain xmlns="urn:ebu:metadata-schema:ebuCore_2016" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:schemaLocation="urn:ebu:metadata-schema:ebuCore_2016 ebucore.xsd" xml:lang="en">
\t<coreMetadata>
\t\t<format>
\t\t\t<audioFormatExtended>
\t\t\t\t<audioProgramme audioProgrammeID="APR_1001" audioProgrammeName="TrackedObjects" start="00:00:00.00000" end="{total_tc}">
\t\t\t\t\t<audioContentIDRef>ACO_1001</audioContentIDRef>
\t\t\t\t</audioProgramme>
\t\t\t\t<audioContent audioContentID="ACO_1001" audioContentName="TrackedObjects_Content">
{programme_refs}
\t\t\t\t</audioContent>
{all_sections}
\t\t\t</audioFormatExtended>
\t\t</format>
\t</coreMetadata>
</ebuCoreMain>
'''


def build_chna(objects_data):
    """
    One 40-byte entry per object per BS.2076-1: uint16 trackIndex, 12-byte
    UID, 14-byte trackFormatID, 11-byte packFormatID, 1 padding byte --
    field widths and order confirmed by reading a real Pro Tools ADM bounce
    byte-for-byte.
    """
    entries = bytearray()
    for _, ids, _ in objects_data:
        entry = struct.pack("<H", ids.index)
        entry += ids.atu.encode("ascii").ljust(12, b"\x00")
        entry += ids.at.encode("ascii").ljust(14, b"\x00")
        entry += ids.ap.encode("ascii").ljust(11, b"\x00")
        entry += b"\x00"
        assert len(entry) == 40
        entries += entry

    n = len(objects_data)
    header = struct.pack("<HH", n, n)  # numTracks, numUIDs
    return bytes(header) + bytes(entries)


def riff_chunk(chunk_id, data):
    padded = data + (b"\x00" if len(data) % 2 else b"")
    return chunk_id + struct.pack("<I", len(data)) + padded


def build_fmt_chunk(num_channels, sample_rate, bit_depth):
    byte_rate = sample_rate * num_channels * bit_depth // 8
    block_align = num_channels * bit_depth // 8
    fmt_data = struct.pack(
        "<HHIIHH",
        1,  # PCM
        num_channels,
        sample_rate,
        byte_rate,
        block_align,
        bit_depth,
    )
    return riff_chunk(b"fmt ", fmt_data)


def build_interleaved_tone_data_chunk(num_channels, duration_seconds, sample_rate,
                                       bit_depth, level_dbfs=-20.0):
    """
    A distinct, audible test tone per channel (rather than silence), so a
    "no signal" result on import means something instead of being
    indistinguishable from working-as-designed silence -- and a different
    frequency per object makes them tellable apart by ear.
    """
    num_samples = max(int(round(duration_seconds * sample_rate)), 1)
    bytes_per_sample = bit_depth // 8
    peak = (2 ** (bit_depth - 1) - 1) * (10 ** (level_dbfs / 20.0))
    freqs = [220.0 * (i + 1) for i in range(num_channels)]

    samples = bytearray()
    for n in range(num_samples):
        for freq in freqs:
            value = int(round(peak * math.sin(2 * math.pi * freq * n / sample_rate)))
            samples += value.to_bytes(bytes_per_sample, byteorder="little", signed=True)
    return riff_chunk(b"data", bytes(samples))


def main():
    input_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_INPUT
    output_path = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_OUTPUT
    mapping_path = sys.argv[3] if len(sys.argv) > 3 else None

    print(f"Reading {input_path} ...")
    curves = load_curve_data(input_path)
    print(f"  found {len(curves)} tracked object(s): {', '.join(curves.keys())}")

    if mapping_path:
        print(f"Applying track mapping from {mapping_path} ...")
        mapping = load_track_mapping(mapping_path)
        curves = apply_track_mapping(curves, mapping)
        if not curves:
            print("  no tracked objects matched the mapping file -- nothing to export.")
            return

    object_names = list(curves.keys())
    print(f"  exporting {len(object_names)} object(s) as: {', '.join(object_names)}")

    if SMOOTHING_WINDOW_SECONDS > 0:
        print(f"Smoothing curves ({SMOOTHING_WINDOW_SECONDS}s moving-average window) ...")
        curves = smooth_curves(curves, SMOOTHING_WINDOW_SECONDS)

    total_duration = max(c["times"][-1] for c in curves.values())

    objects_data = []
    for i, name in enumerate(object_names, start=1):
        c = curves[name]
        ids = ObjectIds(i)
        blocks = build_block_formats(name, c["times"], c["xs"], c["ys"], c["zs"], total_duration)
        objects_data.append((name, ids, blocks))
        print(f"  [{name}] track {i}: {len(blocks)} position blocks")

    print(f"Writing {output_path} ...")
    axml_xml = build_axml(objects_data, total_duration)
    chna_bytes = build_chna(objects_data)

    fmt_chunk = build_fmt_chunk(len(objects_data), SAMPLE_RATE, BIT_DEPTH)
    data_chunk = build_interleaved_tone_data_chunk(
        len(objects_data), total_duration, SAMPLE_RATE, BIT_DEPTH
    )
    axml_chunk = riff_chunk(b"axml", axml_xml.encode("utf-8"))
    chna_chunk = riff_chunk(b"chna", chna_bytes)

    body = b"WAVE" + fmt_chunk + data_chunk + axml_chunk + chna_chunk
    riff = b"RIFF" + struct.pack("<I", len(body)) + body

    with open(output_path, "wb") as f:
        f.write(riff)

    print(f"  {len(objects_data)} object(s), {total_duration:.3f}s total, "
          f"{SAMPLE_RATE}Hz/{BIT_DEPTH}-bit test tones (audible on purpose, "
          f"distinct frequency per object)")
    print("Done.")


if __name__ == "__main__":
    main()
