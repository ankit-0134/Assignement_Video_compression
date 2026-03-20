"""
Smart Behavioral Video Compression
===================================
SentioMind Assignment — solution.py

Algorithm:
  Step 1 : pHash  — drop if > 95% similar to last kept frame
  Step 2 : Optical-flow motion score — discard if score < 0.05
  Step 3 : Haar face detection — always keep frames with a face
  Step 4 : Keep one context frame every 3 seconds minimum
  Step 5 : Re-encode surviving frames to H.264 MP4 @ 12 fps via ffmpeg

Author : <ANKIT_KUMAR_DUBEY_230154>
"""

import cv2
import json
import os
import subprocess
import sys
import tempfile
import time
from typing import Optional

import imagehash
import numpy as np
from PIL import Image

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────
PHASH_SIMILARITY_THRESHOLD = 0.95
OPTICAL_FLOW_THRESHOLD     = 0.05
CONTEXT_FRAME_INTERVAL_S   = 3.0
OUTPUT_FPS                 = 12
HAAR_SCALE_FACTOR          = 1.1
HAAR_MIN_NEIGHBORS         = 4
HAAR_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _phash_similarity(img_bgr, prev_hash):
    pil_img  = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    new_hash = imagehash.phash(pil_img)
    if prev_hash is None:
        return 0.0, new_hash
    distance   = prev_hash - new_hash
    max_bits   = len(new_hash.hash.flatten())
    similarity = 1.0 - (distance / max_bits)
    return similarity, new_hash


def _optical_flow_score(gray_prev, gray_curr):
    if gray_prev is None:
        return 1.0
    flow = cv2.calcOpticalFlowFarneback(
        gray_prev, gray_curr, None,
        pyr_scale=0.5, levels=3, winsize=15,
        iterations=3, poly_n=5, poly_sigma=1.2, flags=0
    )
    mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    return float(np.mean(mag))


def _detect_face(img_bgr, face_cascade):
    gray  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=HAAR_SCALE_FACTOR,
        minNeighbors=HAAR_MIN_NEIGHBORS,
        minSize=(30, 30)
    )
    return len(faces) > 0


# ──────────────────────────────────────────────────────────────────────────────
# Core pipeline functions (signatures fixed)
# ──────────────────────────────────────────────────────────────────────────────

def extract_intelligent_frames(video_path, segments_json_path="segments_kept.json"):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError("Cannot open video: " + video_path)

    fps          = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_s   = total_frames / fps

    face_cascade = cv2.CascadeClassifier(HAAR_CASCADE_PATH)

    kept_frames  = []
    prev_hash    = None
    prev_gray    = None
    last_kept_ts = -CONTEXT_FRAME_INTERVAL_S

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        timestamp_s = frame_idx / fps

        # Progress bar (updates every 10 frames)
        if frame_idx % 10 == 0 or frame_idx == total_frames - 1:
            pct    = (frame_idx / max(total_frames, 1)) * 100
            filled = int(pct / 2)
            bar    = "#" * filled + "-" * (50 - filled)
            mins   = int(timestamp_s // 60)
            secs   = int(timestamp_s % 60)
            sys.stdout.write(
                "\r  [{}] {:5.1f}%  frame {}/{}  time {:02d}:{:02d}  kept {}".format(
                    bar, pct, frame_idx, total_frames,
                    mins, secs, len(kept_frames)
                )
            )
            sys.stdout.flush()

        # Step 1: pHash similarity
        similarity, curr_hash = _phash_similarity(frame, prev_hash)
        if similarity > PHASH_SIMILARITY_THRESHOLD:
            frame_idx += 1
            continue

        # Step 2: Optical flow
        gray_curr    = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        motion_score = _optical_flow_score(prev_gray, gray_curr)

        # Step 3 & 4: Face detection + context frame
        has_face    = _detect_face(frame, face_cascade)
        context_due = (timestamp_s - last_kept_ts) >= CONTEXT_FRAME_INTERVAL_S

        keep   = False
        reason = []

        if has_face:
            keep = True
            reason.append("face_detected")

        if motion_score >= OPTICAL_FLOW_THRESHOLD:
            keep = True
            reason.append("motion")

        if context_due:
            keep = True
            reason.append("context_frame")

        if keep:
            kept_frames.append({
                "frame_index":      frame_idx,
                "timestamp_s":      round(timestamp_s, 4),
                "motion_score":     round(motion_score, 6),
                "has_face":         has_face,
                "phash_similarity": round(similarity, 6),
                "keep_reasons":     reason,
            })
            prev_hash    = curr_hash
            last_kept_ts = timestamp_s

        prev_gray = gray_curr
        frame_idx += 1

    sys.stdout.write("\n")  # newline after progress bar

    cap.release()

    output = {
        "source_video":      video_path,
        "total_frames":      total_frames,
        "kept_frames":       len(kept_frames),
        "source_fps":        fps,
        "source_duration_s": round(duration_s, 2),
        "segments":          kept_frames,
    }
    with open(segments_json_path, "w") as f:
        json.dump(output, f, indent=2)

    print("[extract] " + str(len(kept_frames)) + "/" + str(total_frames) +
          " frames kept -> " + segments_json_path)
    return kept_frames


def compress_video(video_path, segments_kept, output_path="compressed_output.mp4"):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError("Cannot open video: " + video_path)

    kept_indices = {seg["frame_index"] for seg in segments_kept}
    max_idx      = max(kept_indices) if kept_indices else 0

    with tempfile.TemporaryDirectory() as tmp_dir:
        img_pattern = os.path.join(tmp_dir, "frame_%06d.png")
        out_seq_idx = 0
        frame_idx   = 0

        while frame_idx <= max_idx:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx in kept_indices:
                out_name = "frame_" + str(out_seq_idx).zfill(6) + ".png"
                cv2.imwrite(os.path.join(tmp_dir, out_name), frame)
                out_seq_idx += 1
            frame_idx += 1

        cap.release()

        if out_seq_idx == 0:
            raise RuntimeError("No frames were kept — nothing to encode.")

        ffmpeg_cmd = [
            "ffmpeg", "-y",
            "-framerate", str(OUTPUT_FPS),
            "-i", img_pattern,
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "23",
            "-pix_fmt", "yuv420p",
            output_path
        ]
        result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError("ffmpeg failed:\n" + result.stderr)

    print("[compress] Output saved -> " + output_path)
    return output_path


def generate_report(source_path, output_path, segments_kept,
                    report_path="compression_report.html"):
    source_size = os.path.getsize(source_path) / (1024 * 1024)
    output_size = os.path.getsize(output_path) / (1024 * 1024)
    reduction   = (1 - output_size / source_size) * 100 if source_size else 0

    total_frames = max((s["frame_index"] for s in segments_kept), default=0) + 1
    kept_count   = len(segments_kept)
    face_count   = sum(1 for s in segments_kept if s["has_face"])
    motion_count = sum(1 for s in segments_kept if "motion" in s["keep_reasons"])

    # Per-second motion timeline
    timeline = {}
    for s in segments_kept:
        sec = int(s["timestamp_s"])
        if sec not in timeline:
            timeline[sec] = []
        timeline[sec].append(s["motion_score"])

    chart_labels = json.dumps(list(timeline.keys()))
    chart_data   = json.dumps([round(sum(v) / len(v), 4) for v in timeline.values()])

    # Build table rows as plain string (no backslash-in-f-string issue)
    def tag_html(r):
        if r == "face_detected":
            css = "face"
        elif r == "context_frame":
            css = "ctx"
        else:
            css = "motion"
        return '<span class="tag ' + css + '">' + r + "</span>"

    table_rows = ""
    for i, s in enumerate(segments_kept[:100]):
        tags = "".join(tag_html(r) for r in s["keep_reasons"])
        table_rows += (
            "<tr><td>" + str(i + 1) + "</td>"
            "<td>" + str(round(s["timestamp_s"], 2)) + "</td>"
            "<td>" + str(round(s["motion_score"], 4)) + "</td>"
            "<td>" + tags + "</td></tr>\n"
        )

    src_mb_str  = "{:.1f}".format(source_size)
    out_mb_str  = "{:.1f}".format(output_size)
    reduc_str   = "{:.1f}".format(reduction)
    face_pct    = "{:.1f}".format((face_count / max(kept_count, 1)) * 100)
    motion_pct  = "{:.1f}".format((motion_count / max(kept_count, 1)) * 100)

    html = (
        "<!DOCTYPE html>\n"
        '<html lang="en"><head>\n'
        '<meta charset="UTF-8">\n'
        "<title>Compression Report - SentioMind</title>\n"
        "<style>\n"
        "* { box-sizing: border-box; margin: 0; padding: 0; }\n"
        "body { font-family: 'Segoe UI', sans-serif; background: #0f1117; color: #e0e0e0; padding: 2rem; }\n"
        "h1 { color: #4fc3f7; margin-bottom: 0.5rem; }\n"
        "h2 { color: #81d4fa; margin: 2rem 0 1rem; border-bottom: 1px solid #333; padding-bottom: 0.5rem; }\n"
        ".grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px,1fr)); gap:1rem; margin:1rem 0; }\n"
        ".card { background:#1a1d2e; border-radius:10px; padding:1.5rem; text-align:center; }\n"
        ".card .val { font-size:2rem; font-weight:bold; color:#4fc3f7; }\n"
        ".card .lbl { color:#aaa; font-size:0.85rem; margin-top:0.4rem; }\n"
        ".bar-wrap { background:#222; border-radius:4px; height:20px; margin:0.5rem 0; overflow:hidden; }\n"
        ".bar { height:100%; border-radius:4px; }\n"
        ".progress-row { margin:0.7rem 0; }\n"
        ".progress-row span { font-size:0.85rem; color:#aaa; }\n"
        "canvas { max-width:100%; }\n"
        "table { width:100%; border-collapse:collapse; font-size:0.85rem; }\n"
        "th { background:#1a1d2e; color:#4fc3f7; padding:0.6rem; text-align:left; }\n"
        "td { padding:0.5rem; border-bottom:1px solid #222; }\n"
        "tr:hover td { background:#1e2130; }\n"
        ".tag { display:inline-block; padding:2px 6px; border-radius:4px; font-size:0.75rem; margin:1px; }\n"
        ".face   { background:#1b5e20; color:#a5d6a7; }\n"
        ".motion { background:#0d47a1; color:#90caf9; }\n"
        ".ctx    { background:#4a148c; color:#ce93d8; }\n"
        "</style></head><body>\n"
        "<h1>Smart Behavioral Video Compression</h1>\n"
        '<p style="color:#aaa">SentioMind Assignment Report</p>\n'
        "<h2>Size Comparison</h2>\n"
        '<div class="grid">\n'
        '  <div class="card"><div class="val">' + src_mb_str + ' MB</div><div class="lbl">Original Size</div></div>\n'
        '  <div class="card"><div class="val">' + out_mb_str + ' MB</div><div class="lbl">Compressed Size</div></div>\n'
        '  <div class="card"><div class="val">' + reduc_str + '%</div><div class="lbl">Size Reduction</div></div>\n'
        '  <div class="card"><div class="val">' + str(kept_count) + "/" + str(total_frames) + '</div><div class="lbl">Frames Kept</div></div>\n'
        "</div>\n"
        "<h2>Frame Selection Breakdown</h2>\n"
        '<div class="progress-row"><span>Faces Detected (' + str(face_count) + ' frames)</span>\n'
        '<div class="bar-wrap"><div class="bar" style="width:' + face_pct + '%;background:#4caf50"></div></div></div>\n'
        '<div class="progress-row"><span>Motion Frames (' + str(motion_count) + ' frames)</span>\n'
        '<div class="bar-wrap"><div class="bar" style="width:' + motion_pct + '%;background:#2196f3"></div></div></div>\n'
        "<h2>Motion Score Timeline</h2>\n"
        '<canvas id="motionChart" height="120"></canvas>\n'
        "<h2>Kept Frame Log (first 100)</h2>\n"
        "<table><thead><tr><th>#</th><th>Time (s)</th><th>Motion Score</th><th>Reasons</th></tr></thead>\n"
        "<tbody>\n"
        + table_rows +
        "</tbody></table>\n"
        "<script>\n"
        "(function(){\n"
        "  var labels=" + chart_labels + ";\n"
        "  var data=" + chart_data + ";\n"
        "  var canvas=document.getElementById('motionChart');\n"
        "  var ctx=canvas.getContext('2d');\n"
        "  canvas.width=canvas.parentElement.clientWidth||900;\n"
        "  canvas.height=200;\n"
        "  var W=canvas.width,H=canvas.height;\n"
        "  var pad={t:20,r:20,b:40,l:50};\n"
        "  var maxVal=Math.max.apply(null,data)||1;\n"
        "  ctx.fillStyle='#1a1d2e';ctx.fillRect(0,0,W,H);\n"
        "  ctx.strokeStyle='#333';ctx.lineWidth=1;\n"
        "  for(var g=0;g<=5;g++){\n"
        "    var y=pad.t+(H-pad.t-pad.b)*(1-g/5);\n"
        "    ctx.beginPath();ctx.moveTo(pad.l,y);ctx.lineTo(W-pad.r,y);ctx.stroke();\n"
        "    ctx.fillStyle='#aaa';ctx.font='11px sans-serif';\n"
        "    ctx.fillText((maxVal*g/5).toFixed(3),4,y+4);\n"
        "  }\n"
        "  var ty=pad.t+(H-pad.t-pad.b)*(1-0.05/maxVal);\n"
        "  ctx.strokeStyle='#f44336';ctx.setLineDash([4,4]);\n"
        "  ctx.beginPath();ctx.moveTo(pad.l,ty);ctx.lineTo(W-pad.r,ty);ctx.stroke();\n"
        "  ctx.setLineDash([]);\n"
        "  var bw=Math.max(2,(W-pad.l-pad.r)/data.length-1);\n"
        "  data.forEach(function(v,i){\n"
        "    var x=pad.l+i*(W-pad.l-pad.r)/data.length;\n"
        "    var bh=(v/maxVal)*(H-pad.t-pad.b);\n"
        "    ctx.fillStyle=v<0.05?'#555':'#4fc3f7';\n"
        "    ctx.fillRect(x,H-pad.b-bh,bw,bh);\n"
        "  });\n"
        "  ctx.fillStyle='#aaa';ctx.font='11px sans-serif';\n"
        "  for(var i=0;i<labels.length;i+=Math.max(1,Math.floor(labels.length/10))){\n"
        "    var x=pad.l+i*(W-pad.l-pad.r)/data.length;\n"
        "    ctx.fillText(labels[i]+'s',x,H-10);\n"
        "  }\n"
        "})();\n"
        "</script>\n"
        "</body></html>\n"
    )

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html)

    print("[report] HTML report saved -> " + report_path)
    return report_path


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) < 2:
        print("Usage: python solution.py <path_to_video> [output.mp4]")
        sys.exit(1)

    video_path  = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else "compressed_output.mp4"

    t0 = time.time()
    print("=" * 60)
    print(" SentioMind - Smart Behavioral Video Compression")
    print("=" * 60)

    print("\n[1/3] Extracting intelligent frames ...")
    segments = extract_intelligent_frames(video_path, "segments_kept.json")

    print("\n[2/3] Compressing video -> " + output_path + " ...")
    compress_video(video_path, segments, output_path)

    print("\n[3/3] Generating HTML report ...")
    generate_report(video_path, output_path, segments, "compression_report.html")

    elapsed   = time.time() - t0
    src_mb    = os.path.getsize(video_path) / (1024 * 1024)
    out_mb    = os.path.getsize(output_path) / (1024 * 1024)
    reduction = (1 - out_mb / src_mb) * 100 if src_mb else 0

    print("\n" + "=" * 60)
    print("  Done in {:.1f}s".format(elapsed))
    print("  {:.1f} MB  ->  {:.1f} MB  ({:.1f}% reduction)".format(src_mb, out_mb, reduction))
    print("  {} frames kept".format(len(segments)))
    print("=" * 60)


if __name__ == "__main__":
    main()