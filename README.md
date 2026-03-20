# Smart Behavioral Video Compression
**SentioMind Assignment**

## Problem
Reduce 40–80 GB of daily CCTV footage to < 10 GB while retaining every frame that contains a human.

## Algorithm Implemented
| Step | Method | Rule |
|------|--------|------|
| 1 | Perceptual Hash (pHash) | Drop frame if > 95% similar to last kept frame |
| 2 | Optical Flow Motion Score | Discard if score < 0.05 (static scene) |
| 3 | Haar Face Detection | Always keep if a face is detected |
| 4 | Context Frame | Keep one frame every 3 seconds minimum |
| 5 | Re-encode | Surviving frames → H.264 MP4 @ 12 fps via ffmpeg |

## Deliverables
| # | File | Description |
|---|------|-------------|
| 1 | `solution.py` | Main compression script |
| 2 | `compressed_output.mp4` | Compressed output video |
| 3 | `compression_report.html` | Storyboard + size comparison (offline) |
| 4 | `segments_kept.json` | Segment log for SentioMind pipeline integration |
| 5 | `demo.mp4` | Screen recording demo (< 2 min) |

## Setup & Usage

### Requirements
```
Python 3.9+
ffmpeg (must be on PATH)
```

### Install dependencies
```bash
pip install opencv-python==4.9.0 imagehash==4.3.1 numpy==1.26.4 Pillow==10.3.0
```

### Run
```bash
python solution.py video_sample_1.mov
# or with custom output path:
python solution.py video_sample_1.mov compressed_output.mp4
```

### Outputs generated automatically
- `compressed_output.mp4`
- `compression_report.html`
- `segments_kept.json`

## Integration
`segments_kept.json` is the integration contract consumed by `extract_intelligent_frames()` in the SentioMind main pipeline. It contains frame-level metadata including timestamps, motion scores, and face detection results.

## Performance
- Target: ≥ 70% file size reduction
- Target: ≥ 4× real-time processing speed
