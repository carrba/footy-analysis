# Football Match Analysis

Analyse recorded football matches from Veo footage. First metric: **ball-in-play time** — how much of the match the ball was live vs stopped. Later metrics (territory, then possession) build on this.

## Context
- Source footage: Veo **follow-cam** download (1080p .mp4, camera pans/zooms to follow the ball). Panoramic view can't be downloaded from Veo, so it's out of scope for v1.
- The camera moves, so there is no fixed pitch calibration. Avoid anything that needs real-world coordinates in v1.
- Veo recordings include audio; referee whistles are a usable signal.
- Benchmarks: hand-tagged ground truth on a 5-minute clip, then stopwatch spot-checks on a full match.
- Dev environment: Windows + WSL2 (Ubuntu), Docker Desktop. Python 3.11+. Use CUDA if available, otherwise CPU for short clips. Copy video into the repo inside WSL rather than reading from /mnt/c (slow).

## Working agreement
- Work one phase at a time. Stop at the end of each phase, summarise results, and wait for confirmation before moving on.
- Small, runnable scripts/modules for core logic; notebooks only for exploration/visualisation.
- Keep large files (video, weights, outputs) out of git: `data/`, `models/`, `outputs/` are gitignored.
- Every stage writes an intermediate artefact (JSON/Parquet/CSV) so stages can be re-run independently.
- Thresholds (fps, stillness thresholds, min stoppage length, smoothing windows) live in `config.yaml`, not in code.

## Definition: ball in play (v1)
- **Stoppage** starts when the ball leaves the pitch or play is whistled dead; ends at the restart (throw-in, goal kick, corner, free kick, kick-off, drop ball).
- Stoppages shorter than `stoppage.min_seconds` (default 2s) are ignored.
- Half-time and pre/post-match footage are excluded entirely (period boundaries set in config or tagged).
- Output: in-play minutes and %, number of stoppages, median/longest stoppage, longest continuous in-play spell, per half.

## Signals to combine
1. **Ball**: detected? moving? (YOLO ball detection + short-gap interpolation). Ball absent or static for a while suggests a stoppage.
2. **Player motion**: aggregate player speed in frame (detections or optical flow, compensating for camera pan). Players walking/standing suggests a stoppage.
3. **Camera motion**: follow-cam often settles or drifts slowly during stoppages.
4. **Audio**: whistle detection (band-pass ~2-4 kHz energy spikes) marks likely stop/restart points.
Start with the simplest signal that works and add others only if evaluation says so.

## Repo layout (target)
```
data/raw/            # Veo follow-cam downloads (gitignored)
data/clips/          # trimmed test clips
data/labels/         # hand-tagged ground truth (CSV)
models/              # weights (gitignored)
outputs/             # per-frame signals, segments, stats, annotated video
src/footy/
  ingest.py          # trim/resample video, extract audio (ffmpeg)
  tag.py             # manual tagging tool: toggle in-play/stopped -> CSV
  detect.py          # ball + player detection
  motion.py          # player/camera motion signals
  audio.py           # whistle detection
  inplay.py          # combine signals -> in-play segments -> stats
  evaluate.py        # compare segments vs hand labels
config.yaml
```

## Phases
1. **Setup**: repo scaffold, venv, deps (ultralytics, supervision, opencv, ffmpeg, numpy, pandas, librosa), GPU check, gitignore.
2. **Ingest**: trim a 5-minute clip from a follow-cam download; resample to configurable fps (default 10); extract audio to WAV.
3. **Ground truth**: minimal tagging tool. Play the clip; one key toggles in-play/stopped, recording timestamps to CSV. Allow slowed playback and stepping back.
4. **Detection**: ball + player detection on the clip; write per-frame results; render annotated preview video. Report ball detection coverage %.
5. **Signals**: compute per-frame ball-moving, player-motion and camera-motion signals; whistle detection from audio. Plot all signals against the ground-truth timeline.
6. **In-play classifier**: rule-based combination with smoothing and min-stoppage filtering -> segments -> stats.
7. **Evaluate**: frame-level agreement and segment boundary error vs hand labels; tune thresholds; then spot-check a full match with a stopwatch.
8. **Full match + packaging**: CLI for a full match (with period boundaries) -> container; later upload -> queue -> worker.

## Later metrics (not now)
- Territory (which third play is in) per minute.
- Shots / set pieces.
- Possession, using in-play segments + team assignment.
