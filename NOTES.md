# Working notes

Last updated 2026-10-24. Status, results and next steps. `CLAUDE.md` stays the project spec; this file is the running log.

## Status (follow-cam, ball-in-play)

| Phase | State |
|---|---|
| 1 Setup | Done. `pyproject.toml` added; `pip install -e . --no-deps` in `.venv`. CPU-only torch (no GPU visible from WSL). |
| 2 Ingest | Done. `ingest.py` also extracts audio (`--audio-only` for the WAV alone). |
| 3 Tagging | Done. `tag.py` keys: A/B = ball live (team), O = out of play, U undo, Q save. A/B both count as in play. |
| 4 Detection | Done. `detect.py`, yolo11s COCO, CPU ~1 s/frame (~30 min per 3-min clip). |
| 5 Signals | Done. `audio.py` (whistles), `motion.py` (camera pan/zoom, player motion, ball motion). |
| 6 Classifier | Done. `inplay.py`, rule-based vote over smoothed pan/zoom/player-motion. |
| 7 Evaluate | Mostly done. `evaluate.py`. Stopwatch spot-check on a full match still to do. |
| 8 Full match + packaging | Not started. |

## Labelled data

| Clip | Match | In play |
|---|---|---|
| `001002-highlight_0s_10fps` | 1 | 55.5% |
| `002945-highlight_0s_10fps` | 1 | 48.8% |
| `001750-highlight_0s_10fps` | 2 | 59.3% |
| `004725-highlight_0s_10fps` | 2 | 58.8% |

All are random 3-minute follow-cam samples (raw in `data/raw/<name>-highlight.mp4`, labels in `data/labels/<name>-highlight_0s_10fps.csv`). No selection bias. Stale and unrelated: `data/labels/2026-10-23_19-45-40_0s_10fps.csv` (2 test tags on the OBS panorama file).

## Results so far

- Signals that work: camera pan (AUC ~0.82-0.88), camera zoom (~0.84-0.87), player motion (~0.75-0.81).
- Ball signals do not work: COCO ball detection is worse in play than when stopped (AUC 0.40-0.51). Drop them from the classifier.
- Frame accuracy, 4 clips / 2 matches, leave-one-clip-out: **~83-85%** (original grid 85.0%, wider grid 83.3%; difference is noise). Untuned first-guess config was 71.6%. Cross-match (fit on one match, score the other): ~82-83%, so settings are not match-specific. The earlier 91% held-out figure on 1 clip was optimistic.
- Wider search grid raised in-sample accuracy (85.5% to 89.3%) but lowered out-of-clip accuracy: overfitting. Do not widen further.
- Learned classifier (`learn.py`, windowed features incl. past/future context and whistle timing, leave-one-clip-out): logistic regression 83.4%, gradient-boosted trees 81.0%, rules 83.3%. **No gain over the rules.** The limit is the information in the signals, not the model. Largest logreg weights: player motion (5 s), camera pan (10 s), visible player count.
- Ceiling cause: slow live play (defenders passing, camera barely moving) looks like a stoppage in these signals.
- Applied to `config.yaml` (joint fit on 4 clips): smooth 7 s, pan 0.03, zoom 0.03, player 0.4, votes 2, min in-play spell 6 s. Zoom and min spell were unstable across folds.
- In-play % per 3-minute clip is off by about +/-8 points on average out of clip (e.g. +11.5 on 001002, -16.3 on 002945), but the average signed bias is only about -2.3 points, so it is mostly scatter, not a systematic bias. (An earlier note claiming a "-8 point bias" came from in-sample predictions and was wrong.) Stoppage counts are close (19 predicted vs 22 labelled across the 4 clips).
- Calibrating the tuning objective for in-play % error (`evaluate --calibrate`): error falls from 8.1 to ~5.6 points but frame accuracy falls 2-3 points and extra transitions rise from 8 to 20-26. Decided not to apply; config stays at the accuracy-tuned settings. Scatter should mostly average out over a full match if chunk errors are independent. Check with the full-match stopwatch test.
- Boundary timing is good when found (median error ~1-2 s). Errors are missed/extra stoppages.
- Whistles: help at some boundaries only (most stoppages are throw-ins/goal kicks with no whistle).
- Joint-fit parameters (not yet in `config.yaml`; see `outputs/combined_tuned.json`): smooth 2 s, pan 0.03, zoom 0.015, player 0.4, votes 2, min in-play spell 4 s. Applying them means setting `signals.smooth_seconds: 2` and rerunning `motion` for every clip.

## Next steps (agreed plan)

Done since the first notes: 2 more clips (match 2), cross-match check, grid widening, joint parameters applied, learned classifier (no gain). Rules + camera/player signals are at their ceiling of ~83%.

1. **Decide how to break the ceiling** (needs better information, not more tuning): (a) fine-tune a ball detector (needs ball bounding-box labels on a few hundred to ~1000 frames; the COCO ball detector is useless here), or (b) the panorama view (see below), or (c) accept ~83% frame accuracy for v1 and move on.
2. **Stopwatch spot-check on a full match** (still outstanding phase 7 item).
3. (Done, not applied) Calibrating for in-play % error: see Results.
4. **AWS GPU run (written, not yet applied).** Terraform in `infra/aws/storage` (S3 bucket, keep) and `infra/aws/compute` (one-shot spot `g4dn.xlarge` + IAM, `destroy` after each job), `scripts/aws_push_code.sh`, and the step-by-step in `docs/aws-gpu-run.md`. Nothing has been created in AWS yet. First steps: request the G/VT spot vCPU quota, `terraform apply` (bucket only), then benchmark on a 3-minute clip before the full match. Motion step is CPU-bound and may need `g4dn.2xlarge`.
5. Phase 8: CLI for a full match with period boundaries, then container; later upload, queue, worker.
6. Housekeeping: nothing is committed yet; add `*.egg-info/` to `.gitignore`.

## Panorama (open decision)

- `data/raw/2026-10-23_19-45-40.mp4` is an OBS capture of the Veo panorama (5:35, 60fps, same match, audio muted at ~2 kbps, player UI baked into the frame). Not used by the follow-cam pipeline.
- Why it matters: fixed camera gives a pitch homography, so real coordinates: thirds, distance covered/top speed, shot locations for xG (user's own xG site; could be changed back to take coordinates). Not possible from follow-cam.
- Risks: ball only a few pixels (needs tiled inference and a fine-tuned detector), UI overlays to crop, edge distortion, player identity across occlusion, capture runs in real time, audio must be unmuted for whistles.
- Not yet tested: ball visibility and pitch calibration on this file.
- If we go this way, rewrite the "panoramic out of scope" parts of `CLAUDE.md` first (user agreed in principle, not done yet).

## Longer-term metrics (feasibility)

Ball in play: yes. Possession: yes, moderate accuracy (kit-colour team assignment). Thirds: rough from follow-cam, exact from panorama. Shots on goal: semi-automatic, user confirms candidates. xG: needs shot coordinates. Distance/top speed per player: only realistic from panorama. No public Veo download API known; keep downloads manual.

## Running the pipeline for a clip

From `~/footy-analysis` in WSL, `source .venv/bin/activate`, with `S=<name>-highlight_0s_10fps`:

```
python -m footy.ingest data/raw/<name>-highlight.mp4 --duration 180      # video + WAV
python -m footy.audio data/clips/$S.wav --labels data/labels/$S.csv      # whistles
python -m footy.detect data/clips/$S.mp4                                  # slow on CPU
python -m footy.tag data/clips/$S.mp4                                     # user tags
python -m footy.motion data/clips/$S.mp4 --labels data/labels/$S.csv --whistles outputs/${S}_whistles.csv
python -m footy.inplay outputs/${S}_signals.parquet --labels data/labels/$S.csv
python -m footy.evaluate outputs/A_signals.parquet data/labels/A.csv --also outputs/B_signals.parquet data/labels/B.csv   # tune + leave-one-clip-out
python -m footy.evaluate outputs/new_signals.parquet data/labels/new.csv --params outputs/other_tuned.json              # independent score
```

## Environment gotchas

- The Claude Code Bash tool runs on Windows, not WSL. Run project commands through `wsl -d Ubuntu-22.04 -- bash -l <script>` (login shell, or `ffmpeg` is not on PATH). Put multi-line commands in a script file; PowerShell mangles quotes and `$vars`.
- `ffprobe` is not installed; use `ffmpeg -i` to read durations.
- `outputs/` contains sample frames and `probe/` scratch; all gitignored. Nothing has been committed to git yet. `*.egg-info/` is not in `.gitignore`.
