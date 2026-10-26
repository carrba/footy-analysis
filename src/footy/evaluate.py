"""Compare predicted in-play frames with hand labels; tune thresholds on a train window, report on held-out time."""
import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from footy.audio import load_inplay_spans
from footy.ingest import load_config
from footy.inplay import classify, clean
from footy.motion import smooth

SIGNALS = {"cam_speed": "cam_speed_smooth", "zoom_rate": "zoom_rate_smooth", "player_speed": "player_speed_smooth"}


def label_frames(times: np.ndarray, labels: Path, fps: float) -> np.ndarray:
    y = np.zeros(len(times), bool)
    for a, b in load_inplay_spans(labels, times[-1] + 1 / fps):
        y |= (times >= a) & (times < b)
    return y


def predict(smoothed: pd.DataFrame, p: dict, fps: float, min_stop_s: float) -> np.ndarray:
    cfg = {"thresholds": {col: p[k] for k, col in SIGNALS.items()}, "votes_min": p["votes_min"]}
    return clean(classify(smoothed, cfg), fps, min_stop_s, p["min_play_seconds"])


def frame_scores(y: np.ndarray, pred: np.ndarray) -> dict:
    tp, fp, fn = (y & pred).sum(), (~y & pred).sum(), (y & ~pred).sum()
    return {"accuracy": round(float((y == pred).mean()), 3),
            "precision": round(float(tp / max(tp + fp, 1)), 3),
            "recall": round(float(tp / max(tp + fn, 1)), 3)}


def transitions(arr: np.ndarray, times: np.ndarray) -> list[tuple[float, str]]:
    idx = np.flatnonzero(np.diff(arr.astype(np.int8))) + 1
    return [(float(times[i]), "start" if arr[i] else "stop") for i in idx]


def boundary_errors(y, pred, times, mask, tol) -> dict:
    """Match each labelled start/stop to the nearest predicted transition of the same kind within tol seconds."""
    lab = [t for t in transitions(y, times) if mask[np.searchsorted(times, t[0])]]
    got = [t for t in transitions(pred, times) if mask[np.searchsorted(times, t[0])]]
    errs, used = [], set()
    for t, kind in lab:
        cands = [(abs(g - t), j, g - t) for j, (g, k) in enumerate(got) if k == kind and j not in used]
        if cands and min(cands)[0] <= tol:
            _, j, e = min(cands)
            used.add(j)
            errs.append(e)
    return {"labelled": len(lab), "matched": len(errs), "spurious": len(got) - len(used),
            "median_abs_err_s": round(float(np.median(np.abs(errs))), 1) if errs else None,
            "mean_signed_err_s": round(float(np.mean(errs)), 1) if errs else None}


def grid_params(grid: dict) -> list[dict]:
    keys = list(grid)
    return [dict(zip(keys, v)) for v in itertools.product(*grid.values())]


def tune(smoothed: dict, y, fps, min_stop, mask, grid) -> tuple[dict, float]:
    best, best_acc = None, -1.0
    for p in grid_params(grid):
        pred = predict(smoothed[p["smooth_seconds"]], p, fps, min_stop)
        acc = (y[mask] == pred[mask]).mean()
        if acc > best_acc:
            best, best_acc = p, float(acc)
    return best, best_acc


def report(name, y, pred, times, mask, tol) -> dict:
    return {"window": name, **frame_scores(y[mask], pred[mask]), **boundary_errors(y, pred, times, mask, tol)}


def load_clip(signals: Path, labels: Path, full: dict) -> dict:
    raw = pd.read_parquet(signals)
    times = raw.time_s.to_numpy()
    fps = 1 / float(np.median(np.diff(times)))
    seconds = set(full["evaluate"]["grid"]["smooth_seconds"]) | {full["signals"]["smooth_seconds"]}
    return {"name": signals.stem.removesuffix("_signals"), "times": times, "fps": fps,
            "y": label_frames(times, labels, fps), "smoothed": {s: smooth(raw.copy(), s, fps) for s in seconds}}


def tune_pooled(clips: list[dict], min_stop: float, grid: dict) -> dict:
    """Parameters with the best frame accuracy over all frames of the given clips."""
    best, best_acc = None, -1.0
    for p in grid_params(grid):
        ok = sum((c["y"] == predict(c["smoothed"][p["smooth_seconds"]], p, c["fps"], min_stop)).sum() for c in clips)
        acc = ok / sum(len(c["y"]) for c in clips)
        if acc > best_acc:
            best, best_acc = p, float(acc)
    return best


def run_multi(pairs: list[tuple[Path, Path]], full: dict, out: Path, match_split: int = 0) -> None:
    ev, min_stop = full["evaluate"], full["stoppage"]["min_seconds"]
    tol = ev["boundary_tolerance_seconds"]
    clips = [load_clip(s, l, full) for s, l in pairs]
    cur = {"smooth_seconds": full["signals"]["smooth_seconds"], "min_play_seconds": full["inplay"]["min_play_seconds"],
           "votes_min": full["inplay"]["votes_min"],
           **{k: full["inplay"]["thresholds"][col] for k, col in SIGNALS.items()}}

    def score(model, c, prm):
        pred = predict(c["smoothed"][prm["smooth_seconds"]], prm, c["fps"], min_stop)
        return {"model": model, "clip": c["name"], **report("all", c["y"], pred, c["times"], np.ones(len(pred), bool), tol)}

    rows, fold_params = [], {}
    for i, test in enumerate(clips):
        train = [c for j, c in enumerate(clips) if j != i]
        fold_params[test["name"]] = tune_pooled(train, min_stop, ev["grid"])
        rows.append(score("config.yaml (untuned)", test, cur))
        rows.append(score("leave-one-clip-out", test, fold_params[test["name"]]))
    if match_split:  # fit on the clips of one match, score the clips of the other
        sides = (clips[:match_split], clips[match_split:])
        for fit, test in (sides, sides[::-1]):
            prm = tune_pooled(fit, min_stop, ev["grid"])
            fold_params["cross-match fit on " + "+".join(c["name"][:6] for c in fit)] = prm
            rows += [score("cross-match", c, prm) for c in test]
    joint = tune_pooled(clips, min_stop, ev["grid"])
    rows += [score("joint (in-sample, optimistic)", c, joint) for c in clips]

    table = pd.DataFrame(rows).drop(columns="window")
    print(table.to_string(index=False))
    for name, prm in fold_params.items():
        print(f"fit without {name}:", json.dumps(prm))
    print("joint fit:", json.dumps(joint))
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "combined_evaluation.csv", index=False)
    (out / "combined_tuned.json").write_text(json.dumps(joint, indent=2))


def run_calibrated(pairs: list[tuple[Path, Path]], full: dict, out: Path) -> None:
    """Leave-one-clip-out tuning with objective = accuracy - weight * mean per-clip |in-play % error|."""
    ev, min_stop = full["evaluate"], full["stoppage"]["min_seconds"]
    tol = ev["boundary_tolerance_seconds"]
    clips = [load_clip(s, l, full) for s, l in pairs]
    params = grid_params(ev["grid"])
    n = np.array([len(c["y"]) for c in clips], float)
    true_play = np.array([c["y"].sum() for c in clips], float)
    stats = np.zeros((len(params), len(clips), 2))  # per param and clip: correct frames, predicted in-play frames
    for k, p in enumerate(params):
        for i, c in enumerate(clips):
            pred = predict(c["smoothed"][p["smooth_seconds"]], p, c["fps"], min_stop)
            stats[k, i] = (c["y"] == pred).sum(), pred.sum()

    def pick(idx: list[int], weight: float) -> dict:
        acc = stats[:, idx, 0].sum(1) / n[idx].sum()
        err = (np.abs(stats[:, idx, 1] - true_play[idx]) / n[idx]).mean(1)
        return params[int(np.argmax(acc - weight * err))]

    rows = []
    for w in ev["calibration_weights"]:
        for i, test in enumerate(clips):
            prm = pick([j for j in range(len(clips)) if j != i], w)
            pred = predict(test["smoothed"][prm["smooth_seconds"]], prm, test["fps"], min_stop)
            everything = np.ones(len(pred), bool)
            rows.append({"weight": w, "clip": test["name"][:6], **frame_scores(test["y"], pred),
                         **boundary_errors(test["y"], pred, test["times"], everything, tol),
                         "play_pct_true": round(100 * test["y"].mean(), 1), "play_pct_pred": round(100 * pred.mean(), 1),
                         "play_pct_bias": round(100 * (pred.mean() - test["y"].mean()), 1)})
    table = pd.DataFrame(rows)
    print(table[["weight", "clip", "accuracy", "matched", "spurious", "play_pct_true", "play_pct_pred", "play_pct_bias"]]
          .to_string(index=False))
    summary = table.groupby("weight").agg(accuracy=("accuracy", "mean"), play_bias=("play_pct_bias", "mean"),
                                          play_abs_err=("play_pct_bias", lambda s: s.abs().mean()),
                                          matched=("matched", "sum"), spurious=("spurious", "sum")).round(3)
    print("\nleave-one-clip-out, averaged over clips (play_* in percentage points):")
    print(summary.to_string())
    print("\njoint fits (all clips, in-sample):")
    for w in ev["calibration_weights"]:
        print(f"  weight {w}:", json.dumps(pick(list(range(len(clips))), w)))
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "calibration_evaluation.csv", index=False)
    for w in ev["calibration_weights"]:
        (out / f"calibrated_w{w}_tuned.json").write_text(json.dumps(pick(list(range(len(clips))), w), indent=2))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("signals", type=Path, help="signals parquet from footy.motion")
    p.add_argument("labels", type=Path)
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--out", type=Path, default=Path("outputs"))
    p.add_argument("--params", type=Path, help="tuned.json from another clip: score it as-is on this clip, no tuning")
    p.add_argument("--also", nargs=2, action="append", type=Path, metavar=("SIGNALS", "LABELS"),
                   help="another clip; with this, tune on the pooled clips and run leave-one-clip-out")
    p.add_argument("--match-split", type=int, default=0,
                   help="with --also: the first N clips (in order given) are one match, the rest another; adds a cross-match check")
    p.add_argument("--calibrate", action="store_true",
                   help="with --also: sweep the in-play % error weight in the tuning objective (leave-one-clip-out)")
    args = p.parse_args()

    full = load_config(args.config)
    if args.also and args.calibrate:
        run_calibrated([(args.signals, args.labels)] + [tuple(a) for a in args.also], full, args.out)
        return
    if args.also:
        run_multi([(args.signals, args.labels)] + [tuple(a) for a in args.also], full, args.out, args.match_split)
        return
    ev, min_stop = full["evaluate"], full["stoppage"]["min_seconds"]
    raw = pd.read_parquet(args.signals)
    times = raw.time_s.to_numpy()
    fps = 1 / float(np.median(np.diff(times)))
    y = label_frames(times, args.labels, fps)
    tol = ev["boundary_tolerance_seconds"]
    smoothed = {s: smooth(raw.copy(), s, fps) for s in ev["grid"]["smooth_seconds"]}
    cur = {"smooth_seconds": full["signals"]["smooth_seconds"], "min_play_seconds": full["inplay"]["min_play_seconds"],
           "votes_min": full["inplay"]["votes_min"],
           **{k: full["inplay"]["thresholds"][col] for k, col in SIGNALS.items()}}
    smoothed.setdefault(cur["smooth_seconds"], smooth(raw.copy(), cur["smooth_seconds"], fps))

    everything = np.ones(len(times), bool)
    if args.params:  # independent test: nothing here is fitted to this clip
        other = json.loads(args.params.read_text())
        smoothed.setdefault(other["smooth_seconds"], smooth(raw.copy(), other["smooth_seconds"], fps))
        rows = []
        for name, prm in (("config.yaml (untuned)", cur), (f"tuned on {args.params.stem.removesuffix('_tuned')}", other)):
            pred = predict(smoothed[prm["smooth_seconds"]], prm, fps, min_stop)
            rows.append({"model": name, **report("all", y, pred, times, everything, tol)})
        table = pd.DataFrame(rows)
        print(table.to_string(index=False))
        stem = args.signals.stem.removesuffix("_signals")
        args.out.mkdir(parents=True, exist_ok=True)
        table.to_csv(args.out / f"{stem}_evaluation_independent.csv", index=False)
        return

    train = times < ev["train_end_s"]
    held = ~train
    rows = []

    base = predict(smoothed[cur["smooth_seconds"]], cur, fps, min_stop)
    rows += [{"model": "config.yaml (untuned)", **report(n, y, base, times, m, tol)}
             for n, m in (("train", train), ("held-out", held))]

    best, _ = tune(smoothed, y, fps, min_stop, train, ev["grid"])
    tuned = predict(smoothed[best["smooth_seconds"]], best, fps, min_stop)
    rows += [{"model": "tuned on train", **report(n, y, tuned, times, m, tol)}
             for n, m in (("train", train), ("held-out", held))]

    # out-of-fold: tune on all blocks but one, predict the one left out, pool the predictions
    folds = np.array_split(np.arange(len(times)), ev["cv_folds"])
    oof, fold_params = np.zeros(len(times), bool), []
    for f in folds:
        test = np.zeros(len(times), bool)
        test[f] = True
        pf, _ = tune(smoothed, y, fps, min_stop, ~test, ev["grid"])
        fold_params.append(pf)
        oof[f] = predict(smoothed[pf["smooth_seconds"]], pf, fps, min_stop)[f]
    rows.append({"model": f"{ev['cv_folds']}-fold out-of-fold", **report("all", y, oof, times, np.ones(len(times), bool), tol)})

    table = pd.DataFrame(rows)
    print(table.to_string(index=False))
    print("\ntuned on train:", json.dumps(best))
    print("fold params:")
    for pf in fold_params:
        print("  ", json.dumps(pf))
    args.out.mkdir(parents=True, exist_ok=True)
    stem = args.signals.stem.removesuffix("_signals")
    table.to_csv(args.out / f"{stem}_evaluation.csv", index=False)
    (args.out / f"{stem}_tuned.json").write_text(json.dumps(best, indent=2))


if __name__ == "__main__":
    main()
