"""Learned in-play classifier on windowed signal features, scored with leave-one-clip-out."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from footy.evaluate import boundary_errors, frame_scores, label_frames
from footy.ingest import load_config
from footy.inplay import clean

SIGNALS = ["cam_speed", "zoom_rate", "player_speed"]
WHISTLE_CAP_S = 15.0


def whistle_features(times: np.ndarray, whistles: pd.DataFrame) -> pd.DataFrame:
    """Seconds since the last whistle and until the next one (capped); whistles are the audio.py events."""
    starts = np.sort(whistles.start_s.to_numpy()) if len(whistles) else np.array([])
    since = np.full(len(times), WHISTLE_CAP_S)
    until = np.full(len(times), WHISTLE_CAP_S)
    for i, t in enumerate(times):
        past, future = starts[starts <= t], starts[starts > t]
        if len(past):
            since[i] = min(t - past[-1], WHISTLE_CAP_S)
        if len(future):
            until[i] = min(future[0] - t, WHISTLE_CAP_S)
    return pd.DataFrame({"whistle_since": since, "whistle_until": until})


def build_features(sig: pd.DataFrame, whistles: pd.DataFrame, fps: float, cfg: dict) -> pd.DataFrame:
    """Windowed features per frame. Uses past and future context, so this is for offline (post-match) analysis."""
    X = {}
    w = lambda s: max(1, round(s * fps))
    for c in SIGNALS:
        x = sig[c].ffill().bfill()
        for s in cfg["centred_windows_s"]:
            X[f"{c}_med{s}"] = x.rolling(w(s), center=True, min_periods=1).median()
        X[f"{c}_std{cfg['std_window_s']}"] = x.rolling(w(cfg["std_window_s"]), center=True, min_periods=1).std().fillna(0)
        past = x.rolling(w(cfg["side_window_s"]), min_periods=1).mean()
        future = x[::-1].rolling(w(cfg["side_window_s"]), min_periods=1).mean()[::-1]
        X[f"{c}_past"], X[f"{c}_future"] = past, future
    X["n_players"] = sig.n_players.rolling(w(5), center=True, min_periods=1).median()
    X["ball_cover"] = sig.ball_detected.astype(float).rolling(w(5), center=True, min_periods=1).mean()
    X = pd.DataFrame(X)
    return pd.concat([X, whistle_features(sig.time_s.to_numpy(), whistles)], axis=1)


def make_model(kind: str, cfg: dict):
    if kind == "gbt":
        return HistGradientBoostingClassifier(max_depth=cfg["gbt_max_depth"], max_iter=cfg["gbt_iters"],
                                              learning_rate=0.05, l2_regularization=1.0, random_state=0)
    return make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=1000))


def load(signals: Path, labels: Path, full: dict) -> dict:
    sig = pd.read_parquet(signals)
    stem = signals.stem.removesuffix("_signals")
    wpath = signals.parent / f"{stem}_whistles.csv"
    whistles = pd.read_csv(wpath) if wpath.exists() else pd.DataFrame(columns=["start_s", "end_s"])
    times = sig.time_s.to_numpy()
    fps = 1 / float(np.median(np.diff(times)))
    return {"name": stem, "times": times, "fps": fps, "y": label_frames(times, labels, fps),
            "X": build_features(sig, whistles, fps, full["learn"])}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("pairs", nargs="+", type=Path, help="SIGNALS LABELS [SIGNALS LABELS ...]")
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--out", type=Path, default=Path("outputs"))
    args = p.parse_args()
    if len(args.pairs) % 2:
        raise SystemExit("give SIGNALS LABELS pairs")

    full = load_config(args.config)
    cfg, min_stop = full["learn"], full["stoppage"]["min_seconds"]
    tol = full["evaluate"]["boundary_tolerance_seconds"]
    clips = [load(args.pairs[i], args.pairs[i + 1], full) for i in range(0, len(args.pairs), 2)]

    rows = []
    for kind in ("logreg", "gbt"):
        for i, test in enumerate(clips):
            train = [c for j, c in enumerate(clips) if j != i]
            model = make_model(kind, cfg).fit(pd.concat([c["X"] for c in train]), np.concatenate([c["y"] for c in train]))
            prob = pd.Series(model.predict_proba(test["X"])[:, 1])
            prob = prob.rolling(max(1, round(cfg["prob_smooth_s"] * test["fps"])), center=True, min_periods=1).mean()
            pred = clean((prob > cfg["prob_threshold"]).to_numpy(), test["fps"], min_stop, cfg["min_play_seconds"])
            everything = np.ones(len(pred), bool)
            rows.append({"model": kind, "clip": test["name"], **frame_scores(test["y"], pred),
                         **boundary_errors(test["y"], pred, test["times"], everything, tol)})

    table = pd.DataFrame(rows)
    print(table.to_string(index=False))
    print()
    print(table.groupby("model")[["accuracy", "precision", "recall", "matched", "spurious"]].agg(
        {"accuracy": "mean", "precision": "mean", "recall": "mean", "matched": "sum", "spurious": "sum"}).round(3))

    ref = args.out / "combined_evaluation.csv"
    if ref.exists():
        r = pd.read_csv(ref)
        r = r[r.model == "leave-one-clip-out"]
        print(f"\nrules, leave-one-clip-out (from {ref.name}): accuracy {r.accuracy.mean():.3f}, "
              f"precision {r.precision.mean():.3f}, recall {r.recall.mean():.3f}, "
              f"matched {int(r.matched.sum())}/{int(r.labelled.sum())}, spurious {int(r.spurious.sum())}")

    # feature importance of the final gbt model fitted on all clips (permutation-free: just informational via logreg coefs)
    lr = make_model("logreg", cfg).fit(pd.concat([c["X"] for c in clips]), np.concatenate([c["y"] for c in clips]))
    coef = pd.Series(lr[-1].coef_[0], index=clips[0]["X"].columns).sort_values(key=np.abs, ascending=False)
    print("\nlargest standardised logreg coefficients (all clips):")
    print(coef.head(8).round(2).to_string())
    table.to_csv(args.out / "learn_evaluation.csv", index=False)


if __name__ == "__main__":
    main()
