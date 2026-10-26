"""Combine per-frame signals into in-play / stopped segments and per-period stats."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from footy.audio import load_inplay_spans
from footy.ingest import load_config


def runs(mask: np.ndarray) -> list[tuple[int, int, bool]]:
    """Run-length encode a boolean array as (start, end_exclusive, value)."""
    edges = np.flatnonzero(np.diff(mask.astype(np.int8))) + 1
    starts = np.r_[0, edges]
    ends = np.r_[edges, len(mask)]
    return [(int(a), int(b), bool(mask[a])) for a, b in zip(starts, ends)]


def classify(df: pd.DataFrame, cfg: dict) -> np.ndarray:
    votes = sum((df[c].ffill().bfill() > t).to_numpy().astype(int) for c, t in cfg["thresholds"].items())
    return votes >= cfg["votes_min"]


def clean(play: np.ndarray, fps: float, min_stop_s: float, min_play_s: float) -> np.ndarray:
    """Fill stoppages shorter than min_stop_s, then drop in-play blips shorter than min_play_s."""
    play = play.copy()
    for value, min_s in ((False, min_stop_s), (True, min_play_s)):
        for a, b, v in runs(play):
            if v == value and (b - a) < min_s * fps:
                play[a:b] = not value
    return play


def segments(play: np.ndarray, times: np.ndarray, fps: float, period: str) -> pd.DataFrame:
    rows = [{"period": period, "state": "play" if v else "stopped",
             "start_s": round(times[a], 3), "end_s": round(times[b - 1] + 1 / fps, 3),
             "truncated": a == 0 or b == len(play)} for a, b, v in runs(play)]
    return pd.DataFrame(rows)


def stats(segs: pd.DataFrame) -> dict:
    """Stats for one period. Stoppages cut off by the period edge are counted but excluded from median/longest."""
    dur = segs.end_s - segs.start_s
    play, stop = segs.state == "play", segs.state == "stopped"
    full_stops = dur[stop & ~segs.truncated]
    total = float(dur.sum())
    return {
        "in_play_min": round(float(dur[play].sum()) / 60, 2),
        "total_min": round(total / 60, 2),
        "in_play_pct": round(100 * float(dur[play].sum()) / total, 1),
        "stoppages": int(stop.sum()),
        "stoppages_cut_by_period_edge": int((stop & segs.truncated).sum()),
        "median_stoppage_s": round(float(full_stops.median()), 1) if len(full_stops) else None,
        "longest_stoppage_s": round(float(full_stops.max()), 1) if len(full_stops) else None,
        "longest_play_s": round(float(dur[play].max()), 1) if play.any() else None,
    }


def analyse(df: pd.DataFrame, cfg: dict, min_stop_s: float, fps: float) -> tuple[pd.DataFrame, dict]:
    play_all = classify(df, cfg)
    all_segs, all_stats = [], {}
    for p in cfg["periods"]:
        m = ((df.time_s >= p["start_s"]) & (df.time_s < p["end_s"])).to_numpy()
        play = clean(play_all[m], fps, min_stop_s, cfg["min_play_seconds"])
        segs = segments(play, df.time_s.to_numpy()[m], fps, p["name"])
        all_segs.append(segs)
        all_stats[p["name"]] = stats(segs)
    return pd.concat(all_segs, ignore_index=True), all_stats


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("signals", type=Path, help="signals parquet from footy.motion")
    p.add_argument("--labels", type=Path, help="hand-tagged CSV: also print stats from the labels for comparison")
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--out", type=Path, default=Path("outputs"))
    args = p.parse_args()

    full = load_config(args.config)
    df = pd.read_parquet(args.signals)
    fps = 1 / float(np.median(np.diff(df.time_s)))
    segs, st = analyse(df, full["inplay"], full["stoppage"]["min_seconds"], fps)

    stem = args.signals.stem.removesuffix("_signals")
    args.out.mkdir(parents=True, exist_ok=True)
    segs.to_csv(args.out / f"{stem}_segments.csv", index=False)
    (args.out / f"{stem}_stats.json").write_text(json.dumps(st, indent=2))
    print(segs.to_string(index=False))
    table = {f"{k} (predicted)": v for k, v in st.items()}

    if args.labels:
        lab = {}
        for p_ in full["inplay"]["periods"]:
            t = df.time_s.to_numpy()
            m = (t >= p_["start_s"]) & (t < p_["end_s"])
            y = np.zeros(len(t), bool)
            for a, b in load_inplay_spans(args.labels, t[-1] + 1 / fps):
                y |= (t >= a) & (t < b)
            lab[f"{p_['name']} (hand labels)"] = stats(segments(y[m], t[m], fps, p_["name"]))
        table.update(lab)
    print(pd.DataFrame(table).to_string())


if __name__ == "__main__":
    main()
