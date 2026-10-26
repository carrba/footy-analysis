"""Detect referee whistles: narrow-band 2-4 kHz energy bursts. Writes features, events and a plot."""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.io import wavfile
from scipy.signal import stft

from footy.ingest import load_config


def band_features(wav: Path, cfg: dict) -> pd.DataFrame:
    sr, x = wavfile.read(wav)
    x = x.astype(np.float32) / np.iinfo(np.int16).max
    nper = int(sr * cfg["window_ms"] / 1000)
    f, t, Z = stft(x, fs=sr, nperseg=nper, noverlap=nper // 2)
    power = np.abs(Z) ** 2
    band = (f >= cfg["band_low_hz"]) & (f <= cfg["band_high_hz"])
    band_e, total_e = power[band].sum(0), power.sum(0) + 1e-12
    band_db = 10 * np.log10(band_e + 1e-12)
    return pd.DataFrame({"time_s": t, "band_ratio": band_e / total_e, "band_db": band_db})


def find_whistles(feat: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    hit = (feat.band_ratio > cfg["band_ratio_min"]) & (feat.band_db > feat.band_db.median() + cfg["band_level_db"])
    t, hop = feat.time_s.to_numpy(), float(np.median(np.diff(feat.time_s)))
    spans, start = [], None
    for i, h in enumerate(hit.to_numpy()):
        if h and start is None:
            start = i
        elif not h and start is not None:
            spans.append([t[start], t[i - 1] + hop, start, i - 1])
            start = None
    if start is not None:
        spans.append([t[start], t[-1] + hop, start, len(t) - 1])

    merged = []
    for s in spans:
        if merged and s[0] - merged[-1][1] < cfg["merge_gap_seconds"]:
            merged[-1][1], merged[-1][3] = s[1], s[3]
        else:
            merged.append(s)
    rows = [{"start_s": round(a, 3), "end_s": round(b, 3), "peak_db": round(float(feat.band_db[i:j + 1].max()), 1)}
            for a, b, i, j in merged if b - a >= cfg["min_seconds"]]
    return pd.DataFrame(rows, columns=["start_s", "end_s", "peak_db"])


def load_inplay_spans(labels: Path, duration: float) -> list[tuple[float, float]]:
    lab = pd.read_csv(labels)
    ev = list(zip(lab.time_s, lab.label)) + [(duration, "O")]
    return [(a, b) for (a, l), (b, _) in zip(ev, ev[1:]) if l in ("A", "B")]


def plot(feat: pd.DataFrame, whistles: pd.DataFrame, spans: list, out: Path) -> None:
    fig, ax = plt.subplots(2, 1, figsize=(14, 5), sharex=True)
    for a in ax:
        for s, e in spans:
            a.axvspan(s, e, color="tab:green", alpha=0.15, lw=0)
    ax[0].plot(feat.time_s, feat.band_db, lw=0.6)
    ax[0].set_ylabel("2-4 kHz level (dB)")
    ax[1].plot(feat.time_s, feat.band_ratio, lw=0.6)
    ax[1].set_ylabel("band / total energy")
    for _, w in whistles.iterrows():
        for a in ax:
            a.axvspan(w.start_s, w.end_s, color="tab:red", alpha=0.6, lw=0)
    ax[1].set_xlabel("time (s)   green = in play (hand labels), red = detected whistle")
    fig.tight_layout()
    fig.savefig(out, dpi=110)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("wav", type=Path)
    p.add_argument("--labels", type=Path, help="hand-tagged CSV to overlay on the plot")
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--out", type=Path, default=Path("outputs"))
    args = p.parse_args()

    cfg = load_config(args.config)["audio"]
    args.out.mkdir(parents=True, exist_ok=True)
    feat = band_features(args.wav, cfg)
    whistles = find_whistles(feat, cfg)
    feat.to_parquet(args.out / f"{args.wav.stem}_audio.parquet")
    whistles.to_csv(args.out / f"{args.wav.stem}_whistles.csv", index=False)
    spans = load_inplay_spans(args.labels, feat.time_s.iloc[-1]) if args.labels else []
    plot(feat, whistles, spans, args.out / f"{args.wav.stem}_whistles.png")
    print(f"{len(whistles)} whistles")
    print(whistles.to_string(index=False))


if __name__ == "__main__":
    main()
