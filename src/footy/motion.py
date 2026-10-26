"""Per-frame motion signals: camera motion, camera-compensated player motion, camera-compensated ball motion."""
import argparse
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import rankdata

from footy.audio import load_inplay_spans
from footy.ingest import load_config

LK = dict(winSize=(21, 21), maxLevel=3, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03))


def box_mask(shape, boxes, shrink=0.0, invert=False) -> np.ndarray:
    mask = np.zeros(shape, np.uint8) if not invert else np.full(shape, 255, np.uint8)
    for x1, y1, x2, y2 in boxes:
        w, h = x2 - x1, y2 - y1
        a, b = int(x1 + shrink * w), int(y1 + shrink * h)
        c, d = int(x2 - shrink * w), int(y2 - shrink * h)
        mask[max(b, 0):max(d, 0), max(a, 0):max(c, 0)] = 255 if not invert else 0
    return mask


def track(prev, cur, mask, max_pts):
    pts = cv2.goodFeaturesToTrack(prev, maxCorners=max_pts, qualityLevel=0.01, minDistance=6, mask=mask)
    if pts is None:
        return np.empty((0, 2)), np.empty((0, 2))
    nxt, st, _ = cv2.calcOpticalFlowPyrLK(prev, cur, pts, None, **LK)
    ok = st.ravel() == 1
    return pts[ok].reshape(-1, 2), nxt[ok].reshape(-1, 2)


def camera_motion(p0, p1, size):
    """Similarity transform prev->cur from background points. Returns (A, shift_widths, scale) or None."""
    if len(p0) < 10:
        return None
    A, _ = cv2.estimateAffinePartial2D(p0, p1, method=cv2.RANSAC, ransacReprojThreshold=2.0)
    if A is None:
        return None
    centre = np.array([size[0] / 2, size[1] / 2])
    shift = A[:, :2] @ centre + A[:, 2] - centre
    return A, float(np.hypot(*shift) / size[0]), float(np.hypot(A[0, 0], A[1, 0]))


def ball_track(det: pd.DataFrame, n: int, cfg: dict) -> tuple[np.ndarray, set]:
    """Best ball centre per frame (n x 2, NaN where none), short gaps interpolated; plus frames with a raw detection."""
    b = det[(det.cls == "ball") & (det.conf >= cfg["ball_min_conf"])]
    b = b.sort_values("conf").groupby("frame").tail(1).set_index("frame")
    xy = pd.DataFrame(np.nan, index=range(n), columns=["x", "y"])
    xy.loc[b.index, "x"], xy.loc[b.index, "y"] = (b.x1 + b.x2) / 2, (b.y1 + b.y2) / 2
    xy = xy.interpolate(limit=cfg["ball_gap_frames"], limit_area="inside")
    return xy.to_numpy(), set(b.index)


def compute(clip: Path, det: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    cap = cv2.VideoCapture(str(clip))
    fps, n = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    s = cfg["work_width"] / W
    size = (cfg["work_width"], round(H * s))
    players = {f: g[["x1", "y1", "x2", "y2"]].to_numpy() * s for f, g in det[det.cls == "player"].groupby("frame")}
    ball_xy, ball_seen = ball_track(det, n, cfg)
    ball_xy = ball_xy * s
    logo = [(size[0] * 0.85, size[1] * 0.85, size[0], size[1])]  # Veo watermark, bottom right

    rows, prev, A_prev = [], None, None
    for i in range(n):
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(cv2.resize(frame, size), cv2.COLOR_BGR2GRAY)
        row = dict(frame=i, time_s=round(i / fps, 3), n_players=len(players.get(i, [])),
                   ball_detected=i in ball_seen, cam_speed=np.nan, zoom_rate=np.nan,
                   player_speed=np.nan, ball_speed=np.nan)
        if prev is not None:
            boxes = players.get(i - 1, np.empty((0, 4)))
            bg_mask = box_mask(gray.shape, np.vstack([boxes, logo]) if len(boxes) else logo, invert=True)
            cam = camera_motion(*track(prev, gray, bg_mask, cfg["max_bg_points"]), size)
            if cam is not None:
                A, shift, scale = cam
                row["cam_speed"], row["zoom_rate"] = shift * fps, abs(np.log(scale)) * fps
                # player motion: leftover flow inside player boxes after removing camera motion, in body-heights/s
                if len(boxes):
                    p0, p1 = track(prev, gray, box_mask(gray.shape, boxes, cfg["box_shrink"]), cfg["max_player_points"])
                    if len(p0):
                        resid = np.hypot(*(p1 - (p0 @ A[:, :2].T + A[:, 2])).T)
                        inside = (p0[:, None, 0] >= boxes[None, :, 0]) & (p0[:, None, 0] <= boxes[None, :, 2]) & \
                                 (p0[:, None, 1] >= boxes[None, :, 1]) & (p0[:, None, 1] <= boxes[None, :, 3])
                        heights = np.where(inside, (boxes[:, 3] - boxes[:, 1])[None, :], np.inf).min(1)
                        valid = np.isfinite(heights)
                        if valid.any():
                            row["player_speed"] = float(np.median(resid[valid] / heights[valid]) * fps)
                # ball motion: displacement not explained by camera motion, in frame-widths/s
                a, b = ball_xy[i - 1], ball_xy[i]
                if not (np.isnan(a).any() or np.isnan(b).any()):
                    row["ball_speed"] = float(np.hypot(*(b - (A[:, :2] @ a + A[:, 2]))) / size[0] * fps)
        rows.append(row)
        prev = gray
        if i % 300 == 0:
            print(f"{i}/{n}", flush=True)

    return smooth(pd.DataFrame(rows), cfg["smooth_seconds"], fps)


def smooth(df: pd.DataFrame, seconds: float, fps: float) -> pd.DataFrame:
    """Add *_smooth columns (rolling median) computed from the raw signals."""
    win = max(1, round(seconds * fps))
    for c in ("cam_speed", "zoom_rate", "player_speed", "ball_speed"):
        df[c + "_smooth"] = df[c].rolling(win, center=True, min_periods=1).median()
    df["ball_cover_smooth"] = df.ball_detected.astype(float).rolling(win, center=True, min_periods=1).mean()
    return df


def auc(x: pd.Series, y: np.ndarray) -> float:
    """P(signal in-play > signal stopped); 0.5 = no separation, <0.5 = inverted."""
    m = x.notna().to_numpy()
    r = rankdata(x[m])
    pos = y[m]
    n1, n0 = pos.sum(), (~pos).sum()
    return float((r[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def evaluate(df: pd.DataFrame, labels: Path) -> pd.DataFrame:
    spans = load_inplay_spans(labels, df.time_s.iloc[-1] + 0.1)
    y = np.zeros(len(df), bool)
    for a, b in spans:
        y |= ((df.time_s >= a) & (df.time_s < b)).to_numpy()
    rows = [{"signal": c, "in_play_mean": df.loc[y, c].mean(), "stopped_mean": df.loc[~y, c].mean(),
             "auc": auc(df[c], y)} for c in ("cam_speed_smooth", "zoom_rate_smooth", "player_speed_smooth",
                                             "ball_speed_smooth", "ball_cover_smooth")]
    return pd.DataFrame(rows).round(3)


def plot(df: pd.DataFrame, labels: Path, whistles: Path | None, out: Path) -> None:
    spans = load_inplay_spans(labels, df.time_s.iloc[-1] + 0.1)
    sig = [("cam_speed", "camera pan\n(widths/s)"), ("zoom_rate", "camera zoom\n(|dlog s|/s)"),
           ("player_speed", "player motion\n(body heights/s)"), ("ball_speed", "ball motion\n(widths/s)"),
           ("ball_cover", "ball detected\n(share of frames)")]
    fig, ax = plt.subplots(len(sig), 1, figsize=(14, 10), sharex=True)
    w = pd.read_csv(whistles) if whistles and whistles.exists() else pd.DataFrame(columns=["start_s", "end_s"])
    for a, (c, lab) in zip(ax, sig):
        for s, e in spans:
            a.axvspan(s, e, color="tab:green", alpha=0.15, lw=0)
        if c != "ball_cover":
            a.plot(df.time_s, df[c], lw=0.4, color="0.7")
        a.plot(df.time_s, df[c + "_smooth"], lw=1.4, color="tab:blue")
        for _, r in w.iterrows():
            a.axvline(r.start_s, color="tab:red", lw=1)
        a.set_ylabel(lab, fontsize=8)
    ax[-1].set_xlabel("time (s)   green = in play (hand labels), red = whistle, blue = 3 s rolling median")
    fig.tight_layout()
    fig.savefig(out, dpi=100)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("clip", type=Path)
    p.add_argument("--detections", type=Path, help="default: outputs/<clip stem>_detections.parquet")
    p.add_argument("--labels", type=Path, help="hand-tagged CSV: overlay on the plot and report separability")
    p.add_argument("--whistles", type=Path, help="whistles CSV from footy.audio")
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--out", type=Path, default=Path("outputs"))
    args = p.parse_args()

    cfg = load_config(args.config)["signals"]
    det = pd.read_parquet(args.detections or args.out / f"{args.clip.stem}_detections.parquet")
    df = compute(args.clip, det, cfg)
    args.out.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out / f"{args.clip.stem}_signals.parquet")
    if args.labels:
        plot(df, args.labels, args.whistles, args.out / f"{args.clip.stem}_signals.png")
        print(evaluate(df, args.labels).to_string(index=False))


if __name__ == "__main__":
    main()
