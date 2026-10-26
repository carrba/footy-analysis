"""Manual possession tagging: play a clip, press A / B / O to record possession changes to CSV.

Keys: A/B = team in possession, O = out of play / no possession, U = undo last tag,
space = pause, left/right = seek, ,/. = step one frame (paused), -/+ = speed, Q/Esc = save and quit.
"""
import argparse
import csv
import time
from pathlib import Path

import cv2

from footy.ingest import load_config

LABELS = {ord("a"): "A", ord("b"): "B", ord("o"): "O"}
COLOURS = {"A": (255, 160, 0), "B": (60, 60, 255), "O": (170, 170, 170), None: (255, 255, 255)}
KEYS_LEFT, KEYS_RIGHT = (65361, 2424832), (65363, 2555904)  # waitKeyEx codes: X11 / Windows


def load_events(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, newline="") as f:
        return [{"time_s": float(r["time_s"]), "frame": int(r["frame"]), "label": r["label"]} for r in csv.DictReader(f)]


def save_events(path: Path, events: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["time_s", "frame", "label"])
        w.writeheader()
        w.writerows(events)


def state_at(events: list[dict], frame: int) -> str | None:
    current = None
    for e in events:  # events are sorted by frame
        if e["frame"] > frame:
            break
        current = e["label"]
    return current


def add_event(events: list[dict], frame: int, fps: float, label: str) -> None:
    events[:] = [e for e in events if e["frame"] != frame]
    events.append({"time_s": round(frame / fps, 3), "frame": frame, "label": label})
    events.sort(key=lambda e: e["frame"])


def run(clip: Path, out: Path, cfg: dict) -> None:
    cap = cv2.VideoCapture(str(clip))
    if not cap.isOpened():
        raise SystemExit(f"cannot open {clip}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    events = load_events(out)
    speed, paused, frame_idx, img = cfg["speed"], False, 0, None
    seek_frames = round(cfg["seek_seconds"] * fps)
    win = "tag"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    def seek(to: int) -> None:
        nonlocal frame_idx, img
        frame_idx = max(0, min(total - 1, to))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if ok:
            img = frame

    seek(0)
    while True:
        t0 = time.perf_counter()
        if img is not None:
            scale = cfg["display_width"] / img.shape[1]
            view = cv2.resize(img, None, fx=scale, fy=scale)
            label = state_at(events, frame_idx)
            cv2.rectangle(view, (0, 0), (view.shape[1], 36), (0, 0, 0), -1)
            cv2.putText(view, f"{frame_idx / fps:7.1f}s / {total / fps:.0f}s   x{speed:g}   {'PAUSED' if paused else ''}",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(view, f"possession: {label or '-'}", (view.shape[1] - 260, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOURS[label], 2)
            cv2.imshow(win, view)

        wait = 1 if paused else max(1, int(1000 / (fps * speed) - (time.perf_counter() - t0) * 1000))
        key = cv2.waitKeyEx(wait if not paused else 30)
        lower = ord(chr(key).lower()) if 0 <= key < 128 else key

        if lower in LABELS:
            add_event(events, frame_idx, fps, LABELS[lower])
            save_events(out, events)
        elif lower == ord("u") and events:
            events.pop()
            save_events(out, events)
        elif lower == ord(" "):
            paused = not paused
        elif lower in (ord("q"), 27):
            break
        elif key in KEYS_LEFT:
            seek(frame_idx - seek_frames)
        elif key in KEYS_RIGHT:
            seek(frame_idx + seek_frames)
        elif lower == ord(",") and paused:
            seek(frame_idx - 1)
        elif lower == ord(".") and paused:
            seek(frame_idx + 1)
        elif lower in (ord("-"), ord("_")):
            speed = max(0.25, speed / 2)
        elif lower in (ord("+"), ord("=")):
            speed = min(8, speed * 2)
        elif not paused:
            ok, frame = cap.read()
            if not ok:
                paused = True
            else:
                img, frame_idx = frame, frame_idx + 1
        if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
            break

    save_events(out, events)
    cap.release()
    cv2.destroyAllWindows()
    print(f"saved {len(events)} tags to {out}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("clip", type=Path)
    p.add_argument("-o", "--output", type=Path, help="default: data/labels/<clip stem>.csv")
    p.add_argument("--config", default="config.yaml")
    args = p.parse_args()
    out = args.output or Path("data/labels") / f"{args.clip.stem}.csv"
    run(args.clip, out, load_config(args.config)["tag"])


if __name__ == "__main__":
    main()
