"""Detect ball and players per frame; write a detections table and an annotated preview video."""
import argparse
from pathlib import Path

import cv2
import pandas as pd
import supervision as sv
from ultralytics import YOLO
from ultralytics.utils.downloads import attempt_download_asset

from footy.ingest import load_config

COCO_PERSON, COCO_BALL = 0, 32
CLASS_NAMES = {COCO_PERSON: "player", COCO_BALL: "ball"}


def detect(clip: Path, cfg: dict, out_dir: Path, preview: bool = True) -> pd.DataFrame:
    weights = Path("models") / cfg["model"]
    weights.parent.mkdir(exist_ok=True)
    if not weights.exists():
        attempt_download_asset(str(weights))
    model = YOLO(str(weights))

    cap = cv2.VideoCapture(str(clip))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    scale = cfg["preview_width"] / w
    writer = None
    if preview:
        out_dir.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(str(out_dir / f"{clip.stem}_detections.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
                                 fps, (cfg["preview_width"], round(h * scale)))
    box_ann, label_ann = sv.BoxAnnotator(thickness=2), sv.LabelAnnotator(text_scale=0.4)

    min_conf = min(cfg["ball_conf"], cfg["player_conf"])
    rows = []
    for idx in range(total):
        ok, frame = cap.read()
        if not ok:
            break
        res = model.predict(frame, imgsz=cfg["imgsz"], conf=min_conf, classes=[COCO_PERSON, COCO_BALL],
                            device=cfg["device"], verbose=False)[0]
        dets = sv.Detections.from_ultralytics(res)
        keep = ((dets.class_id == COCO_BALL) & (dets.confidence >= cfg["ball_conf"])) | \
               ((dets.class_id == COCO_PERSON) & (dets.confidence >= cfg["player_conf"]))
        dets = dets[keep]
        for (x1, y1, x2, y2), c, k in zip(dets.xyxy, dets.confidence, dets.class_id):
            rows.append({"frame": idx, "time_s": round(idx / fps, 3), "cls": CLASS_NAMES[int(k)],
                         "conf": float(c), "x1": float(x1), "y1": float(y1), "x2": float(x2), "y2": float(y2)})
        if writer:
            labels = [f"{CLASS_NAMES[int(k)]} {c:.2f}" for k, c in zip(dets.class_id, dets.confidence)]
            vis = label_ann.annotate(box_ann.annotate(frame.copy(), dets), dets, labels)
            writer.write(cv2.resize(vis, (cfg["preview_width"], round(h * scale))))
        if idx % 100 == 0:
            print(f"{idx}/{total}", flush=True)
    if writer:
        writer.release()

    df = pd.DataFrame(rows, columns=["frame", "time_s", "cls", "conf", "x1", "y1", "x2", "y2"])
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_dir / f"{clip.stem}_detections.parquet")
    report(df, total)
    return df


def report(df: pd.DataFrame, total: int) -> None:
    ball = df[df.cls == "ball"]
    players = df[df.cls == "player"].groupby("frame").size().reindex(range(total), fill_value=0)
    print(f"frames: {total}")
    print(f"ball coverage: {ball.frame.nunique() / total:.1%} of frames ({ball.frame.nunique()}/{total})")
    print(f"ball conf (best per frame): median {ball.groupby('frame').conf.max().median():.2f}" if len(ball) else "no ball detections")
    print(f"players per frame: median {players.median():.0f}, max {players.max()}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("clip", type=Path)
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--out", type=Path, default=Path("outputs"))
    p.add_argument("--no-preview", action="store_true")
    p.add_argument("--device", help='override detect.device, e.g. "cpu" or "0" for the first GPU')
    args = p.parse_args()
    cfg = load_config(args.config)["detect"]
    if args.device is not None:
        cfg["device"] = args.device
    detect(args.clip, cfg, args.out, preview=not args.no_preview)


if __name__ == "__main__":
    main()
