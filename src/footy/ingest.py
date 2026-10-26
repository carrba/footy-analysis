"""Trim a clip from a Veo export and resample it to a configurable fps."""
import argparse
import json
import subprocess
from pathlib import Path

import yaml


def load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_command(src: Path, dst: Path, cfg: dict) -> list[str]:
    fps = cfg["fps"]
    return [
        "ffmpeg", "-hide_banner", "-y",
        "-ss", str(cfg["start_seconds"]),  # before -i: fast input seek
        "-i", str(src),
        "-t", str(cfg["clip_seconds"]),
        "-vf", f"fps={fps}",
        "-an",
        "-c:v", "libx264", "-preset", cfg["preset"], "-crf", str(cfg["crf"]),
        "-g", str(round(fps * cfg["keyframe_seconds"])),
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        str(dst),
    ]


def ingest(src: Path, dst: Path, cfg: dict) -> Path:
    dst.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(build_command(src, dst, cfg), check=True)
    meta = {"source": str(src), "output": str(dst), **{k: cfg[k] for k in ("fps", "start_seconds", "clip_seconds")}}
    dst.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    return dst


def extract_audio(src: Path, dst: Path, cfg: dict) -> Path:
    """Mono WAV covering the same window as the clip, so audio time == clip time."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([
        "ffmpeg", "-hide_banner", "-y",
        "-ss", str(cfg["start_seconds"]), "-i", str(src), "-t", str(cfg["clip_seconds"]),
        "-vn", "-ac", "1", "-ar", str(cfg["audio_sr"]), "-c:a", "pcm_s16le",
        str(dst),
    ], check=True)
    return dst


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("input", type=Path)
    p.add_argument("-o", "--output", type=Path, help="default: data/clips/<input stem>_<start>s_<fps>fps.mp4")
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--fps", type=float)
    p.add_argument("--start", type=float, help="start offset in seconds")
    p.add_argument("--duration", type=float, help="clip length in seconds")
    p.add_argument("--crf", type=int, help="x264 quality; raise (e.g. 23) for full matches to keep the file small")
    p.add_argument("--audio-only", action="store_true", help="extract the WAV only, skip video re-encode")
    args = p.parse_args()

    cfg = load_config(args.config)["ingest"]
    for key, val in (("fps", args.fps), ("start_seconds", args.start), ("clip_seconds", args.duration), ("crf", args.crf)):
        if val is not None:
            cfg[key] = val

    dst = args.output or Path("data/clips") / f"{args.input.stem}_{cfg['start_seconds']:g}s_{cfg['fps']:g}fps.mp4"
    wav = dst.with_suffix(".wav")
    if args.audio_only:
        print(f"wrote {extract_audio(args.input, wav, cfg)}")
        return
    print(f"wrote {ingest(args.input, dst, cfg)}")
    print(f"wrote {extract_audio(args.input, wav, cfg)}")


if __name__ == "__main__":
    main()
