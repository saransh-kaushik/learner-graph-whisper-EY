"""Local entry point: python local/main.py --file_path input.wav ..."""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from core.pipeline import WhisperDiarizationPipeline

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Local runner for Whisper + Diarization")
    p.add_argument("--file_string", type=str, default=None)
    p.add_argument("--file_url", type=str, default=None)
    p.add_argument("--file_path", type=str, default=None)
    p.add_argument("--num_speakers", type=int, default=None)
    p.add_argument("--translate", type=bool, default=False)
    p.add_argument("--language", type=str, default=None)
    p.add_argument("--prompt", type=str, default=None)
    p.add_argument("--preprocess", type=int, default=0, choices=[0,1,2,3,4])
    p.add_argument("--highpass_freq", type=int, default=45)
    p.add_argument("--lowpass_freq", type=int, default=8000)
    p.add_argument("--prop_decrease", type=float, default=0.3)
    p.add_argument("--stationary", type=bool, default=True)
    p.add_argument("--target_dBFS", type=float, default=-18.0)
    p.add_argument("--device", type=str, default="cpu", choices=["cpu","cuda"])
    p.add_argument("--compute_type", type=str, default="int8")
    p.add_argument(
        "--output", type=str, default=None,
        help=(
            "Base path for output files (no extension). "
            "Saves <output>.json and <output>.txt. "
            "If omitted, JSON is printed to stdout."
        ),
    )
    return p


def _seconds_to_hms(seconds: float) -> str:
    """Convert a float number of seconds to HH:MM:SS string."""
    total = int(seconds)
    h = total // 3600
    m = (total % 3600) // 60
    s = total % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def _build_transcript_txt(segments: list) -> str:
    """Return a plain-text transcript, one line per segment."""
    lines = []
    for seg in segments:
        timestamp = _seconds_to_hms(seg["start"])
        speaker = seg.get("speaker", "UNKNOWN")
        text = seg.get("text", "").strip()
        lines.append(f"[{timestamp}] {speaker}: {text}")
    return "\n".join(lines)


def main():
    args = build_parser().parse_args()
    print(f"DEBUG --> Args: {args}", file=sys.stderr)
    pipeline = WhisperDiarizationPipeline(
        device=args.device,
        compute_type=args.compute_type,
    )

    result = pipeline.predict(
        file_string=args.file_string,
        file_url=args.file_url,
        file_path=args.file_path,
        num_speakers=args.num_speakers,
        translate=args.translate,
        language=args.language,
        prompt=args.prompt,
        preprocess=args.preprocess,
        highpass_freq=args.highpass_freq,
        lowpass_freq=args.lowpass_freq,
        prop_decrease=args.prop_decrease,
        stationary=args.stationary,
        target_dBFS=args.target_dBFS,
    )

    result_dict = result.to_dict()
    json_str = json.dumps(result_dict, indent=2, ensure_ascii=False)
    txt_str = _build_transcript_txt(result_dict.get("segments", []))

    if args.output:
        base = Path(args.output)
        base.parent.mkdir(parents=True, exist_ok=True)

        json_path = base.with_suffix(".json")
        txt_path = base.with_suffix(".txt")

        json_path.write_text(json_str, encoding="utf-8")
        txt_path.write_text(txt_str, encoding="utf-8")

        print(f"Saved JSON → {json_path}", file=sys.stderr)
        print(f"Saved TXT  → {txt_path}", file=sys.stderr)
    else:
        print(json_str)

if __name__ == "__main__":
    main()
