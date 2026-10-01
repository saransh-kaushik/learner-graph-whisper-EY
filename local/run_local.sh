#!/usr/bin/env bash
# Simple helper to create venv, install requirements and run a test transcription.
# Usage: ./run_local.sh /path/to/input.wav
# Must be run from the project root.
set -euo pipefail

INPUT_FILE=${1:-}
if [ -z "$INPUT_FILE" ]; then
  echo "Usage: $0 /path/to/input.wav"
  echo "Example: $0 ./audio.ogg"
  exit 1
fi

echo "Creating and setting up virtual environment..."
python3.11 -m venv .venv
source .venv/bin/activate

echo "Installing project requirements..."
pip install -r ./requirements.txt

echo "Downloading models..."
MODEL_DIR="$(cd "$(dirname "$0")/.." && pwd)/model_cache"
mkdir -p "$MODEL_DIR/whisper" "$MODEL_DIR/diarization"

echo "Downloading Whisper model (small)..."
python -c "from faster_whisper import download_model; download_model('small', output_dir='$MODEL_DIR/whisper/small')"

echo "Downloading Pyannote diarization model..."
export HF_TOKEN=$(cat /run/secrets/HF_TOKEN 2>/dev/null || echo "hf_TOKEN_HERE")
python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='pyannote/speaker-diarization-community-1', local_dir='$MODEL_DIR/diarization/pyannote--speaker-diarization-community-1', token='$HF_TOKEN')"

echo "Running transcription on: $INPUT_FILE"
python3 ./main.py --file_path "$INPUT_FILE"
