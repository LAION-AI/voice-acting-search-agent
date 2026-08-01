#!/bin/bash
# Phase C bring-up: three short-budget canary missions on one GPU.
# Usage: bash scripts/run_canaries.sh [gpu]
set -uo pipefail
cd "$(dirname "$0")/.."
set -a; [ -f .env ] && source .env; set +a
GPU=${1:-1}

run() {  # name mission budget
  local name=$1 mission=$2 budget=$3
  echo "=== CANARY $name ==="
  python run_agent.py --gpu "$GPU" --budget-tool-calls "$budget" \
    --workdir "runs/canary_$name" --mission "$mission" 2>&1 | tee "runs/canary_$name.log"
}

run arousal_ref \
  "Make the reference voice 'ref1' express strongly RAISED Arousal (AROU) while staying recognizably the same speaker (check speaker_sim). Establish a baseline first, explore 2-3 strategies from the manual, verify with score, save the 2 best samples to the hall of fame, then finish with a report of the winning recipe (LoRAs+scales, prompt, sampling, scores)." \
  22

run evil_ghost \
  "Create a convincing 'evil ghost voice': spectral, hollow, menacing, otherworldly, whispery yet threatening. No reference voice. Consider character LoRAs (list them), emotion LoRAs like Malevolence_Malice or Fear, and VoiceNet dims; verify with caption + score, iterate at least twice, save the 2 best samples, then finish reporting the winning recipe." \
  22

run fear_scream \
  "Produce fear-filled SCREAMING with real vocal bursts. Start from the fear_scream edge recipe genome in your context, push Fear strength and burst blend (BLEND) as high as possible (WER is allowed to be ~1 here), compare at least two variants, save the 2 best samples, then finish with the winning genome." \
  18
