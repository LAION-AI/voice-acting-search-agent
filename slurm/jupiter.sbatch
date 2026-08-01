#!/bin/bash
#SBATCH --job-name=vaa-search
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --time=12:00:00
#SBATCH --output=vaa-%j.out

# Jupiter 4-GPU topology: GPU0 = shared vLLM brain, GPU1-3 = one agent worker each.
# Prereqs (offline nodes): pre-populate $HF_HOME with the models listed in README
# ("Offline cache manifest") and clone LAION-AI/procedural-voice-captions next to
# this repo; adjust configs/node_4gpu.yaml paths for the node's local storage.

set -euo pipefail
source ~/.bashrc
set -a; [ -f .env ] && source .env; set +a
export HF_HOME=${HF_HOME:-/tmp/hf_cache}
export HF_HUB_DISABLE_XET=1

CFG=configs/node_4gpu.yaml
LLM_PY=$(python -c "import yaml;print(yaml.safe_load(open('$CFG'))['llm']['vllm_python'])")
MODEL=$(python -c "import yaml;print(yaml.safe_load(open('$CFG'))['llm']['model'])")
PORT=$(python -c "import yaml;print(yaml.safe_load(open('$CFG'))['llm']['port'])")

# 1) shared LLM on GPU0
CUDA_VISIBLE_DEVICES=0 $LLM_PY -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" --port "$PORT" --gpu-memory-utilization 0.90 \
  --max-model-len 65536 --disable-log-requests &
LLM_PID=$!
until curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null; do sleep 5; done

# 2) one worker per remaining GPU. Missions come from missions.txt (one per line).
mapfile -t MISSIONS < missions.txt
PIDS=()
for g in 1 2 3; do
  i=$((g-1))
  [ -z "${MISSIONS[$i]:-}" ] && continue
  python run_agent.py --config "$CFG" --gpu "$g" --no-start-llm \
    --mission "${MISSIONS[$i]}" --workdir "runs/jupiter-${SLURM_JOB_ID}-g$g" &
  PIDS+=($!)
done
for p in "${PIDS[@]}"; do wait "$p"; done
kill $LLM_PID || true
