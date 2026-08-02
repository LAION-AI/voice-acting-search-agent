#!/bin/bash
# S2 arm LLM: the large dense Gemma-4 (the family's "27B-class" model is 31B) —
# google/gemma-4-31B-it-qat-w4a16-ct (official QAT w4a16 compressed-tensors, ~17GB).
# Serves :8802 on GPU0. gpu_memory_utilization 0.35 (28GB) leaves ~50GB so a TTS
# engine stack could co-reside if a stage needs it.
# Usage: bash scripts/vllm_27b.sh [gpu]  (default 0)
GPU=${1:-0}
source /home/deployer/miniconda3/etc/profile.d/conda.sh
conda activate vllm-serve
export HF_HOME=/tmp/hf_cache HF_HUB_DISABLE_XET=1
exec env CUDA_VISIBLE_DEVICES=$GPU python -m vllm.entrypoints.openai.api_server \
  --model google/gemma-4-31B-it-qat-w4a16-ct \
  --host 127.0.0.1 --port 8802 \
  --gpu-memory-utilization 0.5 \
  --max-model-len 65536 \
  --max-num-seqs 16 \
  --enable-prefix-caching
