#!/bin/bash
# S3 arm LLM: Gemma-4 MoE instruct — google/gemma-4-26B-A4B-it (26B total / 4B active).
# Primary: cyankiwi/gemma-4-26B-A4B-it-AWQ-4bit (~15GB, vLLM AWQ-Marlin on A100).
# Fallback (set MOE_MODEL): google/gemma-4-26B-A4B-it-qat-q4_0-unquantized (bf16 ~26GB).
# Serves :8803. gpu_memory_utilization 0.35 leaves room for an engine stack.
# Usage: bash scripts/vllm_moe.sh [gpu]  (default 0)
GPU=${1:-0}
MODEL=${MOE_MODEL:-cyankiwi/gemma-4-26B-A4B-it-AWQ-4bit}
source /home/deployer/miniconda3/etc/profile.d/conda.sh
conda activate vllm-serve
export HF_HOME=/tmp/hf_cache HF_HUB_DISABLE_XET=1
exec env CUDA_VISIBLE_DEVICES=$GPU python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" \
  --host 127.0.0.1 --port 8803 \
  --gpu-memory-utilization 0.35 \
  --max-model-len 65536 \
  --max-num-seqs 16 \
  --enable-prefix-caching
