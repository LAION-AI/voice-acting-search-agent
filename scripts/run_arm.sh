#!/bin/bash
# Run one model arm = the SAME 4 single-dimension missions against a given LLM port.
#   bash scripts/run_arm.sh <tag> <port> [gpu_list]
#   e.g. bash scripts/run_arm.sh 27 8802 "3 4 5 6"   -> runs/singledim27_*
#        bash scripts/run_arm.sh moe 8803            -> runs/singledimmoe_*
# Missions/budget identical to the 12B arm (fairness): 60 tool calls, 8 gens x 8 genomes.
set -u
TAG=$1; PORT=$2; GPUS=(${3:-3 4 5 6})
# PORT may be a full base URL (external API arm), e.g. https://api.hyprlab.io/v1 —
# then also export LLM_MODEL (pin), LLM_API_KEY (Bearer), LLM_REASONING_EFFORT before calling.
case "$PORT" in
  http*) LLMURL="$PORT" ;;
  *)     LLMURL="http://127.0.0.1:${PORT}/v1" ;;
esac
exec >> /tmp/vaa_arm_${TAG}.log 2>&1
source /home/deployer/miniconda3/etc/profile.d/conda.sh; conda activate ml-general
export HF_HOME=/tmp/hf_cache HF_HUB_DISABLE_XET=1 TOKENIZERS_PARALLELISM=false
cd /run/user/1001/vaa_build/voice-acting-search-agent
set -a; [ -f .env ] && source .env; set +a
echo "[arm-$TAG] START $(date) port=$PORT gpus=${GPUS[*]}"

declare -A M
M[arousal]="SINGLE-DIMENSION optimization: maximize Arousal (AROU) as the ONLY target dimension — while the voice stays aesthetic, genuine and ORGANIC (this is the corrected experiment: one dimension per run, not combined). Use run_generation with fitness {maximize:{AROU:1.5}, wer_multiplier:true} (defaults add GENU+BLEND at half the target weight each). Evolution: 8 generations x 8 genomes, mean-of-8 cohorts. Compare at least: vn_AROU_high dose sweep, a related emotion-LoRA route, and prompt-only. Finish with a report of the TOP-3 recipes ranked by cohort mean fitness, each with full cohort stats incl. WER, and save_best the top samples."
M[valence]="SINGLE-DIMENSION optimization: maximize Valence (VALN) as the ONLY target dimension — while the voice stays aesthetic, genuine and ORGANIC. Use run_generation with fitness {maximize:{VALN:1.5}, wer_multiplier:true}. Evolution: 8 generations x 8 genomes, mean-of-8. Compare vn_VALN_high doses, a positive-emotion LoRA route (e.g. Contentment/Elation at moderate scales), and prompt-only. Finish with TOP-3 recipes + full cohort stats incl. WER, save_best."
M[explicit]="SINGLE-DIMENSION optimization: maximize Content Explicitness (EXPL) as the ONLY target dimension — while the voice stays aesthetic, genuine and ORGANIC. Use run_generation with fitness {maximize:{EXPL:2.0}, wer_multiplier:true}. Evolution: 8 generations x 8 genomes, mean-of-8. EXPL is known to be a stubborn axis - be creative: vn_EXPL_high doses, carrier-text choice (the TEXT itself may carry explicit innuendo framing), delivery styles. Finish with TOP-3 recipes + full cohort stats incl. WER, save_best."
M[storyteller]="SINGLE-DIMENSION optimization: maximize Storytelling style (S_STRY) as the ONLY target dimension — while the voice stays aesthetic, genuine and ORGANIC (NOT overacted: the (1-WER) factor protects intelligibility). Use run_generation with fitness {maximize:{S_STRY:1.5}, wer_multiplier:true}. Evolution: 8 generations x 8 genomes, mean-of-8. Compare vn_S_STRY_high doses, vn_S_NARR_high, and prompt-only narration framing. Finish with TOP-3 recipes + full cohort stats incl. WER, save_best."

i=0
for k in arousal valence explicit storyteller; do
  g=${GPUS[$i]}; i=$((i+1))
  nohup python -u run_agent.py --mission "${M[$k]}" --budget-tool-calls 60 --gpu "$g" \
    --no-start-llm --llm-url "$LLMURL" \
    --workdir "runs/singledim${TAG}_$k" > "/tmp/vaa_arm_${TAG}_$k.log" 2>&1 &
  echo "[arm-$TAG] launched $k on GPU$g pid $!"
  sleep 20
done
wait
echo "[arm-$TAG] ALL MISSIONS DONE $(date)"
