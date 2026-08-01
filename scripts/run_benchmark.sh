#!/bin/bash
# Phase D benchmark mission: AROU+VALN+EXPL+S_STRY, constraints GENU/BLEND >= baseline,
# full 10x8 evolution, mean-of-8.
set -uo pipefail
cd "$(dirname "$0")/.."
set -a; [ -f .env ] && source .env; set +a
GPU=${1:-1}

python run_agent.py --gpu "$GPU" --budget-tool-calls 45 \
  --workdir runs/benchmark_arou_valn_expl_stry \
  --mission "BENCHMARK: Maximize Arousal (AROU), Valence (VALN), Content Explicitness (EXPL) and Storytelling style (S_STRY) simultaneously, subject to genuineness (GENU) and vocal-burst blend (BLEND) staying at or above the no-LoRA baseline. Use the FULL evolution protocol: 10 generations x 8 genomes, mean-of-8 fitness. Procedure: (1) compute_baseline with your chosen >=20-word storytelling carrier text; (2) seed 8 genomes from the manuals (consider vn_AROU_high, vn_VALN_high, vn_EXPL_high, vn_S_STRY_high doses and emotion LoRAs like Elation/Triumph plus strong prompts); (3) run one run_generation per generation with fitness {maximize:[AROU,VALN,EXPL,S_STRY], constraints:[{code:GENU,min:baseline},{code:BLEND,min:baseline}]}; (4) between generations keep top-2, mutate 4 (scales +-0.25, reword desc/cue, swap a LoRA, temp +-0.1), inject 2 fresh; (5) log each generation's best/mean fitness and winning genome in memory; (6) save_best the top 3 samples at the end; (7) finish with a report: fitness per generation, best genome breakdown, which strategies won." \
  2>&1 | tee runs/benchmark.log
