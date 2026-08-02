# Learnings - T0005

Exported 2026-08-02T04:24:50Z. Tool calls: 27. Generations: 13.

Fitness: start 4.923 -> best 5.007 (mean path [4.337, 3.939, 4.029, 3.425, 2.852, 3.668, 3.562, 4.024, 4.532, 4.07, 4.127, 3.973, 4.468])

Supervisor score path (local): [5, 7, 7, 6, 7, 6, 6, 7, 3, 8, 5, 7]

## Agent final report

MISSION COMPLETE: 'Speaking while climbing' (Physical Exertion & Environment).

FINAL WINNING RECIPE:
- LoRAs: emotion_Fatigue_Exhaustion (1.2), vn_RESP_high (0.8), vn_TENS_high (0.3)
- Prompt: 'compressed vocal delivery, visceral weight-bearing grunts, sharp inhaled resets, low volume'
- Script: 'One more reach <gasp> just one more <grunt> almost there <grunt> just gotta get a grip <gasp> on this ledge. Steady now.'
- Sampling: temp=0.9, top_p=0.95, top_k=50

KEY FINDINGS:
- Resonance vs. Exertion: High chest resonance (vn_R_CHST_high) tended to trigger 'theatrical shouting' and repetitive loops. Shifting the focus to 'air-starvation' via emotion_Fatigue_Exhaustion and vn_RESP_high produced a more realistic, compressed vocal tract.
- Burst Integration: Placing burst tags (<gasp>, <grunt>) directly inside the script text was critical for breaking repetitive patterns and forcing the model to interleave speech with physical effort.
- Signal Stability: High-pitch modifiers (vn_GEND_high) caused catastrophic failure (whistling artifacts). Low temperature (0.7-0.9) combined with higher top_k/top_p was necessary to maintain articulation while allowing for physical jitter.

HALL-OF-FAME SAMPLES:
- s0820: Best overall balance of respiratory resets and blend.
- s0796: Most intense respiratory effort (RESP 0.75) and pitch jitter.
- s0772: Highest genuineness and stability.
