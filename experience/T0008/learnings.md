# Learnings - T0008

Exported 2026-08-02T04:52:17Z. Tool calls: 12. Generations: 5.

Fitness: start 0.295 -> best 0.357 (mean path [-0.056, 0.019, -0.146, 0.162, -0.038])

Supervisor score path (local): [7, 8, 7, 8]

## Agent final report

MISSION SUCCESS: 'Death scene / fading voice' achieved.

Winning Recipe:
- Reference: ref1
- LoRAs: vn_S_WHIS_high (0.4), vn_RESP_high (0.4)
- Prompt: GENERAL: A voice fading into death, growing weaker and quieter\nSCRIPT:\n(weakly, gradually losing volume phrase by phrase, widening pauses, ending in a long, audible, smooth hiss of air that fades slowly to silence)
- Sampling: temp=0.7, top_p=0.9, top_k=40

Trajectory:
1. Baseline established; initial attempts with Fatigue/Numbness LoRAs failed due to high Arousal and Dynamic Arc.
2. Prompt steering ('unvoiced exhale') proved to be the primary driver of the fading effect, but led to premature truncation.
3. Solved truncation by increasing max_frames and refining the 'gradual' nature of the prompt.
4. Final polish focused on the 'audible air decay' at the end, using a blend of Whisper and Respiration LoRAs to ensure the breath outlasted the voice.

Key Metrics (Best Sample s0240):
- Dynamic Arc: 0.051 (Strongly monotonic/low)
- Arousal: 0.084 (Deeply subdued)
- WER: 0.076 (High intelligibility preserved)
- Blend: 0.377
- Supervisor Score: 8/10
