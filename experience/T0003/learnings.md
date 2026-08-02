# Learnings - T0003

Exported 2026-08-02T04:19:42Z. Tool calls: 26. Generations: 12.

Fitness: start 2.061 -> best 5.641 (mean path [0.097, -0.149, 1.032, 1.143, 2.014, 2.636, 3.181, 3.398, 3.003, 3.572, 4.014, 3.587])

Supervisor score path (local): [0, 2, 2, 5, 4, 4, 6, 5, 2, 7, 7]

## Agent final report

MISSION REPORT: Breaking down mid-eulogy

1. Final Recipe:
- Reference: ref0 (Long-phrased cinematic)
- LoRA Merge: vn_S_FORM_high (0.8), Sadness (0.8), vn_RESP_high (1.0), vn_S_WHIS_high (0.8)
- Prompt: '(Resonant formal projection for the first half then a sudden violent vocal crack into sobbing followed by a long trembling gasp for air and ending in a tiny fragile hushed whisper)'
- Text: 'He was a man of unwavering strength a man who taught us that love is the only thing that lasts I just cannot go on love is all we have left' (no punctuation to avoid early EOS)
- Sampling: temp 0.9, top_p 0.9, max_frames 1200

2. Trajectory:
- Gen 0-3: Focused on S_FORM and Sadness. Encountered severe truncation at ~2 seconds.
- Gen 4-6: Attempted to solve truncation via punctuation removal and temperature increases (1.2). Successfully extended length but introduced 'squeaky' synthetic artifacts due to high LoRA scales.
- Gen 7-8: Stabilized quality by lowering temperature (0.5) and scaling back LoRAs. Recovered S_FORM and BLEND, but the specific 'crack-pause-recovery' arc remained elusive as the model tended to either be too stable or too chaotic.
- Final Attempt: Integrated ref0 to leverage long-phrasing bias and removed all punctuation. Achieved the best balance of formal resonance and emotional blend (s0700).

3. Technical Constraints:
The 'truncation wall' was the primary obstacle. The model consistently predicted EOS tokens immediately following the first phrase of the eulogy, regardless of frame budget or prompt. The most successful takes (s0680, s0660) captured the emotional 'flavor' of the breakdown, but the full temporal arc was limited by the model's internal sequence length preferences for this specific emotional profile.
