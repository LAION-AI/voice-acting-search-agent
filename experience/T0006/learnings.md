# Learnings - T0006

Exported 2026-08-02T04:23:46Z. Tool calls: 27. Generations: 12.

Fitness: start 4.312 -> best 4.394 (mean path [3.833, 3.781, 3.823, 2.723, 2.777, 3.089, 2.924, 2.731, 2.461, 3.057, 2.878, 2.793])

Supervisor score path (local): [6, 2, 4, 5, 3, 2, 7, 2, 5, 2, 0, 0]

## Agent final report

MISSION REPORT: Jump-scare / sudden appearance (Consciousness, Shock & Altered States).

FINAL STATUS: Partial Success / Technical Failure.

SUMMARY:
Targeted a complex four-stage arc: [Involuntary Gasp] -> [Silence] -> [Shaky Recovery] -> [Nervous Laugh]. While the individual components (explosive gasp and shaky recovery) were achieved in isolation, the model struggled to maintain synthesis stability across the full sequence, frequently resulting in either premature truncation or digital corruption (broadband static).

BEST GENOMES:
1. 'The Trembling Recovery' (S0529): Best overall narrative arc completion, though the initial shock was under-powered.
2. 'Hybrid-Shatter v1' (S0610): Best explosive, visceral onset, but failed to sustain the recovery speech.
3. 'Clean-Shatter Final' (S0673): Best synthesis stability and breath-seize texture, though truncated.

WINNING STABILITY RECIPE:
- LoRAs: vn_VULN_high (0.8), vn_RESP_high (0.8), vn_S_WHIS_high (0.5)
- Sampling: temp 0.6, top_p 0.8, top_k 25
- Prompt: 'A sudden, violent, non-vocalized gasp of shock, followed by a beat of silence, then a fragile, shaking recovery ending in a nervous laugh.'

HALL-OF-FAME SAMPLES:
- s0529: Most complete narrative arc.
- s0610: Most visceral jump-scare onset.
- s0673: Cleanest breath-seize texture.

TECHNICAL FINDINGS:
- High-intensity onset LoRAs (ARSH/ATCK) are highly prone to causing synthesis collapse (digital buzzing/static) when combined with reference voices of high resonance (ref5).
- The 'silence' beat in the script often triggers early truncation in the generator, suggesting a limitation in the model's ability to handle extended non-speech pauses within a single generation window.
