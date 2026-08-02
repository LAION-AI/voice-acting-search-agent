---
task_id: T0005
title: "Speaking while climbing"
tags: [physical, edge_case, no_reference, wer_off]
status: REVIEW
owner: null
heartbeat: 2026-08-02T04:24:50Z
budget: {generations: 6, supervisor_rounds: 6}
operationalization:
  maximize: agent-chosen from detector tokens
  constraints: [{code: GENU, min: baseline}, {code: BLEND, min: baseline}]
edge_case_id: 0
reference: null
wer_multiplier: false
resume_hint: null
---

# Speaking while climbing

Family: Physical Exertion & Environment

Coach: The voice is a passenger on a working body. Every reach resets the breath; sentences bend around handholds.

Acoustic signature: Speech chopped into effort-sized units; inhales stolen mid-phrase; a grunt on each pull; pitch jitter from a braced core.

Rank by: Match the breath-rhythm signature × grunt-burst density × Blend of the grunts; Genuineness separates a real gasp from a decorative one.
