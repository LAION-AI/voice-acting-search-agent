---
task_id: T0008
title: "Death scene / fading voice"
tags: [emotion, edge_case, with_reference]
status: REVIEW
owner: null
heartbeat: 2026-08-02T04:52:17Z
budget: {generations: 6, supervisor_rounds: 6}
operationalization:
  maximize: agent-chosen from detector tokens
  constraints: [{code: GENU, min: baseline}, {code: BLEND, min: baseline}]
edge_case_id: 169
reference: /tmp/evc_refs/ref1.wav
wer_multiplier: true
resume_hint: null
---

# Death scene / fading voice

Family: Emotion, Masking & the Social Game

Coach: The instrument runs out of air for good. Each phrase is smaller than the last, then only breath.

Acoustic signature: Monotonic falling dynamic arc to near-silence, lengthening pauses, breath outlasting voice, low arousal.

Rank by: The dying arc is a clean temporal-dynamics signature; rank by monotonic decay + breath-outlasting-voice.
