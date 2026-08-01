---
task_id: T0001
title: "Pretending angry while actually calm"
tags: [emotion, edge_case, no_reference]
status: OPEN
owner: null
heartbeat: 2026-08-01T23:37:02Z
budget: {generations: 6, supervisor_rounds: 6}
operationalization:
  maximize: agent-chosen from detector tokens
  constraints: [{code: GENU, min: baseline}, {code: BLEND, min: baseline}]
edge_case_id: 118
reference: null
wer_multiplier: true
resume_hint: null
---

# Pretending angry while actually calm

Family: Emotion, Masking & the Social Game

Coach: The costume of anger over a resting pulse. The words shout; the body doesn't.

Acoustic signature: Loud/sharp surface but steady breath, low real tension, no arousal follow-through — the tell is the mismatch.

Rank by: ★ ★ The superposition win: rank by EmoNet-Anger(surface) high × Genuineness low × physiological-arousal low. High incongruence = convincing 'fake anger'.
