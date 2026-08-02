---
task_id: T0004
title: "Hoarse voice"
tags: [body, edge_case, with_reference]
status: REVIEW
owner: null
heartbeat: 2026-08-02T04:09:35Z
budget: {generations: 6, supervisor_rounds: 6}
operationalization:
  maximize: agent-chosen from detector tokens
  constraints: [{code: GENU, min: baseline}, {code: BLEND, min: baseline}]
edge_case_id: 26
reference: /tmp/evc_refs/ref3.wav
wer_multiplier: true
resume_hint: null
---

# Hoarse voice

Family: Body, Illness, Pain & Substances

Coach: The instrument is bruised. Every note comes with grit; the top of the range is gone.

Acoustic signature: High roughness, broken harmonicity, breathy leakage, reduced brightness, extra effort for volume.

Rank by: Roughness × harmonicity-loss × brightness-drop is a clean VoiceNet superposition; Genuineness separates real rasp from an actor's growl.
