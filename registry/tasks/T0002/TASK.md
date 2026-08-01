---
task_id: T0002
title: "Tongue clicking during speech"
tags: [voice-craft, edge_case, with_reference]
status: OPEN
owner: null
heartbeat: 2026-08-01T23:37:02Z
budget: {generations: 6, supervisor_rounds: 6}
operationalization:
  maximize: agent-chosen from detector tokens
  constraints: [{code: GENU, min: baseline}, {code: BLEND, min: baseline}]
edge_case_id: 227
reference: /tmp/evc_refs/ref1.wav
wer_multiplier: true
resume_hint: null
---

# Tongue clicking during speech

Family: Voice-Craft, Bursts & Delivery

Coach: A punctuation of the mouth — disapproval, thinking, or habit, ticking between words.

Acoustic signature: Sharp inter-word click transients, otherwise normal speech.

Rank by: Named burst family — detect, count, Blend-rate placement; fully coverable.
