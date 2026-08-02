# Experience digest — overnight program 2026-08-01/02 (preliminary)

What one night of autonomous search taught the system. Every claim cites its run; all of
this is folded into the agents' system context via `build_context.py` (`sec_learnings`).

## Winner recipes per dimension (v2 reward `(w·target + GENU + BLEND) × (1−WER)`)

| dimension | recipe | evidence |
|---|---|---|
| Arousal | `vn_DARC_high@1.0` | +0.65 over baseline, `runs/singledim27_arousal` |
| Valence | `vn_VALN_high@1.25 + Elation@0.5` | `runs/singledim27_valence` |
| Explicitness | `vn_EXPL_high@0.98 + vn_R_ORAL_high@0.75`, sultry-whisper framing | ~4.6× baseline, `runs/singledim_explicit` |
| Storytelling | `vn_S_STRY_high@0.75` | 0.75 beats 1.0 on WER, `runs/singledim27_storyteller` |

**Stabilizer adapters** (31B's independent discovery, `MODEL_COMPARISON.md`):
`vn_ARSH_high` / `vn_BRGT_high` at moderate scale protect intelligibility at high
energy — try them before lowering emotion doses.

**Reward verdict:** the ×(1−WER) multiplier works — it is what stopped over-acted
unintelligible winners (contrast: the pre-WER combined-objective winner).

## Task-difficulty map (8-task edge-case swarm)

**Solvable well:** hoarse voice (T0004) · calm narration under tension / bomb-squad
(T0007) · fading/death voice (T0008).

**Engine-limited** — flag early instead of burning budget: tongue-clicks mid-speech
(T0002) · genuine crying mid-sentence / eulogy breakdown (T0003) · effort-speech while
climbing (T0005) · jump-scare bursts (T0006, engine truncation on burst-heavy prompts).

## Honest under-performance notes

- 12B brain searches weakly (repeats strategies; 42 tool calls vs the 31B's 22 on
  identical missions) — kept only as the low-VRAM option.
- Quasi-periodic takes (fading voices) can falsely trip the self-repetition detector —
  verify before discarding such takes.
- MoE arm lost one final report to an LLM-400 context overflow (now guarded in
  `agent.py`).
