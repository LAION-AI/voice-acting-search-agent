# Voice-Acting Search Agent — Experiment & Implementation Plan

Goal: a **fully self-sufficient, offline-capable agent** that runs on a single 80 GB GPU (A100) — or a
4-GPU node (Jupiter topology) — and autonomously searches for LoRA-combination + prompt strategies that
achieve a target vocal effect on `laion/moss-tts-local-transformer-4.55b-voice-acting-v2` (MOSS-VA-v2).
Example missions: "make this reference voice express extreme arousal", "evil ghost voice",
"fear-filled screaming", "natural laughter", "maximize AROU+VALN+EXPL+S_STRY while staying genuine".

## 1. Architecture

```
┌──────────────────────────── 1×A100-80GB ────────────────────────────┐
│  vLLM (Gemma-4-12B quant, ~12-16GB, gpu_mem_util≈0.22)              │
│  MOSS-VA-v2 bf16 (~10GB) + MOSS-Audio-Tokenizer-v2 (n_vq=12)        │
│  Scorer stack (~8GB): EmoNet experts (EIV-Plus), VoiceNet embed+57   │
│    reg MLPs, VoiceCLAP-commercial + genu/blend MLPs, quality MLPs,   │
│    ECAPA spk-sim, Parakeet (or faster-whisper fallback) ASR          │
│  Agent loop (CPU): ReAct JSON tool-calling, memory, transcripts      │
└──────────────────────────────────────────────────────────────────────┘

4-GPU node (Jupiter): GPU0 = vLLM serving the LLM for N agent workers;
GPU1-3 = one TTS+scorer worker each. Agents call the shared LLM via
localhost OpenAI-compatible API → LLM GPU stays busy, workers stay busy.
```

- **Why self-coded loop, not a framework**: must run air-gapped (no internet at runtime on Jupiter),
  deterministic deps, tight token budget, custom tool schema. Design borrows ReAct + tool-JSON from
  existing frameworks but stays ~500 lines, testable. (smolagents/etc. evaluated in an appendix; if one
  is clearly better offline, builder may adopt it — requirement is: memory, subagent spawning, robust
  JSON-tool parsing with retries, transcript logging.)
- **LLM brain**: start **Gemma-4-12B-it** (quantized AWQ/FP8, must run on vLLM). Later compare 27B dense
  and the MoE variant (Phase E). Fallback if Gemma 4 unavailable/broken on vLLM: Gemma-3-12B-it.

## 2. Tool API (the contract — every tool documented in the system context)

| tool | signature (JSON) | returns |
|---|---|---|
| `list_loras` | `{family?}` | catalog: 40 emotion v3 · 114 VoiceNet (57 dims × high/low) · 120 genuine char · 120 SIDON-refined char, each with known best-dose + side-effect summary from the manuals |
| `merge_loras` | `{loras:[{name,scale}]}` | activates a merge set (PEFT multi-adapter, scaling×base) for subsequent generate calls |
| `generate` | `{text,instruction,language,n=8,reference_id?,sampling{temp,top_p,top_k},max_frames}` | sample_ids + durations (batch gen; empty-retry) |
| `score` | `{sample_ids,metrics?}` | per-sample + mean/std: EmoNet-40, VoiceNet-57, genuineness, vocal-burst blend, quality (RCQL+ESTH), length |
| `transcribe` | `{sample_ids}` | 3× transcription + per-sample WER vs the prompted text |
| `caption` | `{sample_ids}` | procedural caption (procedural-voice-captions @ latest: locator v2, classifier v2, [pause] markers) |
| `speaker_sim` | `{sample_ids,reference_id}` | ECAPA cosine each + mean |
| `load_reference` | `{path}` | reference_id + its caption + scores (for conditioning + spk-sim) |
| `save_best` / `memory` | notes, hall-of-fame wavs + genomes | persisted to workdir |
| `spawn_subagent` | `{task,budget}` | fresh-context copy of self on same GPU worker, returns report |
| `fetch_manual` | `{}` | re-pulls the live manuals (voicenet-manual md/, voiceacting-manual md/, edge-case learnings) → refreshes system context |
| `push_results` | `{path,message}` | commits results dir to the agent repo (needs GH token in env) |

## 3. System context (~20-50k tokens, assembled, auto-refreshable)

1. **Mission preamble** + tool docs (hand-written, ~3k).
2. **Distilled conditioning manual** (auto-built script `build_context.py`):
   - per-emotion: best strategy ± LoRA, best λ, top-3 ± correlations (from moss-voiceacting-manual md/)
   - per-VoiceNet-dim: dose-response 25→125% high/low, best dose, Δgenu/Δblend/Δqual, top-3 ± EmoNet/VN
     correlations (from moss-voicenet-manual md/) — **key learnings**: __high monotonic & strong, __low weak
     (needs 125%); universal cost = burst-blend −0.4, genuineness robust; Triumph/Anger ↔ Contemplation axis
   - edge-case recipes (fear_scream, pain_scream, pain_groan, cold_shiver, amuse_laugh, sad_cry + evolved
     genomes: λ, cue, sampling)
   - prompting best practices: temps with/without reference (ref: temp 0.8-0.9/top_p 0.9; no-ref: 1.0/0.95/25),
     GENERAL/SCRIPT format, containment cues, carrier-sentence guidance (≥20 words), λ caps (emotion ≤1.9,
     VN high 0.75-1.25, VN low needs 1.25)
3. **Evolution protocol** (default): 10 generations × 8 genomes; genome = {loras+scales, desc, cue,
   sampling, text}; mean-of-8 fitness; keep top-2, mutate 4, fresh 2; report hall-of-fame.

## 4. Phases

- **A. Scaffold + tools** (repo `LAION-AI/voice-acting-search-agent`): engine.py (TTS+merge+gen),
  scorers.py (99-vec + WER + captions + spk-sim), tools.py (JSON API), agent.py (loop+memory+subagents),
  llm.py (vLLM client), build_context.py, configs for 1-GPU and 4-GPU. Unit smoke: each tool on GPU 1.
- **B. Context assembly + system prompt**: build_context.py → context/system_context.md; measure tokens;
  iterate until <50k.
- **C. Agent bring-up on GPU 1** (A100-80GB, free now): vLLM Gemma-4-12B quant + full stack co-resident;
  memory budget check; run 3 canary missions (raise AROU@ref; evil-ghost no-ref; fear-scream edge)
  short-budget; fix loop failures; iterate until the agent completes missions unaided.
- **D. Benchmark mission** (the user's test): *maximize Arousal + Valence + Content-Explicitness (EXPL)
  + Storytelling (S_STRY), constraints genuineness + burst-blend ≥ baseline* — full 10×8 evolution,
  mean-of-8, report + hall-of-fame audio, push to repo.
- **E. LLM comparison**: 12B vs 27B (vs MoE if available) on the same missions; wall-clock, tool-call
  validity rate, final fitness. 4-GPU topology test: 1×vLLM + 3 workers sharing it.
- **F. Jupiter packaging**: SLURM script, offline model cache manifest, docs.

## 5. Deliverables
- GitHub repo (code, configs, context builder, SLURM, README with GPU budgets)
- Working agent verified on GPU 1 + canary transcripts committed
- Benchmark results (D) + model comparison (E)
- Agent skill: fetch_manual/push_results contract documented
