# voice-acting-search-agent

> **📄 [SWARM_PLAN.md](SWARM_PLAN.md)** — technical plan for scaling this into a 32–64-agent self-improving swarm (two-tier hive-mind knowledge base, task registry, supervisor loop, consolidation, auto-discovered tools/LoRAs).


A **fully autonomous, offline-capable search agent** that runs on a single A100-80GB (or a
4-GPU Jupiter node) and hunts for LoRA-merge + prompt strategies that achieve a target vocal
effect on **MOSS-VA-v2** (`laion/moss-tts-local-transformer-4.55b-voice-acting-v2`).
An LLM brain (Gemma-4-12B, quantized, served by vLLM on the *same* GPU) drives a
self-coded ReAct tool loop over the TTS engine and a full perceptual scoring stack,
guided by distilled conditioning manuals.

Example missions: *"raise Arousal on this reference voice"*, *"evil ghost voice"*,
*"fear-filled screaming"*, *"maximize AROU+VALN+EXPL+S_STRY while staying genuine"*.

The original experiment plan is in [PLAN.md](PLAN.md).

## Architecture

```
┌──────────────────────────── 1×A100-80GB ─────────────────────────────┐
│  vLLM  google/gemma-4-12B-it-qat-w4a16-ct  (gpu_mem_util 0.22)       │
│  MOSS-VA-v2 bf16 + MOSS-Audio-Tokenizer-v2 (n_vq=12)  [engine.py]    │
│  Scorers: EmoNet-40 experts · VoiceNet embed + 57 reg heads ·        │
│    VoiceCLAP genu/blend MLPs · ECAPA spk-sim · faster-whisper ASR    │
│    · procedural-voice-captions                        [scorers.py]   │
│  Agent loop (CPU): ReAct JSON tools, memory, transcripts [agent.py]  │
└──────────────────────────────────────────────────────────────────────┘
```

| module | role |
|---|---|
| `engine.py` | TTS load, PEFT multi-adapter LoRA merging (per-module `scaling×λ` pattern), batched generation with empty-retry, reference encoding |
| `scorers.py` | 99-vec (EmoNet-40 + VoiceNet-57 + genuineness + burst-blend), WER (Parakeet→faster-whisper fallback, 3 decode variants), procedural captions, ECAPA speaker similarity |
| `tools.py` | the JSON tool API + registry (validation, docs auto-generated into the context) |
| `agent.py` | self-coded ReAct loop: JSON-schema-guided decoding, parse retries, budget, context compression, memory file, JSONL transcript, subagent spawning |
| `llm.py` | vLLM OpenAI-compatible client + server spawner (stdlib only) |
| `build_context.py` | distills the live manuals + evolved genomes into `context/system_context.md` (~38k tokens) |
| `run_agent.py` | CLI entry point |

## LLM brain choice

**`google/gemma-4-12B-it-qat-w4a16-ct`** — Google's official quantization-aware-trained
w4a16 checkpoint in compressed-tensors format, natively supported by vLLM (tested on
vLLM 0.26.0). Chosen over community AWQ/GPTQ quants (QAT ≥ post-hoc quant quality) and
over Gemma-3-12B (newer generation, same ~8GB footprint). Fallback if unavailable:
`google/gemma-3-12b-it` or `RedHatAI/gemma-3-12b-it-FP8-dynamic`.
Served with `gpu_memory_utilization 0.22` and `max_model_len 65536` so the whole stack
fits one 80GB GPU. The agent uses vLLM **structured output** (json_schema) so every
tool call parses.

## GPU memory budget (measured, A100-80GB single-GPU config)

| component | GB |
|---|---|
| vLLM server (Gemma-4-12B w4a16 weights ~8 + 64k KV + CUDA graphs; hard cap 0.22×80) | 17.6 |
| MOSS-VA-v2 bf16 + MOSS-Audio-Tokenizer-v2 (torch alloc after load) | 16.8 |
| scorer stack (BUD-E-Whisper + 40 EmoNet experts, VoiceNet, VoiceCLAP, genu/blend MLPs) | 7.0 |
| faster-whisper `small` (ct2 float16) + ECAPA (loaded lazily) | ~2 |
| generation activations/KV, batch 8 × 384 frames (transient peak) | ~8-12 |
| **total peak** | **~55 GB** (≈25 GB headroom) |

## Reward design (default)

```
fitness = ( Σ w_i·norm(target_i) + w_g·norm(GENU) + w_b·norm(BLEND) ) × (1 − WER),  WER ∈ [0,1]
```

Every reward **multiplies by (1 − WER)** (3-variant ASR vs the prompted text) unless the
mission explicitly targets non-speech (screams etc.) — this is what stops over-acted,
unintelligible deliveries from winning. Targets may be up-weighted (emotions 1.5-2×);
GENU + BLEND together should weigh about as much as the targets (run_generation defaults
`w_g = w_b = Σw_targets / 2`); GENU may be down-weighted for intentionally unnatural
characters but never dropped. `run_generation` implements the formula directly
(`fitness={maximize:{code:w}, genu_weight?, blend_weight?, wer_multiplier?:true}`).

## Tool API

Full auto-generated reference in `context/system_context.md` (section *Tool reference*);
registry in `tools.py`. Summary:

| tool | purpose |
|---|---|
| `list_loras` | catalog: 40 emotion v3 · 114 VoiceNet (`vn_<DIM>_<high\|low>`) · 2×120 character LoRAs (lazy HF download) |
| `merge_loras` | activate a PEFT multi-adapter merge `[{name, scale}]` (`[]` = base model) |
| `generate` | batch-generate n samples (caption `instruction` + spoken `text`, optional reference voice, sampling overrides) |
| `score` | 99-vec readout: GENU/BLEND/QUALITY + any slots by code, means + top emotions/dims |
| `transcribe` | 3-decode-variant ASR + WER vs the prompted text |
| `caption` | procedural voice caption (how the sample actually sounds) |
| `speaker_sim` | ECAPA cosine vs a reference |
| `load_reference` | encode + score + caption a reference wav |
| `compute_baseline` | no-LoRA baseline pool → per-code means for fitness constraints |
| `run_generation` | one evolution generation: merge+generate+score a batch of genomes vs a fitness spec (mean-of-8) |
| `save_best` | hall-of-fame wav + genome + scores |
| `memory` | persistent notes (survive context compression) |
| `spawn_subagent` | fresh-context copy of the agent for a focused subtask |
| `fetch_manual` | re-pull live manuals → rebuild the system context; or fetch one entry |
| `push_results` | commit+push results to this repo |
| `finish` | end the mission with a final report |
| `sidon_enhance`* | speech restoration (Sidon v0.1, 48 kHz out) — clean noisy winners; returns new sample_ids |
| `audio_stretch`* | time-stretch 0.25-4.0x without pitch change (audiostretchy) — pacing experiments |

\* manifest-driven **lazy** tools (`registry/tools.json`, SWARM_PLAN §6): never loaded at
startup — `engine.ToolModelPool` loads them on first call, keeps them warm, auto-unloads
after 300 s idle, and LRU-evicts under a hard VRAM budget. Adding a tool is a data
change: drop a module in `tools_ext/` + one manifest entry; `tools.py` dispatches and
`build_context.py` renders its docs automatically.

## How to run (fresh A100 node, only HF + GH tokens needed)

```bash
git clone https://github.com/LAION-AI/voice-acting-search-agent
cd voice-acting-search-agent
cp .env.example .env        # fill in HF_TOKEN + GITHUB_TOKEN
set -a; source .env; set +a

# agent env: torch 2.6 + transformers 5.x + peft 0.13 + faster-whisper + speechbrain
#            + soundfile + pyyaml (+ tiktoken optional)
# vLLM lives in its OWN env (configs/*.yaml: llm.vllm_python) to avoid torch clashes:
conda create -n vllm-serve python=3.12 -y && conda run -n vllm-serve pip install vllm

# external assets, paths configured in configs/single_gpu.yaml:
#   paths.vb_dataset   reference scorer code (va_rescore.py, spk_sim.py, score layout)
#   paths.emo_loras / vn_loras           LoRA catalogs (HF: laion/moss-voicenet-dimension-loras)
#   paths.pvc_repo     clone of LAION-AI/procedural-voice-captions
#   paths.manual_* / edge_evo_dir / evo_all_dir   manual mirrors + evolved genomes
python build_context.py                  # distill manuals -> context/system_context.md

python run_agent.py --gpu 0 \
  --mission "Make this reference voice express extreme arousal. Reference: ref1" \
  --budget-tool-calls 40
```

`run_agent.py` starts vLLM on the same GPU if none is reachable, loads the stack, and
runs the agent; samples, hall of fame, memory, transcript and report land in
`runs/<mission-slug>-<ts>/`.

### 4-GPU node / SLURM (Jupiter)

`slurm/jupiter.sbatch`: GPU0 serves one vLLM for the node (`gpu_memory_utilization 0.90`),
GPU1-3 each run `run_agent.py --no-start-llm` with a mission from `missions.txt`, all
pointing at the shared server via `configs/node_4gpu.yaml`.

### Offline cache manifest (air-gapped nodes)

Pre-populate `$HF_HOME` with: `laion/moss-tts-local-transformer-4.55b-voice-acting-v2`,
`OpenMOSS-Team/MOSS-Audio-Tokenizer-v2`, `google/gemma-4-12B-it-qat-w4a16-ct`,
`laion/BUD-E-Whisper`, `laion/Empathic-Insight-Voice-Plus`, `laion/voiceclap-commercial`,
the VoiceNet dim-head repo, `speechbrain/spkrec-ecapa-voxceleb`, faster-whisper `small`,
and the LoRA catalogs (`laion/moss-voicenet-dimension-loras`,
`TTS-AGI/moss-character-loras-{genuine,refined}`, emotion v3 LoRAs).

## How the agent self-updates and reports

- **`fetch_manual` (no args)** re-pulls the moss-voiceacting-manual / moss-voicenet-manual
  sources and rebuilds `context/system_context.md`; the loop re-reads it next turn.
- **`push_results`** commits a results directory back to this repo using `GITHUB_TOKEN` —
  the canary transcripts and benchmark results under `results/` landed this way.

## Results

- `results/canary/` — three bring-up canary missions (transcripts + scores, audio-b64 stripped)
- `results/benchmark_arou_valn_expl_stry/` — 10×8 evolution benchmark (RESULTS.md,
  hall-of-fame wavs + genomes, fitness trajectory)

## Known issues (historical)

- **Pre-2026-08-01 artifacts contain doubled audio.** A channel-flatten bug in
  `engine.py` (codec stereo output concatenated by `reshape(-1)` instead of
  channel-averaged) made every generated sample contain its content twice
  back-to-back. All run artifacts committed before the fix (canaries,
  `results/benchmark_arou_valn_expl_stry/`) have 2x durations, ASR transcripts with
  the text twice (WER saturated to ~1.0), and 99-vec scores computed on doubled
  audio (measured shift on the affected cohort: |mean delta| <= 0.06 per code — rankings
  largely unaffected, absolute values slightly off). Guarded by
  `scripts/test_no_doubling.py` since the fix.
