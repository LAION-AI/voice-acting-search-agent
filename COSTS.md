# Supervisor & LLM cost estimates (PRELIMINARY)

*Measured 2026-08-02 from the overnight program's actual logs (one night, 8 swarm tasks +
1 supervised mission + 3 model arms). Prices fetched 2026-08-02 from the sources cited
below. Treat every number as a first calibration point, not a budget guarantee.*

## 1. Measured token usage (not guessed)

### 1.1 Supervisor calls (audio-inclusive)

Basis: all real payloads in `runs/swarm_T000*/supervisor_log.jsonl` (S5: **74 dual
rounds** across 8 tasks) and `runs/benchmark_expl/supervisor_log.jsonl` (**15 rounds**),
plus **~3** judge calls (`scripts/judge_arms.py`). Text tokens estimated as chars/4;
audio billed by Gemini at ~32 tokens/s.

| quantity | S5 mean (74 rounds) | supervised-explicit mean (15 rounds) |
|---|---|---|
| report payload | 3,124 chars ≈ 780 tok | 3,978 chars ≈ 995 tok |
| system + mission text | ~530 tok | ~530 tok |
| audio attached (top-2 takes + 1s gap) | 15.2 s ≈ 486 tok | 17.5 s ≈ 560 tok |
| **input total / call** | **≈ 1,800 tok** | ≈ 2,090 tok |
| verdict output (raw JSON) | 746 chars ≈ 187 tok | ~190 tok |
| thinking overhead (billed as output; not visible via Hyprlab response) | est. 2–4× raw → **≈ 400–800 tok** | same |

**Per-call cost @ official `gemini-3.6-flash`** ($1.50/M in, $7.50/M out incl. thinking):
input $0.0027 + output $0.0014–0.006 → **≈ $0.004–0.009 per supervised generation**
(mid ≈ $0.007). Judge calls are bigger (≈9k in + ~48s audio + ~1.5k out ≈ $0.03 each).

**Last night's actual Gemini usage: ~92 calls** (74 S5 + 15 explicit + 3 judge)
≈ **$0.65–0.95 at official prices** — i.e. about **$1 for the whole overnight program's
supervision + judging.**

### 1.2 Agent brain (31B) — measured from real swarm transcripts

Reconstructed exactly as `agent.py` builds each request (38.2k-token system context +
growing history, compression events applied), chars/4:

| run | generations | LLM turns | input tokens | output tokens |
|---|---|---|---|---|
| `runs/swarm_T0004` (hoarse voice) | 8 | 21 | **934k** | 12.3k |
| `runs/swarm_T0007` (bomb-squad) | 5 | 12 | **525k** | 7.0k |
| extrapolated **10-generation** search | 10 | ~25 | **≈ 1.2M** | ≈ 15k |

Key structural fact: **~79% of all input tokens are the identical 38.2k system prompt**
(25 turns × 38.2k ≈ 0.95M of the 1.2M) — highly cacheable wherever a provider offers
prompt caching.

## 2. Prices used (official, fetched 2026-08-02)

| model | input /M | output /M (incl. thinking) | cached input /M | source |
|---|---|---|---|---|
| gemini-3.6-flash | $1.50 | $7.50 | $0.15 (+$1.00/M/h storage) | [ai.google.dev/gemini-api/docs/pricing](https://ai.google.dev/gemini-api/docs/pricing) |
| gemini-3.5-flash-lite | $0.30 | $2.50 | $0.03 | same |
| gemini-3.1-pro (≤200k) | $2.00 | $12.00 | $0.20 | same |
| DeepSeek V4-Flash | $0.14 (miss) | $0.28 | **$0.0028** (hit) | [deepseek.ai/pricing](https://deepseek.ai/pricing), verified late Jul 2026 |
| Groq Kimi K2 | $1.00 | $3.00 | $0.50 | [cloudzero.com/blog/groq-pricing](https://www.cloudzero.com/blog/groq-pricing/), Aug 2026 |

Notes: the Gemini page does not itemize audio separately for 3.6-flash — audio is
tokenized (~32 tok/s) and billed at the input rate; thinking tokens bill as output.
**Hyprlab** publishes no per-token price list — it is a prepaid-balance model with
per-request "millicent" usage tracking ([hyprlab.io](https://hyprlab.io), docs.hyprlab.io
checked 2026-08-02); historically at-or-below official provider rates. Our ~92-call
night therefore cost **≈ $1 or less** of Hyprlab balance (official-price upper bound).

## 3. Supervisor cost projections (user ask: 100 / 1000 tasks × 5 / 10 rounds)

Per-call basis from §1.1 (mid $0.007; range $0.004–0.009 at 3.6-flash; $0.011 at 3.1-pro):

| scale | calls | gemini-3.6-flash (range) | gemini-3.1-pro | Hyprlab (est.) |
|---|---|---|---|---|
| 100 tasks × 5 gens | 500 | **$3.50** ($2.0–4.5) | $5.50 | ≲ $3.50 |
| 100 tasks × 10 gens | 1,000 | **$7** ($4–9) | $11 | ≲ $7 |
| 1,000 tasks × 5 gens | 5,000 | **$35** ($20–45) | $55 | ≲ $35 |
| 1,000 tasks × 10 gens | 10,000 | **$70** ($40–90) | $110 | ≲ $70 |

Even the 1,000-task, 10-generation swarm buys its entire audio-listening supervision for
well under the cost of a single GPU-day. Supervision cost is **not** a scaling blocker.

## 4. Agent brain via API instead of local 31B?

Measured 10-generation task (§1.2): **≈1.2M input (0.95M cacheable) + 15k output.**

| provider/model | no caching | with prompt caching | 1,000 tasks ×10 gens (cached) |
|---|---|---|---|
| DeepSeek V4-Flash | $0.17 | **$0.042** | **$42** |
| gemini-3.5-flash-lite | $0.40 | **$0.14** | $140 |
| Groq Kimi K2 | $1.25 | $0.77 | $770 |
| gemini-3.6-flash | $1.91 | $0.63 | $630 |

(cached = 0.95M at the cache-hit rate + 0.25M fresh + output.)

**Conclusion (preliminary):** yes — a cheap API brain beats local 31B **on cost** at any
realistic utilization: a 10-gen search costs $0.04–0.63 via API vs a dedicated ~40GB
vLLM slice of an A100 for ~1–2h (≈$1.5–4 at common rental rates), and caching makes our
38k system prompt nearly free on DeepSeek/Gemini. Two caveats before switching: (a)
**quality is unvalidated** — the 31B won against *local* 12B/MoE, no API model has been
benchmarked as the brain yet (worth one arm-run: `scripts/run_arm.sh` against any
OpenAI-compatible endpoint); (b) offline/air-gapped deployments (Jupiter) still need the
local brain. The local 31B remains the recommended default until an API arm is judged.

## Measured: gpt-5.6-luna API brain — 4 single-dim missions (2026-08-02)

Actual token usage captured from `runs/singledimluna_*/llm_usage.json` (Hyprlab `gpt-5.6-luna`,
$0.10/M input, $0.60/M output, ≤272k tier):

| mission | calls | input tok | output tok | $ |
|---|---|---|---|---|
| arousal | 13 | 587,284 | 10,251 | $0.065 |
| valence | 17 | 774,000 | 12,295 | $0.085 |
| explicit | 17 | 773,536 | 11,736 | $0.084 |
| storyteller | 15 | 693,273 | 10,828 | $0.076 |
| **total (4 missions)** | **62** | **2,828,093** | **45,110** | **≈ $0.31** |

≈ **$0.077 per 8-generation mission** with NO local LLM GPU. In the 4-arm Gemini judge
([MODEL_COMPARISON.md](MODEL_COMPARISON.md)) luna scored **9 / 9 / 10** — statistically tied with the
local 31B (9/10/9) and clearly ahead of 12B/MoE — while costing cents and freeing the GPU entirely for
the TTS engine. It independently used both the stabilizer adapters AND the aesthetics (ESTH) term
(e.g. storytelling `vn_S_STRY_high@0.65 + vn_ARSH_high@0.20 + vn_BRGT_low@0.20` → S_STRY 0.998, ESTH 0.590,
WER 0.000, 10.96 s). **Preliminary conclusion: a cheap fast API brain is the recommended default when
online; keep local 31B for air-gapped Jupiter runs.**
