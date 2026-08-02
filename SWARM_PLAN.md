# SWARM: Scaling the Voice-Acting Search Agent into a Self-Improving Hive Mind

**Status: technical plan (v1, Aug 2026) — extension of the single-agent system in this repo.**

This document specifies how to run **32–64 concurrent agent instances** on different voice-acting
search tasks so that (a) every instance benefits from what every earlier instance learned, (b) every
run is supervised, validated and steerable by a stronger model, (c) every run is crash-resumable, and
(d) the shared knowledge base — the *manual* — grows and improves over time like a hive mind.

The single-agent core (engine / scorers / tools / agent loop / vLLM brain) is unchanged. SWARM adds
four layers around it:

```
                        ┌────────────────────────────────────────────┐
                        │                ORCHESTRATOR                │
                        │  (API frontier model or local Gemma-4-27B) │
                        │  seeds tasks · reclaims stale · triggers   │
                        │  consolidation · scales the swarm          │
                        └───────┬────────────────────────────┬───────┘
                                │ reads/writes               │ runs offline
                                ▼                            ▼
   ┌─────────────────── GITHUB REPO (this repo) ──────────────────────┐
   │  registry/   tasks.jsonl · claims/ · tools.json · loras.json     │
   │  experience/ <task_id>/ TASK.md · learnings.md · reports/ (TIER-2 raw,│
   │              NOT auto-loaded into agent context)                 │
   │  manual/approved/  objectives/<slug>.md  (TIER-1 curated,        │
   │              THE ONLY knowledge auto-loaded into agents)         │
   └──────┬───────────────────────────────────────────────▲──────────┘
          │ sparse pull every K generations               │ push learnings,
          ▼                                               │ reports, status
   ┌─────────────────────── WORKER SWARM ─────────────────┴──────────┐
   │  N nodes × (GPU0: shared vLLM · GPU1-3: one agent worker each)  │
   │  each worker: claim task → evolve → report → sync → repeat      │
   └──────┬──────────────────────────────────────────────────────────┘
          │ every R generations: compressed report            ▲
          ▼                                                   │ verdict 0-10,
   ┌──────────────────────── SUPERVISOR ─────────────────────┴──────┐
   │  Gemini 3.6 (Hyprlab/standard API, thinking) — or a local      │
   │  audio-understanding model (MOSS-Audio-8B-Thinking) that can   │
   │  actually LISTEN to the top samples                            │
   └────────────────────────────────────────────────────────────────┘
   Artifacts (mp3/tar, too big for git) → HF dataset repo, keyed by task_id
```

---

## 1. Two-tier knowledge: `experience/` vs `manual/approved/`

The core idea: **raw experience is written constantly; curated knowledge is promoted deliberately.**
Agents write everything they learn, but they only *read* what has been vetted — otherwise 64 agents
would drown each other in noisy, contradictory, unverified notes.

| | TIER 2 — `experience/` | TIER 1 — `manual/approved/` |
|---|---|---|
| written by | every worker, every few generations | consolidation job only |
| read by | orchestrator, consolidation job, humans | **every agent's system prompt** (via `build_context.py`) |
| format | free-form `learnings.md` + structured `reports/*.json` | standardized objective pages (schema below) |
| quality bar | none — negative results are explicitly wanted | frontier-model reviewed, statistically grounded |
| loaded into agent context | **never** automatically | always (current approved set) |

### 1.1 Repo layout

```
registry/
  tasks.jsonl                # append-only task registry (one JSON per line)
  claims/<task_id>.json      # one file per claim → no merge conflicts
  tools.json                 # tool manifest (auto-discovery, §6)
  loras.json                 # LoRA catalog manifest (auto-discovery, §6)
experience/
  T0042_angry_wife_scene/
    TASK.md                  # objective, status, owner, heartbeat, budget
    learnings.md             # free-form: what worked / what did NOT / dead ends
    reports/gen_005.json     # standardized supervisor reports (schema §3)
    reports/gen_010.json
    artifacts.json           # pointers into the HF artifact repo
manual/
  approved/
    _index.json              # machine index: slug → objective tags, updated_at
    objectives/anger-shouting-scene.md
    objectives/evil-ghost-voice.md
    dimensions/…             # the existing VoiceNet/emotion manual content
  candidates/                # consolidation output awaiting sign-off (§5)
```

### 1.2 Approved-page schema (one page per objective family)

```markdown
# <Objective family, e.g. "Furious shouting at another person (scene)">
objective-tags: anger, shouting, scene, dyadic
last-consolidated: 2026-08-03  source-tasks: T0042, T0057, T0061

## Best known recipes  (ranked; each with evidence)
1. LoRAs: Anger@1.7 + vn_VOLT_high@0.6 · prompt: "…" · sampling: t1.3
   evidence: T0042 fitness 2.31, supervisor 9/10, n=24 samples
## What does NOT work            ← negative knowledge is first-class
- vn_ARSH_high ≥1.0 → BLEND −0.44, supervisor rejected 3/10 (T0042 gen4)
## Operationalization            ← how to MEASURE this objective
- maximize: Anger, VOLT; constraints: GENU ≥ baseline, spk_sim ≥ 0.5
## Open questions
- untested: interaction with character LoRAs
```

Agents get the pages whose `objective-tags` overlap their task tags (plus the always-on core manual);
`build_context.py` already assembles the context — it gains a *tag-filtered manual loader*.

---

## 2. Task registry & lifecycle (crash-resumable, conflict-free)

### 2.1 Lifecycle

```
            ┌─────────┐  claim (atomic file add)   ┌─────────┐
  seed ───► │  OPEN   │ ─────────────────────────► │ CLAIMED │
            └─────────┘                            └────┬────┘
                 ▲                                      │ first report pushed
                 │ heartbeat stale > 30 min             ▼
                 │ (orchestrator reopens)          ┌─────────┐   every R gens:
                 └──────────────────────────────── │ RUNNING │ ◄─ report + push
                                                   └────┬────┘
                          supervisor ≥ 9/10             │ budget exhausted OR
                          OR budget out                 ▼ supervisor approves
            ┌─────────┐  final verdict < 7:       ┌─────────┐
            │REOPENED │ ◄──────────────────────── │ REVIEW  │
            └─────────┘  reopen w/ directives     └────┬────┘
                                                       ▼ verdict ≥ 7
                                                  ┌─────────┐
                                                  │ CLOSED  │ → consolidation queue
                                                  └─────────┘
```

- **Claiming without merge conflicts**: a worker claims `T0042` by *adding* the file
  `registry/claims/T0042.json` (`{task_id, agent_id, host, claimed_at}`) and pushing. Git add of a
  new unique path cannot conflict; if the push is rejected (someone else claimed first), pull-rebase
  → claim file exists → pick the next OPEN task. This is the whole locking protocol.
- **Heartbeat**: the worker touches `TASK.md` (`heartbeat:` line) at every sync. Orchestrator reopens
  tasks whose heartbeat is older than 30 min (worker killed) — the next claimer **resumes**: it reads
  `TASK.md`, past `reports/`, `learnings.md` and the artifact index, and continues from the last
  generation instead of starting over.
- **Everything needed to resume is pushed continuously** (small: markdown + JSON + evolution log;
  audio goes to HF, §4). A killed worker loses at most R generations of work.

### 2.2 `TASK.md` header (machine-parsable front matter)

```yaml
task_id: T0042
title: Woman furiously screams at her spouse (rage, hate) — scene
tags: [anger, shouting, scene, female]
status: RUNNING          # OPEN|CLAIMED|RUNNING|REVIEW|CLOSED|REOPENED
owner: worker-jupiter-n03-g2
heartbeat: 2026-08-03T14:22:05Z
budget: {generations: 30, supervisor_rounds: 6}
operationalization:      # proposed by seeder, may be REVISED by supervisor
  maximize: [Anger, VOLT]
  constraints: [{code: GENU, min: baseline}, {code: BLEND, min: baseline}]
resume_hint: gen 12 done; best fitness 1.94; see reports/gen_010.json
```

---

## 3. Supervisor loop (validation + steering)

Every **R generations** (default R=5; configurable per task) the worker compiles a **compressed,
standardized report** — never raw sample dumps — and sends it to the supervisor:

### 3.1 Report schema (`reports/gen_NNN.json`, also rendered as md)

```json
{
  "task_id": "T0042", "generation": 10, "tool_calls_used": 122,
  "objective": "…verbatim task…",
  "operationalization": {"maximize": ["Anger","VOLT"], "constraints": ["GENU>=baseline"]},
  "proposed_solution":  {"recipe": {"loras": [...], "prompt": "…", "sampling": {...}},
                         "scores": {"Anger": 2.31, "GENU": 0.31, "spk_sim": 0.55},
                         "artifact": "hf://…/T0042/gen10_best.mp3"},
  "alternative_solution": { "...same shape, 2nd-best distinct strategy..." },
  "dead_ends": [
    {"strategy": "vn_ARSH_high@1.2", "why_abandoned": "BLEND -0.44, Anger only +0.1",
     "stats": {"fitness_mean": 0.42, "n": 16}}
  ],
  "fitness_trajectory": [0.8, 1.1, 1.4, 1.61, 1.94],
  "worker_assessment": "believe intensity ceiling reached; unsure about operationalization",
  "questions_for_supervisor": ["should VOLT stay in the maximize set?"]
}
```

### 3.2 Supervisor contract

The supervisor receives: the report, the **full tool-API documentation** (same as the worker's, so
its advice is *actionable*), and the relevant approved manual pages — with a supervisor-specific
system prompt. It must answer in a fixed schema:

```json
{"score_0_10": 6,
 "verdict": "REVISE",                    // APPROVE | REVISE | REDIRECT
 "agrees_with_proposed": false,
 "best_strategy_seen": "alternative_solution",
 "directives": [
   "Drop VOLT from maximize; it is double-counting loudness already implied by Anger.",
   "Try Anger@1.7 + vn_ATCK_high@0.5 with a contained-then-erupting two-phase script cue.",
   "Reward: add 0.5*z(BLEND) — the screams sound pasted-on (BLEND 0.19)."
 ],
 "operationalization_change": {"maximize": ["Anger"], "constraints": ["GENU>=baseline","BLEND>=0.3"]},
 "close_task": false}
```

Directives are injected verbatim into the worker's next planning turn. `operationalization_change`
**overrides** the task's reward definition — the supervisor may re-operationalize a fuzzy objective.
Termination: `score_0_10 ≥ 9` (or ≥ 7 at budget end) → REVIEW→CLOSED; else REOPENED with directives.

### 3.3 Supervisor backends (pluggable, `configs/supervisor.yaml`)

| backend | reach | notes | **empirical verdict (overnight 2026-08-02, 89 dual rounds)** |
|---|---|---|---|
| `gemini-hyprlab` | text-only report | `base_url https://api.hyprlab.io`, model `gemini-3.6-flash`; **known gotchas:** needs `max_output_tokens ≥ 300`; `thinking_config` → 400 on Hyprlab (omit; use `-pro`/thinking on the standard API instead) | works; audio also accepted in practice — used as the ACTIVE director in S5, ~$0.007/call (COSTS.md) |
| `gemini-standard` | text + **audio** | Gemini 3.6 w/ thinking; can receive the top mp3s inline → true listening judgment | **best judge**: fitness-tracking r=0.72 vs local 0.46; critiques specific and diagnostic — RECOMMENDED active supervisor |
| `local-audio-lm` | text + **audio**, fully offline | MOSS-Audio-8B-Thinking (or similar audio-understanding LM) on a swarm GPU — required for air-gapped Jupiter runs | **do NOT let it direct**: as active supervisor it collapsed a search (0.417→0.114 vs control 0.625, prompt-contaminated generic directives). Usable as decontaminated score-only gate (reliable at extremes; S5 r=0.486, lenient mean 5.73 vs 3.66) |

The worker attaches the top-2/3 mp3 artifacts whenever the backend supports audio input; scoring by
*listening* beats scoring by numbers alone and catches scorer-gaming (high EmoNet score, absurd audio).

---

## 4. Artifacts: HF dataset repo, keyed by task

Git holds text; audio goes to a Hugging Face dataset repo (e.g. `TTS-AGI/voice-acting-swarm-artifacts`,
private):

```
tasks/T0042/gen_010/best.mp3 · alt.mp3 · hall_of_fame.tar.gz
tasks/T0042/final/top3/*.mp3 + genomes.json
```

`experience/<task>/artifacts.json` maps logical names → HF paths (+ sha256). Uploads happen at every
report sync (idempotent, `upload_file` with fixed paths). The orchestrator (and any resuming worker)
can therefore always reconstruct: *what was tried, what it sounded like, what was learned* — even for
a task killed mid-flight.

---

## 5. Consolidation: how the hive actually gets smarter

A periodic **offline job** (frontier model — Claude/GPT/Gemini via API, or a large local model on
Jupiter) run by the orchestrator, e.g. nightly or every M closed tasks:

```
experience/*/learnings.md + reports/*.json  (CLOSED tasks since last run)
        │  1. cluster by objective-tags (taxonomy: emotion × delivery ×
        │     scene-type × voice-type — extend the existing acting-edge-cases
        │     + EmoNet + VoiceNet taxonomies)
        │  2. per cluster: distill into/merge with the approved-page schema —
        │     keep only claims with supervisor-validated evidence; keep
        │     NEGATIVE results; update "best known recipes" ranking
        │  3. write manual/candidates/<slug>.md + a diff summary
        ▼
manual/candidates/  ──(review gate: frontier-model self-check or human PR
                      approval; small diffs may auto-merge)──►  manual/approved/
```

Rules: a candidate page **must** cite source tasks + n + scores for every claim; contradictions
between tasks are surfaced as "disputed" rather than silently overwritten; approved pages are the
**only** thing workers load — so a bad experience entry can never poison the swarm directly.

**Sync cadence (worker side):** at every report boundary the worker runs
`git pull --rebase` (sparse checkout of `manual/approved/ registry/`) and rebuilds its system
context if `manual/approved/_index.json` changed. New knowledge propagates to the whole swarm within
minutes of approval — the "woman screams at spouse" learnings are in context when the
"man screams at boss" task starts.

---

## 6. Auto-discovery of new tools and new LoRAs

Both are **manifest-driven** — adding a capability is a data change, not a code change:

- `registry/tools.json` — one entry per tool:
  ```json
  {"name": "sidon_enhance", "module": "tools_ext.sidon", "vram_gb": 1.2,
   "lifecycle": "lazy", "ttl_s": 300,
   "doc": "Restore/denoise a sample (SIDON v0.1, 48kHz). Args: {sample_ids}. Returns new sample_ids.",
   "requires": ["sarulab-speech/sidon-v0.1"]}
  ```
  `build_context.py` renders every manifest entry into the tool section of the system prompt;
  `tools.py` dispatches by manifest. A tool marked `lazy` is loaded on first call and unloaded after
  `ttl_s` seconds idle (LRU pool with a VRAM budget — see `engine.ToolModelPool`). **The first two
  lazy tools are `sidon_enhance` (sarulab-speech/Sidon) and `audio_stretch`
  (twardoch/audiostretchy: time-stretch 0.25–4.0× without pitch change).**
- `registry/loras.json` — regenerated by a small script from the HF collections (emotion v3,
  VoiceNet dims, characters genuine/refined + anything new): name, family, best-dose, one-line
  effect summary (pulled from the manuals). Workers refresh it at sync time; a newly uploaded LoRA
  repo becomes visible to the whole swarm at the next sync without touching agent code.

---

## 7. Orchestrator

A single long-running process (or cron) — implementation-agnostic brain (API frontier model, or
local Gemma-4-27B on a spare GPU; the orchestrator is 95% bookkeeping, 5% judgment):

1. **Seed**: turn a human goal list (or a generator prompt) into `registry/tasks.jsonl` entries with
   tags + operationalization proposals.
2. **Monitor**: poll registry + heartbeats; REOPEN stale tasks; requeue REOPENED with directives.
3. **Consolidate**: trigger §5 and manage the candidates→approved gate.
4. **Scale**: decide how many workers per node; on Jupiter submit `slurm/jupiter.sbatch` array jobs.
5. **Report**: maintain a swarm dashboard page (tasks by status, fitness distributions, manual
   growth over time) — pushed to the repo's GitHub Pages `docs/`.

**Jupiter node layout** (4×GH200/A100 per node): GPU0 runs one vLLM server; GPU1-3 each run a worker
(TTS + scorers + lazy tools); all three workers call the shared LLM over localhost — the LLM GPU is
the bottleneck resource and this keeps it saturated instead of idle per-worker. One node ≈ 3 workers;
a 12-node allocation ≈ 36 concurrent tasks.

---

## 8. Implementation roadmap

| step | deliverable | est. effort |
|---|---|---|
| S1 | `registry/` + `experience/` scaffolding, TASK.md schema, claim/heartbeat lib (`swarm/registry.py`) | S — registry/tasks + claims + experience/ exports **used in production** (heartbeat lib pending) |
| S2 | report generator in the agent loop (compressed schema §3.1) + `swarm/supervisor.py` with the 3 backends + verdict injection | M — **VALIDATED IN PRODUCTION** (89 dual rounds, S5) |
| S3 | worker daemon `run_worker.py`: claim→(resume?)→evolve→report→sync loop, kill-safe | M |
| S4 | HF artifact uploader + `artifacts.json` (idempotent) | S — **VALIDATED IN PRODUCTION** (8 tasks → TTS-AGI/voice-acting-swarm-artifacts) |
| S5 | tag-filtered manual loader in `build_context.py` + sparse-checkout sync | S |
| S6 | manifest-driven tools (`registry/tools.json`, `ToolModelPool` lazy/TTL) **+ sidon_enhance + audio_stretch** | M — **VALIDATED IN PRODUCTION** (incl. sidon fidelity guard) |
| S7 | `swarm/consolidate.py` (frontier-model distillation, candidates gate) | M |
| S8 | orchestrator v1 (seed/monitor/reopen/dashboard) + 4-GPU single-node swarm test (3 workers) | M |
| S9 | Jupiter: sbatch array, offline model cache manifest, air-gapped supervisor (local audio LM) | M |
| S10 | dry-run: 8 workers × 24 h on mixed task list; measure manual growth + cross-task transfer (does task B start above task A's baseline?) | — |

**Success metric for the whole system**: *transfer*. When a new task shares tags with a closed one,
its generation-1 fitness should already be significantly above the cold-start baseline — that is the
hive mind working. Track it on the dashboard from day one.

---

## 9. Failure modes & mitigations

- **Scorer gaming** (high score, garbage audio): audio-capable supervisor listens to top samples;
  BLEND/GENU constraints in every operationalization by default.
- **Manual poisoning**: only consolidation writes approved pages; every claim carries evidence;
  disputed claims flagged, not merged.
- **Registry races**: unique-file claims (§2.1); append-only tasks.jsonl; per-task directories.
- **Token/API outages**: supervisor optional per-round — worker continues autonomously and queues
  reports; local supervisor backend for air-gapped runs.
- **VRAM creep from lazy tools**: ToolModelPool hard budget (e.g. 6 GB) + LRU eviction.
- **Stale context**: system context rebuilt only on `_index.json` change (cheap check every sync).
