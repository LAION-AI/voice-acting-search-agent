# Dual-supervisor experiment: local MOSS-Audio-8B (ACTIVE) vs Gemini 3.6 (SHADOW)

Setup: supervised mission 'explicit' (fitness = (2·EXPL + GENU + BLEND) × (1−WER)), 16 generations × 8 genomes,
supervisor verdict after EVERY generation. Both supervisors received the IDENTICAL compressed report
(SWARM_PLAN §3.1: agent's own interpretation + cohort stats) plus the top-2 takes as audio.
LOCAL = MOSS-Audio-8B-Thinking (no-think) on GPU 2 — its directives were INJECTED into the agent's next turn.
SHADOW = Gemini (standard API, gemini-3.6-flash, audio inline) — logged only, never injected.
Zero backend errors across 15 dual rounds.

## Score series

| gen | local (active) | gemini (shadow) | best fitness | cohort WER |
|---|---|---|---|---|
| 1 | 6 | 8 | 0.417 | 0.21 |
| 2 | 3 | 4 | 0.125 | 0.9 |
| 3 | 6 | 7 | 0.242 | 0.5 |
| 4 | 6 | 4 | 0.205 | 0.89 |
| 5 | 7 | 5 | 0.166 | 0.62 |
| 6 | 5 | 4 | 0.188 | 0.74 |
| 7 | 7 | 3 | 0.277 | 0.62 |
| 8 | 6 | 5 | 0.155 | 0.79 |
| 9 | 7 | 5 | 0.16 | 0.79 |
| 10 | 6 | 2 | 0.127 | 0.8 |
| 11 | 5 | 3 | 0.149 | 0.87 |
| 12 | 2 | 3 | 0.089 | 0.96 |
| 13 | 2 | 2 | 0.139 | 0.84 |
| 14 | 4 | 2 | 0.142 | 0.82 |
| 15 | 5 | 2 | 0.073 | 0.91 |

**Score correlation:** Pearson r = 0.443 (p=0.098), Spearman ρ = 0.523 (p=0.045) — moderate agreement.

**Who tracked reality?** Correlation with the actual generation fitness: local 0.461 vs **Gemini 0.724**.
Gemini's scores follow the objective decay (WER collapse 0.21→0.9) closely; local MOSS stayed lenient (5-7,
three APPROVEs) while cohorts were becoming unintelligible.

## Verbatim example pairs (same audio, same report)

**gen 1** — local 6/REVISE vs gemini 8/REVISE
> LOCAL: The delivery lacks emotional depth and explicitness; the voice sounds too flat and lacks the necessary breath pressure and dynamic range to convey the scene's atmosphere.
> GEMINI: The delivery is slightly clinical and monotone; adding subtle dynamic emphasis on key adjectives will make the delivery feel more deliberately explicit and impactful rather than purely mechanical.

**gen 7** — local 7/APPROVE vs gemini 3/REVISE
> LOCAL: The audio could benefit from more dynamic breath pressure and a slightly more dramatic intonation to enhance the emotional impact. The room tone is clean, which is good, but the voice could have a bit more presence.
> GEMINI: Take 1 exhibits extreme, unnatural chest bellowing and artificial pitch booming. Take 2 displays hyper-dramatic, erratic pitch swings and rushed pacing that severely compromise naturalness.

**gen 10** — local 6/REVISE vs gemini 2/REDIRECT
> LOCAL: The voice lacks the necessary intensity and explicitness to match the dramatic nature of the text. The delivery is too smooth and lacks the required emotional weight.
> GEMINI: Both takes truncate abruptly after 'jagged mountains', leaving more than half of the text unread and causing severe word error rate penalties.

**gen 13** — local 2/REVISE vs gemini 2/REDIRECT
> LOCAL: The voice lacks the dramatic intensity and emotional weight required for the mission. The delivery is too flat and lacks the necessary breath pressure and vocal presence.
> GEMINI: Heavy gargling, artificial throat constriction, and extreme vocal distortion render the speech slurred and largely unintelligible. Additionally, phrases cut off prematurely or dissolve into mumbled cadence.

## Supervised vs unsupervised control (same mission, same reward, same brain)

| | supervised (local MOSS active) | unsupervised control (singledim_explicit) |
|---|---|---|
| generations | 16 | 9 |
| best fitness trajectory | 0.417 (gen1) → declines → 0.114 (gen16) | 0.357 → **0.625** (gen8, rising) |
| final cohort WER | 0.80–0.96 | 0.06–0.27 |
| outcome | agent chased 'more breath pressure / more dramatic' directives into overacting; the (1−WER) reward kept punishing, fitness collapsed | agent followed the fitness signal alone, found a low-WER 'sultry whisper' recipe with the best fitness of both runs |

## Conclusions for the SWARM

1. **The active local supervisor HURT the search.** MOSS-8B listens, and its prose sounds plausible, but its
   directives are generic and repetitive ('more breath pressure', 'more dramatic intonation', 'room tone') —
   several are verbatim echoes of the example phrases in its own system prompt (prompt contamination). Injected
   every generation, they consistently pushed toward intensity, i.e. AGAINST the (1−WER) reward.
2. **Gemini is the better acoustic judge**: fitness correlation 0.72 vs 0.46, and its critiques are specific and
   diagnostic ('extreme, unnatural chest bellowing', 'phrases dissolve into mumbled cadence') rather than template-like.
3. **Usability verdict**: MOSS-Audio-8B as ACTIVE supervisor is NOT usable as-is for the offline swarm. Options:
   (a) demote local to sanity-scorer (score only, no directives) and let fitness steer;
   (b) fix the prompt-contamination (no example phrases; force it to quote concrete audio moments);
   (c) online swarms should use Gemini as the active supervisor and keep MOSS as the air-gapped fallback.
4. Agreement is high at the extremes (both ~2 when audio is truly broken, gen 12-15) — MOSS works as a
   coarse gate (PASS/FAIL), just not as a fine-grained director.
