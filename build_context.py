#!/usr/bin/env python3
"""Assemble the distilled system context (context/system_context.md) from the live
manuals + evolved genomes.  Re-runnable: the agent's fetch_manual tool (no args)
re-pulls the sources and calls build() again.

Sections:
  1. Role + score system (99-vec layout, codes)
  2. Tool reference (generated from tools.py)
  3. Prompting best practices + lambda caps + sampling temps
  4. Evolution protocol (default 10 gens x 8 genomes, mean-of-8)
  5. Per-emotion conditioning manual (distilled tips + condition table + evolved genome)
  6. Per-VoiceNet-dim dose-response manual (57 dims, high/low)
  7. Edge-case recipes with evolved genomes
  8. Character LoRA catalog + reference voices
"""
import argparse
import glob
import json
import os
import re
import sys
import urllib.request

import yaml


def _strip_html(s):
    return re.sub(r"<[^>]+>", "", s or "")


def _tok(s):
    """Rough token count (chars/4); tiktoken if available."""
    try:
        import tiktoken
        return len(tiktoken.get_encoding("cl100k_base").encode(s))
    except Exception:
        return len(s) // 4


# ------------------------------------------------------------------ sections
def sec_header(cfg):
    vn_names = json.load(open(cfg["paths"]["vn_dim_names"]))
    lay = json.load(open(cfg["paths"]["score_layout"]))
    emo = [s["code"] for s in lay["slots"] if s["block"] == "emotion"]
    return f"""# Voice-Acting Search Agent — System Context

You are an autonomous voice-acting search agent controlling MOSS-VA-v2, a
voice-acting TTS model, through tools. Your job: find LoRA-merge + prompt
strategies that achieve a target vocal effect, verified by a scoring stack.

## Score system (the 99-vector)
Every generated sample can be scored on 99 normalized slots:
- slots 0-39: EmoNet emotion strengths, z-scored (typical range -1..4; >0.5 = clearly audible,
  >2 = strong). Codes: {", ".join(emo)}.
- slots 40-96: 57 VoiceNet dimensions, minmax-normalized to [0,1] (0.5 = average voice).
  Codes and names: {", ".join(f"{k}={v}" for k, v in vn_names.items())}.
- slot 97 GENU: genuineness [0,1] — how authentically felt (vs performed) the voice sounds.
- slot 98 BLEND: vocal-burst blend [0,1] — how naturally non-speech bursts (laughs, sobs,
  screams) blend with speech.
QUALITY = mean of RCQL (recording quality) + ESTH (esthetics), both [0,1].
Use `score` with metrics=[codes] for targeted readouts; WER from `transcribe` measures
intelligibility of the prompted text (0 good, 1 = unintelligible/replaced by bursts).
"""


def sec_prompting():
    return """## Prompting best practices (measured, follow unless experimenting)
- Conditioning format: `instruction` = voice-acting caption:
  `GENERAL: <voice description>\\nSCRIPT:\\n(<delivery cue>)` ; `text` = the words spoken.
  The (delivery cue) leads the script; vocal-burst tags like `<terrified scream>`,
  `<sobbing>`, `<laughing>` can be appended inside the cue to elicit real bursts.
- Carrier text: >= 20 words, content that MOTIVATES the target delivery (a scream needs
  something to scream about). Short texts (<8 words) let bursts dominate but hurt WER.
- Sampling: WITH a reference voice: temp 0.8-0.9, top_p 0.9, top_k 40-50.
  WITHOUT reference: temp 1.0, top_p 0.95, top_k 25.
- LoRA scale caps: emotion LoRAs effective 0.5-1.9 (sweet spot often 0.75-1.5; >1.9 breaks
  speech). VoiceNet __high adapters: monotonic and strong, best 0.75-1.25. VoiceNet __low
  adapters: weak, need 1.25. Character LoRAs: 1.0.
- Universal cost of VN LoRAs: burst-blend drops ~0.4 at full dose; genuineness is robust.
- Known axis: Triumph/Anger movement trades against Contemplation (and vice versa).
- Emotion LoRAs typically RAISE target emotion strength but COST blend/quality/WER;
  prompt-only steering (BASE_P) is often the better reward when audible strength is
  not required. Combine: moderate LoRA (0.75-1.25) + evolved prompt is the usual winner.
- Raising several targets at once: prefer one strong driver LoRA + small helpers, not
  many at full scale; merged scales add up in effect and quickly destroy intelligibility.
- Duration sanity: the codec runs at 12.5 frames/s, so a sample can never exceed
  max_frames/12.5 seconds; ~3 words/s is normal speech. If transcribe shows the text
  spoken twice, treat it as a bug and report it, not as a sampling quirk.
"""


def sec_reward():
    return """## Reward design (HARD DEFAULT — deviate only when the mission says so)
DEFAULT fitness per sample:
    ( sum_i w_i*norm(target_i) + w_g*norm(GENU) + w_b*norm(BLEND) ) * (1 - WER),  WER clamped to [0,1]
- Every reward MUST multiply by (1 - WER): unintelligible delivery is worthless, and the
  multiplier is what stops over-acted, smeared speech from winning. Only missions that
  explicitly target non-speech vocalization (screams, sobs, laughter bursts) may set
  wer_multiplier=false.
- Weights: mission targets may be up-weighted (emotions typically 1.5-2x). GENU + BLEND
  together should normally weigh about as much as the targets combined (the run_generation
  default: w_g = w_b = half the summed target weights). GENU may be down-weighted for
  intentionally horrible/unnatural characters — but must never be absent by default.
- run_generation implements this formula directly; per-sample WER comes from the 3-variant
  ASR. Watch the per-genome WER mean in results: >0.4 means the delivery is eating the words.
- AESTHETICS you can (and often should) optimize: ESTH = VoiceNet Aesthetics ("nice to
  listen to") and ENJOY = EIV-Plus content-enjoyment head (0-1). Both are first-class
  fitness codes (e.g. maximize {AROU:1.5, ESTH:0.5} = "high arousal but pleasant").
- SCENE DURATION: scene/edge-case missions target ~10-second takes — set max_frames~130
  and fitness duration_target_s:[8,12] (soft multiplicative preference; keep speech
  intelligible unless the mission is explicitly non-speech).
"""


def sec_learnings():
    return """## Validated learnings (overnight benchmark + 8-task swarm, 2026-08-02 — grounded, cite runs)
Single-dimension winner recipes (v2 reward, 31B-arm unless noted; use as STARTING genomes):
- AROU: vn_DARC_high@1.0 -> +0.65 over baseline (runs/singledim27_arousal)
- VALN: vn_VALN_high@1.25 + Elation@0.5 (runs/singledim27_valence)
- EXPL: vn_EXPL_high@0.98 + vn_R_ORAL_high@0.75 with a sultry-whisper delivery — ~4.6x baseline
  (runs/singledim_explicit; prompt framing carries much of the effect)
- S_STRY: vn_S_STRY_high@0.75 — 0.75 BEATS 1.0 because WER stays low (runs/singledim27_storyteller)
Stabilizer adapters (31B discovery, MODEL_COMPARISON.md): adding vn_ARSH_high or vn_BRGT_high at
moderate scale PROTECTS intelligibility when pushing energy/intensity — try them before lowering doses.
Reward: the (1-WER) multiplier WORKS — it reliably prevents over-acted unintelligible winners; keep it
on for all speech missions (evidence: supervised-explicit collapse vs unsupervised control 0.625).
Task-difficulty map (8-task swarm, runs/swarm_T000*):
- SOLVABLE well: hoarse voice (T0004), calm-narration-under-tension/bomb-squad (T0007),
  fading/death voice (T0008).
- ENGINE-LIMITED — do NOT burn budget, state the limitation early and deliver the best partial:
  tongue-clicks mid-speech (T0002), genuine crying mid-sentence (T0003), effort-speech while
  climbing (T0005), jump-scare bursts (T0006: engine truncation on burst-heavy prompts).
Honest caveats: 12B brain searches weakly (repeats itself; 42 calls vs 31B's 22 for the same missions);
quasi-periodic takes (fading voices) can trip self-repetition checks — verify before discarding.
"""


def sec_evolution(cfg):
    ev = cfg["evolution"]
    return f"""## Evolution protocol (default search procedure)
Genome = {{loras:[{{name,scale}}], desc, cue, text, temp, top_p, top_k, reference_id?}}.
Fitness = mean over {ev['samples_per_genome']} samples (mean-of-{ev['samples_per_genome']}) of
(weighted mean of maximized slots) - penalty * sum(constraint shortfalls).
Default run: {ev['generations']} generations x {ev['population']} genomes.
Per generation: keep top-{ev['keep_top']}, mutate {ev['mutate']} (perturb scales +-0.25, reword
desc/cue, swap one LoRA, tweak temp +-0.1, occasionally change text), inject {ev['fresh']} fresh
genomes from manual knowledge. Use `compute_baseline` FIRST when constraints reference the
baseline, then one `run_generation` call per generation; `save_best` the hall of fame;
record per-generation best/mean fitness in `memory`.
Seed generation 0 from: the distilled manual's best per-target strategies, evolved genomes
below, and 1-2 wildcards. Batch everything: run_generation does merge+generate+score for
all genomes in one tool call.
"""


def sec_emotions(cfg):
    tips = json.load(open(cfg["paths"]["manual_emo_tips"]))
    entries = json.load(open(cfg["paths"]["manual_emo_entries"]))
    evo = {}
    for f in glob.glob(os.path.join(cfg["paths"]["evo_all_dir"], "*.json")):
        try:
            d = json.load(open(f))
            evo[d["emotion"]] = d
        except Exception:
            pass
    out = ["## Emotion conditioning manual (per emotion: distilled tip, measured conditions, evolved genome)",
           "Conditions: BASE=neutral prompt no LoRA; BASE_P=evolved steering prompt no LoRA; "
           "LoRA50/100/150=LoRA at 0.5/1.0/1.5 with steering prompt. "
           "reward balances emotion strength, genuineness, blend, quality, WER."]
    for emo in sorted(entries):
        e = entries[emo]
        lines = [f"### {emo}"]
        tip = _strip_html(tips.get(emo, ""))
        if tip:
            lines.append(tip)
        conds = e.get("conditions", {})
        rows = []
        for c in ("BASE", "BASE_P", "LoRA50", "LoRA100", "LoRA150"):
            if c in conds:
                v = conds[c]
                rows.append(f"{c}: rew={v['reward'][0]:.2f} emo={v['emo'][0]:.2f} "
                            f"genu={v['genu'][0]:.2f} blend={v['blend'][0]:.2f} "
                            f"qual={v['qual'][0]:.2f} wer={v['wer'][0]:.2f}")
        lines.append(" | ".join(rows))
        ev = evo.get(emo) or evo.get(emo.replace("_&_", "_and_"))
        if ev and ev.get("hof"):
            g = ev["hof"][0].get("genome", {})
            if g:
                lines.append(
                    f"Evolved genome (fit={ev['hof'][0]['fit']:.2f}, emo={ev['hof'][0]['mean_emo']:.2f}): "
                    f"lam={g.get('lam')} temp={g.get('temp')} top_p={g.get('top_p')} top_k={g.get('top_k')} "
                    f"desc=\"{g.get('desc')}\" cue=\"{g.get('cue')}\"")
        out.append("\n".join(lines))
    return "\n\n".join(out)


def sec_voicenet(cfg):
    data = json.load(open(cfg["paths"]["manual_vn_entries"]))
    entries = data["entries"]
    out = ["## VoiceNet dimension-LoRA manual (57 dims x high/low; dose-response 0.25-1.25)",
           "Each line: best dose, target shift at best dose, side-effects (genu/blend/quality "
           "shifts), top correlated shifts. LoRA names: vn_<DIM>_high / vn_<DIM>_low. "
           "KEY LEARNINGS: __high adapters are monotonic and strong; __low adapters are weak "
           "(use 1.25). Universal cost: burst-blend ~-0.4 at full dose; genuineness robust."]
    for code in sorted(entries):
        e = entries[code]
        lines = [f"### {code} — {e['name']}"]
        for side in ("high", "low"):
            s = e.get(side)
            if not s:
                continue
            fmt_corr = lambda lst: ", ".join(f"{n}({v:+.2f})" for n, v in lst[:3])
            lines.append(
                f"[{side}] best@{s['best_dose']}x Δtarget={s['best_target_shift']:+.2f} "
                f"Δgenu={s['best_genu_shift']:+.2f} Δblend={s['best_blend_shift']:+.2f} "
                f"Δqual={s['best_quality_shift']:+.2f} | VN+ {fmt_corr(s['vn_pos'])} | "
                f"VN- {fmt_corr(s['vn_neg'])} | Emo+ {fmt_corr(s['emo_pos'])} | "
                f"Emo- {fmt_corr(s['emo_neg'])}")
        out.append("\n".join(lines))
    return "\n\n".join(out)


def sec_edges(cfg):
    out = ["## Edge-case recipes (evolved genomes for extreme vocal effects)"]
    for f in sorted(glob.glob(os.path.join(cfg["paths"]["edge_evo_dir"], "*.json"))):
        try:
            d = json.load(open(f))
        except Exception:
            continue
        tgt = d.get("target", os.path.basename(f)[:-5])
        cfg_ = d.get("cfg", {})
        hof = (d.get("hof") or [{}])[0]
        g = hof.get("genome", {})
        out.append(
            f"### {tgt}\nLoRA: {cfg_.get('lora')} | fit={hof.get('fit')} emo={hof.get('emo')} "
            f"blend={hof.get('blend')} wer={hof.get('wer')}\n"
            f"genome: lam={g.get('lam')} temp={g.get('temp')} top_p={g.get('top_p')} "
            f"top_k={g.get('top_k')} desc=\"{g.get('desc')}\" cue=\"{g.get('cue')}\" "
            f"tags={g.get('tags')} text=\"{g.get('text')}\"\n"
            f"(short text + burst tags let the burst dominate; WER~1 is EXPECTED here)")
    return "\n\n".join(out)


def sec_chars_refs(cfg):
    names = []
    cache = os.path.join(cfg["paths"]["char_lora_cache"], "char_names.json")
    try:
        from huggingface_hub import HfApi
        files = HfApi().list_repo_files(cfg["hf"]["char_repo_genuine"])
        names = sorted({f.split("/")[0] for f in files if "/" in f})
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        json.dump(names, open(cache, "w"))
    except Exception:
        if os.path.exists(cache):
            names = json.load(open(cache))
    refs = []
    rj = os.path.join(cfg["paths"]["refs_dir"], "refs.json")
    if os.path.exists(rj):
        for r in json.load(open(rj)):
            refs.append(f"- ref{r['ref_id']}: {r['caption'][:160]}")
    return (
        "## Character LoRAs (name format: char_genuine/<n> or char_refined/<n>; scale 1.0; "
        "refined = SIDON speech-enhanced variants, usually cleaner)\n"
        + ", ".join(names) +
        "\n\n## Reference voices available via load_reference('ref0'..'ref5')\n"
        + "\n".join(refs))


def build(cfg, tool_docs_md, out_path):
    parts = [
        sec_header(cfg),
        "## Tool reference\n" + tool_docs_md,
        sec_prompting(),
        sec_reward(),
        sec_learnings(),
        sec_evolution(cfg),
        sec_emotions(cfg),
        sec_voicenet(cfg),
        sec_edges(cfg),
        sec_chars_refs(cfg),
    ]
    text = "\n\n".join(parts)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        f.write(text)
    return {"path": out_path, "tokens": _tok(text), "chars": len(text)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(os.path.dirname(__file__),
                                                     "configs/single_gpu.yaml"))
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__),
                                                  "context/system_context.md"))
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from tools import tool_docs, register_manifest_tools
    register_manifest_tools(cfg)   # manifest entries (registry/tools.json) included in docs
    info = build(cfg, tool_docs(), args.out)
    print(f"[context] wrote {info['path']}  ~{info['tokens']} tokens ({info['chars']} chars)")


if __name__ == "__main__":
    main()
