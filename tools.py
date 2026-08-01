"""Tool API for the voice-acting search agent.

Every tool is a plain function taking (ctx, **args) and returning a JSON-serializable
dict.  The registry TOOLS carries the docs + lightweight parameter schemas used both
for validation and for the auto-generated tool reference in the system context.
"""
import json
import os
import shutil
import subprocess
import time
import urllib.request

import numpy as np
import soundfile as sf


# ============================================================== context object
class ToolContext:
    def __init__(self, cfg, engine, scorers, workdir):
        self.cfg = cfg
        self.engine = engine
        self.scorers = scorers
        self.workdir = workdir
        os.makedirs(f"{workdir}/samples", exist_ok=True)
        os.makedirs(f"{workdir}/hall_of_fame", exist_ok=True)
        self.samples = {}     # sample_id -> meta dict (wav on disk)
        self.refs = {}        # reference_id -> {codes, wav_path, sr, caption, scores}
        self._nsample = 0
        self._nref = 0
        self.baseline = None  # {code: mean normalized value} from compute_baseline
        self.memory_path = f"{workdir}/memory.md"
        self.spawn_fn = None  # injected by agent.py
        self.context_refresh_fn = None  # injected by run_agent.py (rebuild system context)

    # ---------------- samples
    def add_sample(self, wav, dur, meta):
        self._nsample += 1
        sid = f"s{self._nsample:04d}"
        path = f"{self.workdir}/samples/{sid}.wav"
        sf.write(path, wav.reshape(-1, 1), self.engine.sr)
        self.samples[sid] = {"path": path, "dur": dur, "scores": None, **meta}
        return sid

    def load_wav(self, sid):
        s = self.samples[sid]
        w, sr = sf.read(s["path"])
        if getattr(w, "ndim", 1) > 1:
            w = w.mean(1)
        return np.asarray(w, np.float32), sr

    def ensure_scores(self, sid):
        s = self.samples[sid]
        if s["scores"] is None:
            w, sr = self.load_wav(sid)
            fs = self.scorers.score_full(w, sr)
            s["scores"] = {"vec": [round(float(x), 4) for x in fs["vec"]],
                           "emo_raw": {k: round(float(v), 4) for k, v in fs["emo_raw"].items()},
                           "vn_raw": {k: round(float(v), 4) for k, v in fs["vn_raw"].items()},
                           "genu_raw": round(float(fs["genu_raw"]), 4),
                           "blend_raw": round(float(fs["blend_raw"]), 4)}
        return s["scores"]

    def slot(self, sid, code):
        vec = np.asarray(self.ensure_scores(sid)["vec"], np.float32)
        return self.scorers.slot_value(vec, code)


def _r(x):
    return round(float(x), 3)


def _summary_row(ctx, sid, metrics):
    sc = ctx.ensure_scores(sid)
    vec = np.asarray(sc["vec"], np.float32)
    row = {"id": sid, "dur": ctx.samples[sid]["dur"],
           "GENU": _r(ctx.scorers.slot_value(vec, "GENU")),
           "BLEND": _r(ctx.scorers.slot_value(vec, "BLEND")),
           "QUALITY": _r(ctx.scorers.quality(vec))}
    for m in metrics or []:
        if m in ("GENU", "BLEND", "QUALITY"):
            continue
        row[m] = _r(ctx.scorers.slot_value(vec, m))
    return row


def _top_slots(ctx, sids, block, k=5):
    """Mean normalized value of every slot in a block across samples -> top-k."""
    vecs = np.stack([np.asarray(ctx.ensure_scores(s)["vec"], np.float32) for s in sids])
    mean = vecs.mean(0)
    lay = ctx.scorers.layout["slots"]
    items = [(s["code"], mean[s["slot"]]) for s in lay if s["block"] == block]
    items.sort(key=lambda t: -t[1])
    return [[c, _r(v)] for c, v in items[:k]]


# ============================================================== tools
def t_list_loras(ctx, family=None, contains=None):
    e = ctx.engine
    fams = {
        "emotion": e.emotion_lora_names,
        "voicenet": e.vn_lora_names,
        "character_genuine": lambda: [f"char_genuine/{n}" for n in e.character_lora_names()],
        "character_refined": lambda: [f"char_refined/{n}" for n in e.character_lora_names()],
    }
    if family and family not in fams:
        return {"error": f"unknown family '{family}'; one of {list(fams)}"}
    out = {}
    for f, fn in fams.items():
        if family and f != family:
            continue
        names = fn()
        if contains:
            names = [n for n in names if contains.lower() in n.lower()]
        out[f] = {"count": len(names), "names": names}
    return out


def t_merge_loras(ctx, loras):
    if not isinstance(loras, list):
        return {"error": "loras must be a list of {name, scale}"}
    norm = []
    warnings = []
    for l in loras:
        if not isinstance(l, dict) or "name" not in l:
            return {"error": f"bad lora entry {l!r}; expected {{name, scale}}"}
        scale = float(l.get("scale", 1.0))
        if abs(scale) > 2.5:
            return {"error": f"scale {scale} for {l['name']} exceeds the hard cap 2.5"}
        if l["name"].startswith("vn_") and scale > 1.5:
            warnings.append(f"{l['name']}@{scale}: VN LoRAs degrade fast above 1.25-1.5")
        norm.append({"name": l["name"], "scale": scale})
    try:
        res = ctx.engine.merge(norm)
    except FileNotFoundError as ex:
        return {"error": str(ex)}
    if warnings:
        res["warnings"] = warnings
    return res


def t_generate(ctx, text, instruction="", language="English", n=4, reference_id=None,
               temp=None, top_p=None, top_k=None, max_frames=None, seed=None):
    n = max(1, min(int(n), 16))
    if seed is None:  # fresh randomness per call (a fixed default made replays identical)
        import random as _rnd
        seed = _rnd.randint(1, 10**8)
    ref_codes = None
    if reference_id is not None:
        if reference_id not in ctx.refs:
            return {"error": f"unknown reference_id '{reference_id}'; call load_reference first"}
        ref_codes = ctx.refs[reference_id]["codes"]
    t0 = time.time()
    outs = ctx.engine.generate(text=text, instruction=instruction, language=language,
                               n=n, ref_codes=ref_codes, temp=temp, top_p=top_p,
                               top_k=top_k, max_frames=max_frames, seed=seed)
    meta = {"text": text, "instruction": instruction, "reference_id": reference_id,
            "merge": list(ctx.engine.active),
            "sampling": {"temp": temp, "top_p": top_p, "top_k": top_k,
                         "max_frames": max_frames, "seed": seed}}
    ids = [ctx.add_sample(o["wav"], o["dur"], dict(meta)) for o in outs]
    return {"sample_ids": ids, "durations": [o["dur"] for o in outs],
            "n_requested": n, "n_generated": len(ids),
            "active_merge": ctx.engine.active, "gen_seconds": round(time.time() - t0, 1)}


def t_score(ctx, sample_ids, metrics=None):
    bad = [s for s in sample_ids if s not in ctx.samples]
    if bad:
        return {"error": f"unknown sample_ids {bad}"}
    rows = [_summary_row(ctx, s, metrics) for s in sample_ids]
    keys = [k for k in rows[0] if k not in ("id",)]
    mean = {k: _r(np.mean([r[k] for r in rows])) for k in keys}
    out = {"samples": rows, "mean": mean}
    if not metrics:
        out["top_emotions"] = _top_slots(ctx, sample_ids, "emotion", 5)
        out["top_voicenet"] = _top_slots(ctx, sample_ids, "voicenet", 5)
    return out


def t_transcribe(ctx, sample_ids):
    out = []
    for sid in sample_ids:
        if sid not in ctx.samples:
            return {"error": f"unknown sample_id '{sid}'"}
        w, sr = ctx.load_wav(sid)
        res = ctx.scorers.transcribe(w, sr, ref_text=ctx.samples[sid].get("text"))
        out.append({"id": sid, "wer_mean": res.get("wer_mean"),
                    "wer_median": res.get("wer_median"),
                    "transcript": res["transcripts"][0][:300]})
    return {"backend": ctx.scorers._asr_backend, "results": out}


def t_caption(ctx, sample_ids):
    out = []
    for sid in sample_ids:
        if sid not in ctx.samples:
            return {"error": f"unknown sample_id '{sid}'"}
        fs = ctx.ensure_scores(sid)
        full = {"vn_raw": fs["vn_raw"], "emo_raw": fs["emo_raw"],
                "genu_raw": fs["genu_raw"], "blend_raw": fs["blend_raw"]}
        out.append({"id": sid, "caption": ctx.scorers.caption(full, seed=hash(sid) & 0xFFFF)})
    return {"captions": out}


def t_speaker_sim(ctx, sample_ids, reference_id):
    if reference_id not in ctx.refs:
        return {"error": f"unknown reference_id '{reference_id}'"}
    ref = ctx.refs[reference_id]
    rw, rsr = sf.read(ref["wav_path"])
    if getattr(rw, "ndim", 1) > 1:
        rw = rw.mean(1)
    sims = []
    for sid in sample_ids:
        if sid not in ctx.samples:
            return {"error": f"unknown sample_id '{sid}'"}
        w, sr = ctx.load_wav(sid)
        sims.append({"id": sid, "cos": _r(ctx.scorers.speaker_sim(w, sr, rw, rsr))})
    return {"sims": sims, "mean": _r(np.mean([s["cos"] for s in sims]))}


def t_load_reference(ctx, path):
    # shortcut: refN -> <refs_dir>/refN.wav
    if not os.path.exists(path):
        cand = os.path.join(ctx.cfg["paths"]["refs_dir"], f"{path}.wav")
        if os.path.exists(cand):
            path = cand
        else:
            return {"error": f"reference path not found: {path}"}
    codes, (w, rsr) = ctx.engine.encode_reference(path)
    ctx._nref += 1
    rid = f"ref{ctx._nref:02d}"
    fs = ctx.scorers.score_full(w, rsr)
    vec = fs["vec"]
    ctx.refs[rid] = {"codes": codes, "wav_path": path, "sr": rsr}
    top_emo = sorted(
        [(s["code"], float(vec[s["slot"]])) for s in ctx.scorers.layout["slots"]
         if s["block"] == "emotion"], key=lambda t: -t[1])[:3]
    return {"reference_id": rid, "path": path, "dur": round(len(w) / rsr, 2),
            "caption": ctx.scorers.caption(fs, seed=7),
            "AROU": _r(ctx.scorers.slot_value(vec, "AROU")),
            "VALN": _r(ctx.scorers.slot_value(vec, "VALN")),
            "GENU": _r(ctx.scorers.slot_value(vec, "GENU")),
            "BLEND": _r(ctx.scorers.slot_value(vec, "BLEND")),
            "top_emotions": [[c, _r(v)] for c, v in top_emo]}


def t_save_best(ctx, sample_ids, note=""):
    saved = []
    for sid in sample_ids:
        if sid not in ctx.samples:
            return {"error": f"unknown sample_id '{sid}'"}
        s = ctx.samples[sid]
        dst = f"{ctx.workdir}/hall_of_fame/{sid}.wav"
        shutil.copy(s["path"], dst)
        meta = {k: v for k, v in s.items() if k not in ("path", "scores")}
        if s["scores"] is not None:
            vec = np.asarray(s["scores"]["vec"], np.float32)
            meta["key_scores"] = {
                "GENU": _r(ctx.scorers.slot_value(vec, "GENU")),
                "BLEND": _r(ctx.scorers.slot_value(vec, "BLEND")),
                "QUALITY": _r(ctx.scorers.quality(vec))}
            meta["vec99"] = s["scores"]["vec"]
        meta["note"] = note
        json.dump(meta, open(f"{ctx.workdir}/hall_of_fame/{sid}.json", "w"), indent=1)
        saved.append(sid)
    return {"saved": saved, "dir": f"{ctx.workdir}/hall_of_fame"}


def t_memory(ctx, action, text=None):
    if action == "append":
        if not text:
            return {"error": "append requires text"}
        with open(ctx.memory_path, "a") as f:
            f.write(text.rstrip() + "\n")
        return {"ok": True, "bytes": os.path.getsize(ctx.memory_path)}
    if action == "read":
        if not os.path.exists(ctx.memory_path):
            return {"memory": ""}
        return {"memory": open(ctx.memory_path).read()[-6000:]}
    return {"error": "action must be 'append' or 'read'"}


def t_compute_baseline(ctx, text, instruction="", n=8, reference_id=None):
    """Generate n samples with NO LoRAs -> per-code mean normalized scores.
    Stored as the 'baseline' referenced by fitness constraints (min: 'baseline')."""
    prev = list(ctx.engine.active)
    ctx.engine.merge([])
    try:
        res = t_generate(ctx, text=text, instruction=instruction, n=n,
                         reference_id=reference_id, seed=4242)
        if "error" in res:
            return res
        sids = res["sample_ids"]
        vecs = np.stack([np.asarray(ctx.ensure_scores(s)["vec"], np.float32) for s in sids])
        mean = vecs.mean(0)
        base = {s["code"]: _r(mean[s["slot"]]) for s in ctx.scorers.layout["slots"]
                if s["block"] in ("emotion", "voicenet")}
        base["GENU"] = _r(mean[97]); base["BLEND"] = _r(mean[98])
        base["QUALITY"] = _r((base["RCQL"] + base["ESTH"]) / 2)
        ctx.baseline = base
        json.dump(base, open(f"{ctx.workdir}/baseline.json", "w"))
        key = {k: base[k] for k in ("GENU", "BLEND", "QUALITY", "AROU", "VALN")}
        return {"n": len(sids), "sample_ids": sids, "key_baseline": key,
                "note": "full per-code baseline stored; constraints may use min='baseline'"}
    finally:
        ctx.engine.merge(prev)


def _fitness_of(ctx, vec, fitness):
    tgt = fitness.get("maximize", {})
    if isinstance(tgt, list):
        tgt = {c: 1.0 for c in tgt}
    vals = {c: ctx.scorers.slot_value(vec, c) for c in tgt}
    score = float(np.average(list(vals.values()), weights=list(tgt.values()))) if vals else 0.0
    penalty = 0.0
    pen_w = float(fitness.get("penalty", 2.0))
    detail = dict(vals)
    for c in fitness.get("constraints", []):
        code = c["code"]
        mn = c.get("min", "baseline")
        if mn == "baseline":
            if ctx.baseline is None:
                raise ValueError("constraint min='baseline' but compute_baseline was never called")
            mn = ctx.baseline[code]
        v = ctx.scorers.slot_value(vec, code)
        detail[code] = v
        penalty += max(0.0, float(mn) - v) * pen_w
    return score - penalty, detail


def t_run_generation(ctx, genomes, fitness, n_per_genome=None):
    """Evaluate one evolution generation: for each genome merge->generate->score,
    fitness = mean over samples of (weighted target mean - constraint penalties)."""
    if not isinstance(genomes, list) or not genomes:
        return {"error": "genomes must be a non-empty list"}
    if len(genomes) > 12:
        return {"error": "max 12 genomes per generation"}
    npg = int(n_per_genome or ctx.cfg["evolution"]["samples_per_genome"])
    dup_warning = None
    sig = json.dumps(genomes, sort_keys=True, default=str)
    if sig == getattr(ctx, "_last_genomes_sig", None):
        dup_warning = ("WARNING: this genome list is IDENTICAL to the previous "
                       "generation — mutations were NOT applied. The next call must "
                       "actually change scales/desc/cue/loras of the mutated genomes.")
    ctx._last_genomes_sig = sig
    results = []
    for gi, g in enumerate(genomes):
        try:
            mr = t_merge_loras(ctx, g.get("loras", []))
            if "error" in mr:
                results.append({"idx": gi, "error": mr["error"]})
                continue
            instruction = g.get("instruction")
            if instruction is None:
                desc = g.get("desc", "")
                cue = g.get("cue", "")
                instruction = (f"GENERAL: {desc}\nSCRIPT:\n{cue}" if desc
                               else f"SCRIPT:\n{cue}")
            gr = t_generate(ctx, text=g.get("text", ""), instruction=instruction,
                            language=g.get("language", "English"), n=npg,
                            reference_id=g.get("reference_id"),
                            temp=g.get("temp"), top_p=g.get("top_p"), top_k=g.get("top_k"),
                            max_frames=g.get("max_frames"), seed=g.get("seed"))
            if "error" in gr:
                results.append({"idx": gi, "error": gr["error"]})
                continue
            sids = gr["sample_ids"]
            if not sids:
                results.append({"idx": gi, "error": "no non-empty samples generated"})
                continue
            fits, best_sid, best_fit = [], None, -1e9
            agg = {}
            for sid in sids:
                vec = np.asarray(ctx.ensure_scores(sid)["vec"], np.float32)
                f, detail = _fitness_of(ctx, vec, fitness)
                fits.append(f)
                for k, v in detail.items():
                    agg.setdefault(k, []).append(v)
                agg.setdefault("QUALITY", []).append(ctx.scorers.quality(vec))
                if f > best_fit:
                    best_fit, best_sid = f, sid
            row = {"idx": gi, "fitness_mean": _r(np.mean(fits)),
                   "fitness_best": _r(best_fit), "n": len(sids),
                   "best_sample_id": best_sid,
                   "means": {k: _r(np.mean(v)) for k, v in agg.items()}}
            results.append(row)
        except Exception as ex:  # keep the generation going
            results.append({"idx": gi, "error": str(ex)[:200]})
    log = {"t": time.time(), "fitness": fitness,
           "genomes": [{k: v for k, v in g.items() if k != "instruction"} for g in genomes],
           "results": results}
    with open(f"{ctx.workdir}/evolution_log.jsonl", "a") as f:
        f.write(json.dumps(log) + "\n")
    ranked = sorted([r for r in results if "fitness_mean" in r],
                    key=lambda r: -r["fitness_mean"])
    out = {"results": results,
           "ranking": [[r["idx"], r["fitness_mean"]] for r in ranked]}
    if dup_warning:
        out["warning"] = dup_warning
    return out


def t_spawn_subagent(ctx, task, budget=None):
    if ctx.spawn_fn is None:
        return {"error": "subagent spawning not available in this run"}
    budget = int(budget or ctx.cfg["agent"]["subagent_default_budget"])
    return ctx.spawn_fn(task, budget)


def t_fetch_manual(ctx, topic=None, name=None):
    """No args: re-pull manuals + rebuild the distilled system context (auto-refresh).
    With topic+name: fetch one live manual entry (falls back to the local mirror)."""
    if topic is None:
        if ctx.context_refresh_fn is None:
            return {"error": "context refresh not available"}
        info = ctx.context_refresh_fn()
        return {"refreshed": True, **info}
    m = ctx.cfg["manuals"]
    urls = {"emotion": m["emo_raw_base"], "voicenet": m["vn_raw_base"]}
    if topic in urls and name:
        try:
            with urllib.request.urlopen(urls[topic] + name + ".md", timeout=20) as r:
                return {"topic": topic, "name": name, "content": r.read().decode()[:6000]}
        except Exception as ex:
            pass  # fall through to local mirror
    p = ctx.cfg["paths"]
    try:
        if topic == "emotion" and name:
            tips = json.load(open(p["manual_emo_tips"]))
            return {"topic": topic, "name": name, "content": tips.get(name, "not found")}
        if topic == "voicenet" and name:
            ent = json.load(open(p["manual_vn_entries"]))["entries"]
            return {"topic": topic, "name": name,
                    "content": json.dumps(ent.get(name, "not found"))[:6000]}
        if topic == "edge" and name:
            d = json.load(open(os.path.join(p["edge_evo_dir"], f"{name}.json")))
            d.pop("hof", None)
            return {"topic": topic, "name": name, "content": json.dumps(d)[:6000]}
    except Exception as ex:
        return {"error": f"manual fetch failed: {str(ex)[:150]}"}
    return {"error": "specify topic in {emotion, voicenet, edge} plus name, or no args to refresh"}


def t_push_results(ctx, path, message):
    """Commit+push a path inside the agent repo checkout (uses GITHUB_TOKEN env)."""
    repo_dir = ctx.cfg.get("repo_dir") or os.path.dirname(os.path.abspath(__file__))
    tok = os.environ.get("GITHUB_TOKEN", "")
    if not tok:
        return {"error": "GITHUB_TOKEN not set"}
    url = ctx.cfg["repo"]["url"].replace("https://", f"https://x-access-token:{tok}@")
    try:
        subprocess.run(["git", "-C", repo_dir, "add", path], check=True, capture_output=True)
        r = subprocess.run(["git", "-C", repo_dir, "commit", "-m", message],
                           capture_output=True, text=True)
        if r.returncode != 0 and "nothing to commit" not in r.stdout + r.stderr:
            return {"error": (r.stdout + r.stderr)[-300:]}
        p = subprocess.run(["git", "-C", repo_dir, "push", url, "HEAD:main"],
                           capture_output=True, text=True)
        if p.returncode != 0:
            return {"error": (p.stdout + p.stderr)[-300:]}
        return {"pushed": path}
    except subprocess.CalledProcessError as ex:
        return {"error": str(ex)[:300]}


# ============================================================== registry
TOOLS = {
    "list_loras": {
        "fn": t_list_loras,
        "desc": "List available LoRAs. Families: emotion (40), voicenet (114: vn_<DIM>_<high|low>), "
                "character_genuine, character_refined (120 each, name format char_genuine/<n>).",
        "params": {"family": {"type": str, "required": False},
                   "contains": {"type": str, "required": False}},
    },
    "merge_loras": {
        "fn": t_merge_loras,
        "desc": "Activate a LoRA merge set for subsequent generate calls. "
                "loras=[{name, scale}]; [] = plain base model. Caps: emotion<=1.9, vn<=1.25 recommended.",
        "params": {"loras": {"type": list, "required": True}},
    },
    "generate": {
        "fn": t_generate,
        "desc": "Generate n audio samples with the current merge. instruction is the voice-acting "
                "caption ('GENERAL: <voice description>\\nSCRIPT:\\n(<delivery cue>)'), text is the spoken "
                "script (>=20 words recommended). Returns sample_ids.",
        "params": {"text": {"type": str, "required": True},
                   "instruction": {"type": str, "required": False},
                   "language": {"type": str, "required": False},
                   "n": {"type": int, "required": False},
                   "reference_id": {"type": str, "required": False},
                   "temp": {"type": float, "required": False},
                   "top_p": {"type": float, "required": False},
                   "top_k": {"type": int, "required": False},
                   "max_frames": {"type": int, "required": False},
                   "seed": {"type": int, "required": False}},
    },
    "score": {
        "fn": t_score,
        "desc": "Score samples: always returns GENU, BLEND, QUALITY per sample + mean; "
                "metrics=[codes] adds specific slots (e.g. ['AROU','Fear','S_STRY']). "
                "Without metrics also returns top-5 emotions + top-5 voicenet dims.",
        "params": {"sample_ids": {"type": list, "required": True},
                   "metrics": {"type": list, "required": False}},
    },
    "transcribe": {
        "fn": t_transcribe,
        "desc": "ASR transcription (3 decode variants) + WER vs the prompted text.",
        "params": {"sample_ids": {"type": list, "required": True}},
    },
    "caption": {
        "fn": t_caption,
        "desc": "Procedural voice caption (GENERAL line) describing how each sample sounds.",
        "params": {"sample_ids": {"type": list, "required": True}},
    },
    "speaker_sim": {
        "fn": t_speaker_sim,
        "desc": "ECAPA cosine similarity of samples vs a loaded reference voice.",
        "params": {"sample_ids": {"type": list, "required": True},
                   "reference_id": {"type": str, "required": True}},
    },
    "load_reference": {
        "fn": t_load_reference,
        "desc": "Load a reference voice wav (path, or shortcut 'ref0'..'ref5'). Returns "
                "reference_id (for generate/speaker_sim), its caption and key scores.",
        "params": {"path": {"type": str, "required": True}},
    },
    "save_best": {
        "fn": t_save_best,
        "desc": "Persist hall-of-fame samples (wav + genome + scores) to the workdir.",
        "params": {"sample_ids": {"type": list, "required": True},
                   "note": {"type": str, "required": False}},
    },
    "memory": {
        "fn": t_memory,
        "desc": "Persistent scratch notes. action='append' with text, or action='read'. "
                "Write findings here BEFORE they scroll out of context.",
        "params": {"action": {"type": str, "required": True},
                   "text": {"type": str, "required": False}},
    },
    "compute_baseline": {
        "fn": t_compute_baseline,
        "desc": "Generate n no-LoRA samples for a text and store per-code baseline means. "
                "Required before fitness constraints with min='baseline'.",
        "params": {"text": {"type": str, "required": True},
                   "instruction": {"type": str, "required": False},
                   "n": {"type": int, "required": False},
                   "reference_id": {"type": str, "required": False}},
    },
    "run_generation": {
        "fn": t_run_generation,
        "desc": "Evaluate ONE evolution generation (batch merge+generate+score). "
                "genomes=[{loras:[{name,scale}], desc, cue, text, temp, top_p, top_k, "
                "reference_id?, seed?}], fitness={maximize:[codes]|{code:w}, "
                "constraints:[{code, min:'baseline'|num}], penalty:2.0}. "
                "Returns per-genome mean fitness (mean-of-n), per-code means, best_sample_id, ranking.",
        "params": {"genomes": {"type": list, "required": True},
                   "fitness": {"type": dict, "required": True},
                   "n_per_genome": {"type": int, "required": False}},
    },
    "spawn_subagent": {
        "fn": t_spawn_subagent,
        "desc": "Spawn a fresh-context copy of yourself for a focused subtask; returns its report.",
        "params": {"task": {"type": str, "required": True},
                   "budget": {"type": int, "required": False}},
    },
    "fetch_manual": {
        "fn": t_fetch_manual,
        "desc": "No args: re-pull the live manuals and refresh the system context. "
                "Or topic in {emotion, voicenet, edge} + name for one full entry.",
        "params": {"topic": {"type": str, "required": False},
                   "name": {"type": str, "required": False}},
    },
    "push_results": {
        "fn": t_push_results,
        "desc": "git add+commit+push a results path inside the agent repo (GITHUB_TOKEN env).",
        "params": {"path": {"type": str, "required": True},
                   "message": {"type": str, "required": True}},
    },
    "finish": {
        "fn": None,  # handled by the agent loop
        "desc": "End the mission with a final report: {report: '...'}.",
        "params": {"report": {"type": str, "required": True}},
    },
}


def validate_args(tool, args):
    spec = TOOLS[tool]["params"]
    if not isinstance(args, dict):
        return f"args must be an object, got {type(args).__name__}"
    unknown = [k for k in args if k not in spec]
    if unknown:
        return f"unknown args {unknown}; allowed: {list(spec)}"
    missing = [k for k, v in spec.items() if v["required"] and k not in args]
    if missing:
        return f"missing required args {missing}"
    for k, v in args.items():
        want = spec[k]["type"]
        if v is None:
            continue
        if want in (int, float) and isinstance(v, (int, float)):
            continue
        if not isinstance(v, want):
            return f"arg '{k}' should be {want.__name__}, got {type(v).__name__}"
    return None


def run_tool(ctx, tool, args):
    err = validate_args(tool, args)
    if err:
        return {"error": err}
    try:
        return TOOLS[tool]["fn"](ctx, **args)
    except Exception as ex:
        import traceback
        return {"error": f"{type(ex).__name__}: {str(ex)[:300]}",
                "trace": traceback.format_exc()[-400:]}


def tool_docs():
    lines = []
    for name, t in TOOLS.items():
        ps = ", ".join(
            f"{k}{'' if v['required'] else '?'}: {v['type'].__name__}"
            for k, v in t["params"].items())
        lines.append(f"### {name}({ps})\n{t['desc']}")
    return "\n\n".join(lines)
