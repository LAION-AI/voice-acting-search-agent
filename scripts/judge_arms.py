#!/usr/bin/env python3
"""S4 judge: score each model arm (12B / 27B(=31B dense) / MoE) on its 4 single-dim
missions with Gemini (Hyprlab gemini-3.6-flash; thinking-native — do NOT send
thinking_config; max_tokens >= 400). Audio attached when the endpoint accepts it,
else text-only (noted). Emits MODEL_COMPARISON.md.

  python scripts/judge_arms.py [--out MODEL_COMPARISON.md]
"""
import argparse
import base64
import glob
import json
import os
import re
import subprocess
import sys
import tempfile
import urllib.request

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

ARMS = {"12B": "runs/singledim_{d}", "27B": "runs/singledim27_{d}",
        "MoE": "runs/singledimmoe_{d}"}
DIMS = ["arousal", "valence", "explicit", "storyteller"]
KEY = os.environ.get("HYPRLAB_API_KEY", "")
MODEL = "gemini-3.6-flash"

PROMPT = """You are judging THREE LLM "brains" (12B, 27B, MoE) that each autonomously drove
the SAME voice-acting search system on the SAME 4 missions (maximize one vocal dimension —
arousal / valence / explicitness / storytelling — while staying genuine and intelligible)
with the SAME tool budget (60 calls, 8 generations x 8 genomes). You get each arm's final
mission reports plus fitness trajectories{audio_note}.

Score EACH ARM 0-10 on:
1. search_process: strategy diversity, systematic dose sweeps, self-correction on failure,
   use of evidence (did it react to WER/fitness signals or repeat itself?)
2. result_quality: final cohort fitness reached per dimension, believability of recipes
3. report_clarity: is the final report specific, ranked, reproducible?

Same-budget framing: judge efficiency of the SEARCH, not verbosity.
Reply ONLY with JSON:
{{"arms": {{"12B": {{"search_process": n, "result_quality": n, "report_clarity": n,
  "comment": "..."}}, "27B": {{...}}, "MoE": {{...}}}},
  "winner": "12B|27B|MoE", "verdict": "2-3 sentences",
  "notable_quotes": ["<arm>: <verbatim snippet that impressed or worried you>", ...]}}
"""


def hyprlab(content, max_tokens=1500):
    body = {"model": MODEL, "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": content}]}
    req = urllib.request.Request(
        "https://api.hyprlab.io/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {KEY}"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.loads(r.read())["choices"][0]["message"]["content"]


def arm_summary(tag, pattern):
    parts = []
    audio = []
    for d in DIMS:
        run = os.path.join(HERE, pattern.format(d=d))
        rj = os.path.join(run, "result.json")
        if not os.path.exists(rj):
            parts.append(f"### {d}: MISSION DID NOT FINISH")
            continue
        res = json.load(open(rj))
        traj = []
        el = os.path.join(run, "evolution_log.jsonl")
        if os.path.exists(el):
            for line in open(el):
                try:
                    dd = json.loads(line)
                    rows = [r["fitness_mean"] for r in dd["results"] if "fitness_mean" in r]
                    if rows:
                        traj.append(round(max(rows), 3))
                except Exception:
                    pass
        parts.append(f"### {d} (tool calls {res.get('tool_calls_used')}, "
                     f"best-fitness trajectory {traj})\n{res['report'][:2500]}")
        hof = sorted(glob.glob(os.path.join(run, "hall_of_fame", "*.wav")))
        if hof:
            audio.append((d, hof[0]))
    return "\n\n".join(parts), audio


def to_mp3_b64(wavs):
    """Concatenate top wavs (one per dim) -> one mp3 b64."""
    import numpy as np
    import soundfile as sf
    seg = []
    sr0 = None
    for _, p in wavs:
        w, sr = sf.read(p)
        if getattr(w, "ndim", 1) > 1:
            w = w.mean(1)
        sr0 = sr0 or sr
        seg.append(np.asarray(w, np.float32))
        seg.append(np.zeros(sr0, np.float32))
    if not seg:
        return None
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as t:
        sf.write(t.name, np.concatenate(seg).reshape(-1, 1), sr0)
        wp = t.name
    mp = wp[:-4] + ".mp3"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", wp, "-ac", "1",
                    "-b:a", "96k", mp], check=True)
    os.remove(wp)
    b = base64.b64encode(open(mp, "rb").read()).decode()
    os.remove(mp)
    return b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "MODEL_COMPARISON.md"))
    args = ap.parse_args()
    blocks = []
    all_audio = []
    for tag, pat in ARMS.items():
        text, audio = arm_summary(tag, pat)
        blocks.append(f"## ARM {tag}\n{text}")
        for d, p in audio:
            all_audio.append((f"{tag}/{d}", p))
    body_text = PROMPT + "\n\n" + "\n\n".join(blocks)

    content = None
    audio_used = False
    b64 = to_mp3_b64(all_audio[:6])
    if b64:
        try:  # probe audio support once
            content = [{"type": "input_audio",
                        "input_audio": {"data": b64, "format": "mp3"}},
                       {"type": "text", "text": body_text.replace(
                           "{audio_note}",
                           " and ONE audio file with the arms' top takes in order " +
                           ", ".join(t for t, _ in all_audio[:6]) + " (1s gaps)")}]
            txt = hyprlab(content)
            audio_used = True
        except Exception as ex:
            print(f"[judge] audio rejected ({str(ex)[:120]}); falling back to text-only")
            content = None
    if content is None:
        txt = hyprlab(body_text.replace("{audio_note}", ""))

    m = re.search(r"\{.*\}", txt, re.S)
    obj = json.loads(m.group(0)) if m else {"raw": txt}
    lines = ["# Model-arm comparison (same missions, same budget)",
             "",
             f"Judge: Hyprlab {MODEL} (thinking-native), audio attached: {audio_used}",
             "",
             "| arm | search process | result quality | report clarity | comment |",
             "|---|---|---|---|---|"]
    for tag in ARMS:
        a = obj.get("arms", {}).get(tag, {})
        lines.append(f"| {tag} | {a.get('search_process','-')} | "
                     f"{a.get('result_quality','-')} | {a.get('report_clarity','-')} | "
                     f"{a.get('comment','-')} |")
    lines += ["", f"**Winner: {obj.get('winner','?')}** — {obj.get('verdict','')}", ""]
    for q in obj.get("notable_quotes", []):
        lines.append(f"> {q}")
    lines += ["", "## Raw verdict JSON", "```json",
              json.dumps(obj, indent=1)[:4000], "```"]
    open(args.out, "w").write("\n".join(lines) + "\n")
    print(f"[judge] wrote {args.out}  (audio_used={audio_used})")


if __name__ == "__main__":
    main()
