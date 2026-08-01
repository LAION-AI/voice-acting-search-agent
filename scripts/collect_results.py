#!/usr/bin/env python3
"""Collect a run's artifacts into results/<name>/ for committing:
transcript (b64-stripped), report, memory, evolution log, baseline,
hall-of-fame genomes + mp3-compressed audio.

    python scripts/collect_results.py runs/canary_arousal_ref results/canary/arousal_ref
"""
import glob
import json
import os
import re
import shutil
import subprocess
import sys

B64RE = re.compile(r'"(?:b64|mp3|wav_b64)":\s*"[A-Za-z0-9+/=]{200,}"')

def strip_b64(line):
    return B64RE.sub('"b64":"[stripped]"', line)

def main(src, dst):
    os.makedirs(dst, exist_ok=True)
    for f in glob.glob(f"{src}/transcript*.jsonl"):
        with open(f) as i, open(os.path.join(dst, os.path.basename(f)), "w") as o:
            for line in i:
                o.write(strip_b64(line))
    for f in ("report.md", "memory.md", "result.json", "baseline.json", "evolution_log.jsonl"):
        p = os.path.join(src, f)
        if os.path.exists(p):
            shutil.copy(p, dst)
    hof = os.path.join(src, "hall_of_fame")
    if os.path.isdir(hof):
        hd = os.path.join(dst, "hall_of_fame")
        os.makedirs(hd, exist_ok=True)
        for j in glob.glob(f"{hof}/*.json"):
            shutil.copy(j, hd)
        for w in glob.glob(f"{hof}/*.wav"):
            mp3 = os.path.join(hd, os.path.basename(w)[:-4] + ".mp3")
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", w,
                            "-ac", "1", "-b:a", "128k", mp3], check=False)
    print(f"[collect] {src} -> {dst}")

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
