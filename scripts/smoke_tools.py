#!/usr/bin/env python3
"""No-LLM smoke test: load the full stack on one GPU and exercise every tool once.
    CUDA_VISIBLE_DEVICES=1 python scripts/smoke_tools.py
"""
import json
import os
import sys
import time

here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, here)
os.environ.setdefault("HF_HOME", "/tmp/hf_cache")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import yaml
cfg = yaml.safe_load(open(os.path.join(here, "configs/single_gpu.yaml")))
cfg["repo_dir"] = here

from engine import Engine
from scorers import ScorerStack
import tools as T

t0 = time.time()
engine = Engine(cfg)
print(f"[smoke] engine loaded {time.time()-t0:.0f}s alloc={engine.gpu_memory_gb()}GB")
scorers = ScorerStack(cfg)
print(f"[smoke] scorers loaded {time.time()-t0:.0f}s alloc={engine.gpu_memory_gb()}GB")

wd = os.path.join(here, "runs", "smoke")
os.makedirs(wd, exist_ok=True)
ctx = T.ToolContext(cfg, engine, scorers, wd)

def call(tool, **args):
    t = time.time()
    r = T.run_tool(ctx, tool, args)
    s = json.dumps(r, default=str)
    print(f"\n== {tool} ({time.time()-t:.1f}s) ==\n{s[:900]}")
    assert "error" not in r, f"{tool} FAILED: {r}"
    return r

call("list_loras", family="emotion")
call("list_loras", family="voicenet", contains="AROU")
call("list_loras", family="character_genuine", contains="gravel")
ref = call("load_reference", path="ref1")
rid = ref["reference_id"]
call("merge_loras", loras=[{"name": "Fear", "scale": 1.0}, {"name": "vn_AROU_high", "scale": 0.75}])
g = call("generate", text="Something is in here with us, oh god, please, we have to get out right now, hurry, please just hurry!",
         instruction="GENERAL: A terrified voice, trembling and breathless.\nSCRIPT:\n(trembling, terrified)",
         n=2, reference_id=rid)
sids = g["sample_ids"]
call("score", sample_ids=sids)
call("score", sample_ids=sids, metrics=["AROU", "Fear", "S_STRY"])
call("transcribe", sample_ids=sids)
call("caption", sample_ids=sids)
call("speaker_sim", sample_ids=sids, reference_id=rid)
call("merge_loras", loras=[])
call("compute_baseline", text="The gates will open at dawn and the caravan leaves soon after, so gather everything you need today.", n=2)
call("run_generation",
     genomes=[{"loras": [{"name": "Anger", "scale": 1.25}],
               "desc": "A furious voice, seething and exploding into a rant.",
               "cue": "(furious, ranting)",
               "text": "How dare you do this to me after everything, I am absolutely furious right now, you have no idea how enraged I am!"},
              {"loras": []}],
     fitness={"maximize": ["AROU", "VALN"], "constraints": [{"code": "GENU", "min": "baseline"}]},
     n_per_genome=2)
call("save_best", sample_ids=sids[:1], note="smoke test")
call("memory", action="append", text="- smoke test note")
call("memory", action="read")
call("fetch_manual", topic="voicenet", name="AROU")
call("fetch_manual", topic="emotion", name="Fear")
print(f"\n[smoke] ALL TOOLS OK  alloc={engine.gpu_memory_gb()}GB")
