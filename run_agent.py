#!/usr/bin/env python3
"""CLI entry point.

    python run_agent.py --mission "raise Arousal on ref1" --budget-tool-calls 40 --gpu 1

Starts (or reuses) the vLLM server on the SAME GPU, loads the TTS engine + scorer
stack, assembles the system context, and runs the ReAct agent until it finishes
or exhausts its tool-call budget.  Everything lands in --workdir.
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mission", required=True)
    ap.add_argument("--budget-tool-calls", type=int, default=None)
    ap.add_argument("--gpu", type=int, default=None, help="CUDA device index (default from config)")
    ap.add_argument("--config", default=None)
    ap.add_argument("--workdir", default=None, help="default: runs/<slug>-<ts>")
    ap.add_argument("--start-llm", action="store_true", default=True)
    ap.add_argument("--no-start-llm", dest="start_llm", action="store_false",
                    help="assume a vLLM server is already running (4-GPU node workers)")
    ap.add_argument("--rebuild-context", action="store_true")
    ap.add_argument("--supervised", action="store_true",
                    help="enable the acoustic supervisor loop (local MOSS active + Gemini shadow)")
    args = ap.parse_args()
    return args


def main():
    args = parse_args()
    here = os.path.dirname(os.path.abspath(__file__))
    import yaml
    cfg_path = args.config or os.path.join(here, "configs/single_gpu.yaml")
    cfg = yaml.safe_load(open(cfg_path))
    if args.llm_url:
        cfg["llm"]["base_url"] = args.llm_url
    cfg["repo_dir"] = here

    gpu = cfg["gpu"] if args.gpu is None else args.gpu
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    os.environ.setdefault("HF_HOME", "/tmp/hf_cache")
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    slug = re.sub(r"[^a-z0-9]+", "-", args.mission.lower())[:40].strip("-")
    workdir = args.workdir or os.path.join(here, "runs", f"{slug}-{int(time.time())}")
    os.makedirs(workdir, exist_ok=True)
    print(f"[run] workdir={workdir} gpu={gpu}", flush=True)

    sys.path.insert(0, here)
    import llm as L

    client = L.LLMClient(cfg)
    proc = None
    if not client.healthy():
        if not args.start_llm:
            print("[run] ERROR: LLM server not reachable and --no-start-llm given", file=sys.stderr)
            sys.exit(2)
        print(f"[run] starting vLLM ({cfg['llm']['model']}) on GPU {gpu} "
              f"(gpu_mem_util={cfg['llm']['gpu_memory_utilization']})...", flush=True)
        proc = L.start_server(cfg, os.path.join(workdir, "vllm.log"),
                              env_extra={"CUDA_VISIBLE_DEVICES": str(gpu)})
        if not L.wait_healthy(client, proc):
            print("[run] ERROR: vLLM did not become healthy", file=sys.stderr)
            sys.exit(2)
    print("[run] LLM healthy", flush=True)

    # ---- heavy stack ----
    from engine import Engine
    from scorers import ScorerStack
    from tools import ToolContext, tool_docs, register_manifest_tools
    import build_context as BC
    from agent import Agent

    t0 = time.time()
    engine = Engine(cfg)
    scorers = ScorerStack(cfg)
    print(f"[run] engine+scorers loaded in {time.time()-t0:.0f}s "
          f"(torch alloc {engine.gpu_memory_gb()} GB)", flush=True)

    register_manifest_tools(cfg)   # manifest tools (registry/tools.json) before docs render
    ctx_path = os.path.join(here, "context/system_context.md")
    if args.rebuild_context or not os.path.exists(ctx_path):
        info = BC.build(cfg, tool_docs(), ctx_path)
        print(f"[run] system context: ~{info['tokens']} tokens", flush=True)

    ctx = ToolContext(cfg, engine, scorers, workdir)
    ctx.mission = args.mission
    if args.supervised:
        from swarm.supervisor import LocalSupervisor, GeminiSupervisor
        sup_cfg = cfg.get("supervisor", {})
        local = LocalSupervisor(sup_cfg.get("local_url", "http://127.0.0.1:8802"))
        if not local.healthy():
            print("[run] WARNING: local supervisor server not reachable "
                  "(swarm/supervisor_server.py on GPU2)", flush=True)
        ctx.supervisors = {"local": local, "gemini": GeminiSupervisor()}
        print("[run] supervisors: local MOSS (active) + Gemini (shadow)", flush=True)

    def refresh_context():
        info = BC.build(cfg, tool_docs(), ctx_path)
        return {"tokens": info["tokens"]}
    ctx.context_refresh_fn = refresh_context

    def get_system_context():
        return open(ctx_path).read()

    budget = args.budget_tool_calls or cfg["agent"]["budget_tool_calls"]
    agent = Agent(cfg, client, ctx, get_system_context, mission=args.mission,
                  budget=budget, transcript_path=os.path.join(workdir, "transcript.jsonl"))
    res = agent.run()

    with open(os.path.join(workdir, "report.md"), "w") as f:
        f.write(f"# Mission\n{args.mission}\n\n# Report\n{res['report']}\n")
    print("\n========== FINAL REPORT ==========")
    print(res["report"])
    print(f"\n[run] tool calls used: {res['tool_calls_used']}  workdir: {workdir}")
    smi = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used",
                          "--format=csv,noheader"], capture_output=True, text=True).stdout
    print(f"[run] GPU memory now:\n{smi}")
    json.dump({"mission": args.mission, "report": res["report"],
               "tool_calls_used": res["tool_calls_used"]},
              open(os.path.join(workdir, "result.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
