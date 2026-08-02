#!/usr/bin/env python3
"""S5 orchestrator: run the 8 edge-case swarm tasks (registry/tasks/T0001-T0008)
against the winning 31B brain (:8802), Gemini ACTIVE + MOSS score-only shadow (:8809).

Batch 1: T0001-T0006 on GPUs 1,3,4,5,6,7.  Batch 2: T0007-T0008.
Per task on completion: experience export (+git push) and HF artifact upload.
Detached-safe: run under setsid; state in runs/s5_state.json.
"""
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

LLM_URL = "http://127.0.0.1:8802/v1"
BUDGET = 26
BATCHES = [
    [("T0001", 1), ("T0002", 3), ("T0003", 4), ("T0004", 5), ("T0005", 6), ("T0006", 7)],
    [("T0007", 3), ("T0008", 4)],
]


def log(msg):
    print(f"[s5 {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def launch(task_id, gpu):
    mission = open(f"{HERE}/registry/tasks/{task_id}/mission.txt").read()
    wd = f"runs/swarm_{task_id}"
    cmd = [sys.executable, "run_agent.py", "--mission", mission,
           "--budget-tool-calls", str(BUDGET), "--gpu", str(gpu),
           "--no-start-llm", "--llm-url", LLM_URL, "--supervised",
           "--workdir", wd]
    lf = open(f"/tmp/vaa_s5_{task_id}.log", "w")
    p = subprocess.Popen(cmd, cwd=HERE, stdout=lf, stderr=lf,
                         start_new_session=True)
    log(f"launched {task_id} on GPU{gpu} pid {p.pid}")
    return p


def postprocess(task_id):
    try:
        from swarm.experience import export_task
        from swarm.artifacts import upload_task
        wd = os.path.join(HERE, f"runs/swarm_{task_id}")
        paths = upload_task(task_id, wd)
        log(f"{task_id}: uploaded {len(paths)} artifacts")
        export_task(task_id, wd,
                    task_md=os.path.join(HERE, f"registry/tasks/{task_id}/TASK.md"),
                    artifact_paths=paths, push=True)
        log(f"{task_id}: experience exported + pushed")
    except Exception as ex:
        log(f"{task_id}: POSTPROCESS ERROR {str(ex)[:200]}")


def main():
    state = {"done": [], "started": time.time()}
    for bi, batch in enumerate(BATCHES):
        log(f"=== batch {bi + 1}: {[t for t, _ in batch]}")
        procs = {t: launch(t, g) for t, g in batch}
        pending = set(procs)
        while pending:
            time.sleep(60)
            for t in sorted(pending):
                res = os.path.join(HERE, f"runs/swarm_{t}/result.json")
                alive = procs[t].poll() is None
                if os.path.exists(res) or not alive:
                    pending.discard(t)
                    ok = os.path.exists(res)
                    log(f"{t}: {'FINISHED' if ok else 'DIED (no result.json)'}")
                    postprocess(t)
                    state["done"].append({"task": t, "ok": ok, "t": time.time()})
                    json.dump(state, open(os.path.join(HERE, "runs/s5_state.json"), "w"))
    log("=== S5 ALL TASKS PROCESSED")
    json.dump(state, open(os.path.join(HERE, "runs/s5_state.json"), "w"))


if __name__ == "__main__":
    main()
