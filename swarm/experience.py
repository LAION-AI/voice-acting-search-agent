"""TIER-2 experience exporter (SWARM_PLAN §1/§2): after a mission/task finishes, write
experience/<task_id>/{TASK.md, learnings.md, reports/, artifacts.json} and git push
(pull-rebase; retried). Raw experience is written constantly - it is NOT auto-loaded
into agent context (that is manual/approved/'s job after consolidation).

    from swarm.experience import export_task
    export_task("T0001", "runs/swarm_T0001", task_md="registry/tasks/T0001/TASK.md",
                artifact_paths=[...hf paths...])
"""
import datetime
import json
import os
import re
import subprocess

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _trajectory(workdir):
    traj = []
    p = os.path.join(workdir, "evolution_log.jsonl")
    if os.path.exists(p):
        for line in open(p):
            try:
                d = json.loads(line)
                rows = [r["fitness_mean"] for r in d["results"] if "fitness_mean" in r]
                if rows:
                    traj.append({"best": round(max(rows), 3),
                                 "mean": round(sum(rows) / len(rows), 3)})
            except Exception:
                pass
    return traj


def _supervisor_summary(workdir):
    rounds = []
    p = os.path.join(workdir, "supervisor_log.jsonl")
    if os.path.exists(p):
        for line in open(p):
            try:
                d = json.loads(line)
                rounds.append({
                    "gen": d.get("gen"),
                    "local": (d.get("local") or {}).get("score_0_10"),
                    "gemini": (d.get("gemini") or {}).get("score_0_10"),
                    "directives": (d.get("local") or {}).get("directives")})
            except Exception:
                pass
    return rounds


def export_task(task_id, workdir, task_md=None, artifact_paths=None, push=True):
    exp = os.path.join(HERE, "experience", task_id)
    os.makedirs(os.path.join(exp, "reports"), exist_ok=True)
    now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    if task_md and os.path.exists(task_md):
        content = open(task_md).read()
        content = re.sub(r"(?m)^status: .*$", "status: REVIEW", content)
        content = re.sub(r"(?m)^heartbeat: .*$", f"heartbeat: {now}", content)
        open(os.path.join(exp, "TASK.md"), "w").write(content)

    res_p = os.path.join(workdir, "result.json")
    res = json.load(open(res_p)) if os.path.exists(res_p) else {}
    traj = _trajectory(workdir)
    sup = _supervisor_summary(workdir)

    report = {
        "task_id": task_id, "exported_at": now,
        "tool_calls_used": res.get("tool_calls_used"),
        "fitness_trajectory": traj,
        "supervisor_rounds": sup,
        "final_report": res.get("report", ""),
    }
    with open(os.path.join(exp, "reports", "final.json"), "w") as f:
        json.dump(report, f, indent=1)

    lines = [f"# Learnings - {task_id}", "",
             f"Exported {now}. Tool calls: {res.get('tool_calls_used')}. "
             f"Generations: {len(traj)}.", ""]
    if traj:
        lines += [f"Fitness: start {traj[0]['best']} -> best "
                  f"{max(t['best'] for t in traj)} (mean path "
                  f"{[t['mean'] for t in traj]})", ""]
    if sup:
        lines += ["Supervisor score path (local): "
                  + str([r["local"] for r in sup]), ""]
    lines += ["## Agent final report", "", res.get("report", "(mission unfinished)"), ""]
    mem_p = os.path.join(workdir, "memory.md")
    if os.path.exists(mem_p) and os.path.getsize(mem_p):
        lines += ["## Agent memory", "", open(mem_p).read()]
    open(os.path.join(exp, "learnings.md"), "w").write("\n".join(lines))

    json.dump({"hf_repo": "TTS-AGI/voice-acting-swarm-artifacts",
               "paths": artifact_paths or []},
              open(os.path.join(exp, "artifacts.json"), "w"), indent=1)

    if push:
        git_push(f"experience: {task_id} export")
    return exp


def git_push(msg, retries=3):
    tok = os.environ.get("GH_TOKEN", "")
    url = (f"https://x-access-token:{tok}@github.com/LAION-AI/"
           "voice-acting-search-agent" if tok else "origin")
    for _ in range(retries):
        subprocess.run(["git", "-C", HERE, "add", "-A"], check=False)
        subprocess.run(["git", "-C", HERE, "commit", "-qm", msg], check=False)
        r = subprocess.run(["git", "-C", HERE, "push", "-q", url, "HEAD:main"],
                           capture_output=True)
        if r.returncode == 0:
            return True
        subprocess.run(["git", "-C", HERE, "pull", "-q", "--rebase", url, "main"],
                       check=False)
    return False
