"""HF artifact uploader (SWARM_PLAN §4): audio + reports are too big for git, so each
task's artifacts go to the PRIVATE dataset repo TTS-AGI/voice-acting-swarm-artifacts,
keyed by task_id. Idempotent: files are hashed and skipped when unchanged (tracked in a
local .uploaded index per task).

    from swarm.artifacts import upload_task
    upload_task("T0001", "runs/swarm_T0001")     # audio, logs, reports
"""
import hashlib
import json
import os

REPO = "TTS-AGI/voice-acting-swarm-artifacts"

INCLUDE = (".mp3", ".wav", ".json", ".jsonl", ".md", ".txt")


def _api():
    from huggingface_hub import HfApi
    return HfApi(token=os.environ.get("HF_TOKEN"))


def ensure_repo():
    api = _api()
    api.create_repo(REPO, repo_type="dataset", private=True, exist_ok=True)
    readme = (
        "# voice-acting-swarm-artifacts (private)\n\n"
        "Per-task artifacts of the LAION voice-acting search swarm "
        "(code: LAION-AI/voice-acting-search-agent; task registry + learnings live "
        "in that repo's registry/ and experience/ - THIS repo holds the big files).\n\n"
        "Layout:\n"
        "```\n"
        "tasks/<task_id>/\n"
        "  hall_of_fame/*.mp3|.json   best takes + genome metadata\n"
        "  samples_top/*.mp3          top-of-cohort takes per generation\n"
        "  evolution_log.jsonl        per-generation genomes + fitness\n"
        "  supervisor_log.jsonl       dual supervisor verdicts (local MOSS + Gemini)\n"
        "  report.md                  agent final report\n"
        "  learnings.md               distilled findings (also in git experience/)\n"
        "```\n")
    api.upload_file(path_or_fileobj=readme.encode(), path_in_repo="README.md",
                    repo_id=REPO, repo_type="dataset",
                    commit_message="readme")


def _sha(path, n=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(n)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def upload_task(task_id, workdir, extra_files=None):
    """Upload a task's artifacts; returns list of uploaded repo paths."""
    api = _api()
    ensure_repo()
    idx_path = os.path.join(workdir, ".uploaded.json")
    idx = json.load(open(idx_path)) if os.path.exists(idx_path) else {}
    ops = []
    todo = []
    for root, _, files in os.walk(workdir):
        rel_root = os.path.relpath(root, workdir)
        if rel_root.startswith("samples"):  # full cohorts stay local; tops are copied
            continue
        for fn in files:
            if not fn.endswith(INCLUDE) or fn.startswith("."):
                continue
            p = os.path.join(root, fn)
            rel = os.path.normpath(os.path.join(rel_root, fn))
            digest = _sha(p)
            if idx.get(rel) == digest:
                continue
            todo.append((p, rel, digest))
    for p, rel, digest in sorted(todo):
        repo_path = f"tasks/{task_id}/{rel}"
        api.upload_file(path_or_fileobj=p, path_in_repo=repo_path,
                        repo_id=REPO, repo_type="dataset",
                        commit_message=f"{task_id}: {rel}")
        idx[rel] = digest
        ops.append(repo_path)
    for p in extra_files or []:
        repo_path = f"tasks/{task_id}/{os.path.basename(p)}"
        api.upload_file(path_or_fileobj=p, path_in_repo=repo_path,
                        repo_id=REPO, repo_type="dataset",
                        commit_message=f"{task_id}: {os.path.basename(p)}")
        ops.append(repo_path)
    json.dump(idx, open(idx_path, "w"))
    return ops
