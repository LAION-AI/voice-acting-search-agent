#!/usr/bin/env python3
"""S5 seeder: pick 8 acting edge-cases (one per taxonomy family; the largest family
gets 2) from edge_cases.json and write registry task seeds per SWARM_PLAN §2
(registry/tasks/T00NN/TASK.md front matter + mission.txt). First real use of the
registry.

Selection: within a family prefer cases whose detector_tokens map onto dimensions the
99-vec scorer actually covers (VoiceNet dims / burst names) — "rank_by says coverable".
Alternate with/without reference voice (/tmp/evc_refs/ref{0..5}.wav). Burst-dominated
cases run without the (1-WER) multiplier (like fear_scream).

  python scripts/make_swarm_tasks.py [--edge-json /tmp/edge_cases.json] [--n 8]
"""
import argparse
import datetime
import json
import os
import re

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

VN_DIMS = {"RESP", "CHNK", "TENS", "VOLT", "AROU", "VALN", "DARC", "ARSH", "ATCK",
           "CLRT", "ROUG", "RANG", "PACE", "PAUS", "S_WHIS", "S_DRAM", "S_STRY",
           "S_NARR", "S_FORM", "EXPL", "PTCH", "TMPO"}
BURST_HINT = re.compile(r"scream|sob|laugh|cry|grunt|gasp|moan|wail|shriek|pant|burst",
                        re.I)


def coverability(case):
    toks = " ".join(case.get("detector_tokens", []))
    score = sum(1 for d in VN_DIMS if d in toks)
    score += toks.count("(burst)")
    if "Genuineness" in case.get("rank_by", "") or "Blend" in case.get("rank_by", ""):
        score += 2
    return score


def burst_dominated(case):
    toks = " ".join(case.get("detector_tokens", []))
    return (toks.count("(burst)") >= 2
            or bool(BURST_HINT.search(case.get("acoustic_signature", ""))
                    and "burst" in toks.lower()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--edge-json", default="/tmp/edge_cases.json")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--start-id", type=int, default=1)
    args = ap.parse_args()
    cases = json.load(open(args.edge_json))
    fams = {}
    for c in cases:
        fams.setdefault(c["category"], []).append(c)
    for f in fams:
        fams[f].sort(key=lambda c: -coverability(c))
    # one per family; the largest family contributes the extras
    order = sorted(fams, key=lambda f: -len(fams[f]))
    picks = [fams[f][0] for f in order]
    extra_i = 1
    while len(picks) < args.n:
        picks.append(fams[order[0]][extra_i])
        extra_i += 1
    picks = picks[: args.n]

    now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    tasks_dir = os.path.join(HERE, "registry", "tasks")
    os.makedirs(os.path.join(HERE, "registry", "claims"), exist_ok=True)
    manifest = []
    for i, c in enumerate(picks):
        tid = f"T{args.start_id + i:04d}"
        use_ref = i % 2 == 1
        ref = f"/tmp/evc_refs/ref{i % 6}.wav" if use_ref else None
        wer_on = not burst_dominated(c)
        tdir = os.path.join(tasks_dir, tid)
        os.makedirs(tdir, exist_ok=True)
        mission = (
            f"ACTING EDGE-CASE '{c['title']}' ({c['category']}): produce audio that "
            f"sounds like this coach describes: \"{c['coach']}\" Target acoustic "
            f"signature: {c['acoustic_signature']} "
            + (f"Clone the reference voice at {ref} (load_reference first) and keep "
               f"speaker similarity plausible. " if ref else
               "No reference voice — design a fitting voice from scratch. ")
            + "Evolution protocol: 6 generations x 8 genomes, mean-of-8. Fitness: "
            + ("DEFAULT v2 reward with the (1-WER) multiplier ON (intelligible speech "
               "carries this scene). " if wer_on else
               "this scene is dominated by NON-SPEECH vocalization: set "
               "wer_multiplier:false and lean on GENU+BLEND for the bursts. ")
            + "Choose maximize targets yourself from the detector tokens: "
            + ", ".join(c.get("detector_tokens", [])) + ". "
            "SUPERVISED (mandatory, enforced): after EVERY run_generation interpret the "
            "cohort, then call supervisor_review with your interpretation + top-2 "
            "sample_ids; realize the returned sonic directives next generation. "
            "Finish with the winning recipe + trajectory; save_best top 3.")
        front = "\n".join([
            "---",
            f"task_id: {tid}",
            f"title: {json.dumps(c['title'])}",
            f"tags: [{c['category'].split(' ')[0].lower().rstrip(',')}, edge_case"
            + (", with_reference" if ref else ", no_reference")
            + (", wer_off" if not wer_on else "") + "]",
            "status: OPEN",
            "owner: null",
            f"heartbeat: {now}",
            "budget: {generations: 6, supervisor_rounds: 6}",
            "operationalization:",
            "  maximize: agent-chosen from detector tokens",
            "  constraints: [{code: GENU, min: baseline}, {code: BLEND, min: baseline}]",
            f"edge_case_id: {c['id']}",
            f"reference: {ref or 'null'}",
            f"wer_multiplier: {str(wer_on).lower()}",
            "resume_hint: null",
            "---", "",
            f"# {c['title']}", "",
            f"Family: {c['category']}", "",
            f"Coach: {c['coach']}", "",
            f"Acoustic signature: {c['acoustic_signature']}", "",
            f"Rank by: {c['rank_by']}", "",
        ])
        open(os.path.join(tdir, "TASK.md"), "w").write(front)
        open(os.path.join(tdir, "mission.txt"), "w").write(mission)
        manifest.append({"task_id": tid, "edge_case_id": c["id"], "title": c["title"],
                         "family": c["category"], "reference": ref,
                         "wer_multiplier": wer_on,
                         "coverability": coverability(c)})
        print(f"{tid}  [{c['category'][:28]:28s}] ref={bool(ref)} wer={wer_on}  "
              f"{c['title']}")
    with open(os.path.join(tasks_dir, "seed_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    print(f"[seed] {len(manifest)} tasks -> {tasks_dir}")


if __name__ == "__main__":
    main()
