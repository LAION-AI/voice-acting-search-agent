#!/usr/bin/env python3
"""Assemble results/benchmark_*/RESULTS.md from a benchmark run's artifacts.

    python scripts/make_benchmark_report.py runs/benchmark_arou_valn_expl_stry \
        results/benchmark_arou_valn_expl_stry
"""
import glob
import json
import os
import sys


def main(src, dst):
    os.makedirs(dst, exist_ok=True)
    gens = []
    for line in open(os.path.join(src, "evolution_log.jsonl")):
        d = json.loads(line)
        rows = [r for r in d["results"] if "fitness_mean" in r]
        if not rows:
            continue
        best = max(rows, key=lambda r: r["fitness_mean"])
        gens.append({
            "n_genomes": len(d["genomes"]),
            "best": best["fitness_mean"],
            "mean": round(sum(r["fitness_mean"] for r in rows) / len(rows), 3),
            "best_means": best.get("means", {}),
            "best_genome": d["genomes"][best["idx"]],
        })
    lines = ["# Benchmark: maximize AROU + VALN + EXPL + S_STRY",
             "",
             "Constraints: GENU >= no-LoRA baseline, BLEND >= no-LoRA baseline "
             "(penalty-weighted). Fitness = mean of the four target slots "
             "(mean-of-8 samples per genome) minus constraint penalties.",
             "",
             "## Fitness per generation",
             "",
             "| gen | genomes | best fitness | mean fitness |",
             "|---|---|---|---|"]
    for i, g in enumerate(gens):
        lines.append(f"| {i} | {g['n_genomes']} | {g['best']:.3f} | {g['mean']:.3f} |")
    if gens:
        overall = max(range(len(gens)), key=lambda i: gens[i]["best"])
        bg = gens[overall]
        lines += ["", f"## Best genome (generation {overall}, fitness {bg['best']:.3f})",
                  "", "```json", json.dumps(bg["best_genome"], indent=1), "```",
                  "", "Per-code means of the best genome:",
                  "", "```json", json.dumps(bg["best_means"], indent=1), "```"]
    base = os.path.join(src, "baseline.json")
    if os.path.exists(base):
        b = json.load(open(base))
        lines += ["", "## No-LoRA baseline (constraint reference)",
                  "", "| code | value |", "|---|---|"]
        for c in ("AROU", "VALN", "EXPL", "S_STRY", "GENU", "BLEND", "QUALITY"):
            if c in b:
                lines.append(f"| {c} | {b[c]} |")
    rep = os.path.join(src, "report.md")
    if os.path.exists(rep):
        lines += ["", "## Agent's own final report", ""]
        lines += open(rep).read().splitlines()
    mem = os.path.join(src, "memory.md")
    if os.path.exists(mem):
        lines += ["", "## Agent memory (search notes)", "", "```"]
        lines += open(mem).read().splitlines()
        lines += ["```"]
    hof = sorted(glob.glob(os.path.join(src, "hall_of_fame", "*.json")))
    if hof:
        lines += ["", "## Hall of fame", ""]
        for h in hof:
            d = json.load(open(h))
            d.pop("vec99", None)
            lines += [f"### {os.path.basename(h)[:-5]}", "```json",
                      json.dumps(d, indent=1, default=str), "```", ""]
    with open(os.path.join(dst, "RESULTS.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[report] {dst}/RESULTS.md  ({len(gens)} generations)")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
