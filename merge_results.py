"""Merge bench runs into results/summary.md (nominal variants + Monte-Carlo-gated variant)."""
import json
import statistics as st
from pathlib import Path

parts = [("results/bench_heuristic/bench.json", ("roles", "no_roles")),
         ("results/bench_heuristic_mc/bench.json", ("roles+MC",))]
rows = ["| circuit | variant | converged | median iterations | median ngspice runs | best strict pass | median MC yield* |",
        "|---|---|---|---|---|---|---|"]
data = {}
for path, keep in parts:
    for k, runs in json.loads(Path(path).read_text()).items():
        name, v = k.split("/")
        if v in keep:
            data[(name, v)] = runs
for name in ("ota5t", "miller", "ldo"):
    for v in ("roles", "no_roles", "roles+MC"):
        runs = data.get((name, v))
        if not runs:
            continue
        succ = [r for r in runs if r["success"]]
        its = [r["success_iter"] for r in succ]
        ys = [r.get("mc_yield_final") for r in runs if r.get("mc_yield_final") is not None]
        n = len(runs[0]["final_pass_strict"])
        y = f"{st.median(ys)*100:.0f} % (min {min(ys)*100:.0f} %)" if ys and name != "ldo" else "-"
        rows.append(f"| {name} | {v} | {len(succ)}/{len(runs)} | {st.median(its) if its else '-'} | "
                    f"{st.median([r['total_sims'] for r in runs]):.0f} | "
                    f"{max(sum(r['final_pass_strict'].values()) for r in runs)}/{n} | {y} |")
txt = "\n".join(rows) + ("\n\n*50-sample Monte Carlo of each final design against the strict specs "
                         "(Pelgrom A_VT = 5 mV*um, sigma_W,L = 5 nm). Offline backend, 5 seeds, 40-iteration budget "
                         "(+12 per failed yield check). PTM 180 nm.\n")
Path("results/summary.md").write_text(txt)
print(txt)
