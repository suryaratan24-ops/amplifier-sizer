"""Benchmark: run the agent on every circuit card for several seeds and summarise.

    python bench.py --backend heuristic --seeds 5 --iters 40
    python bench.py --backend gemini --model gemini-2.5-flash --seeds 3 --iters 25 --circuits ota5t
"""
from __future__ import annotations

import argparse
import json
import math
import statistics as st
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from sizeagent.agent import Agent, AgentConfig  # noqa: E402
from sizeagent.circuit import Circuit  # noqa: E402
from sizeagent.gmid import generate_lut, load_lut, save_lut  # noqa: E402
from sizeagent.llm import make_backend  # noqa: E402


def run_one(card, backend, model, seed, iters, out, use_roles=True, accept=None, yield_target=0.0, mc_eval=0):
    c = Circuit.load(card)
    lut_path = Path(c.model_path).with_suffix(f".gmid_{c.vdd:g}V.json")
    if lut_path.exists():
        lut = load_lut(lut_path)
    else:
        lut = generate_lut(c.model_path, c.vdd)
        save_lut(lut, lut_path)
    be = make_backend(backend, model, **({"seed": seed} if backend == "heuristic" else {}))
    cfg = AgentConfig(max_iter=iters, accept=accept or ("best" if backend == "heuristic" else "always"),
                      use_roles=use_roles, out_dir=str(out), verbose=False, yield_target=yield_target)
    ag = Agent(c, be, cfg, lut)
    s = ag.run()
    if mc_eval:
        from sizeagent.netlist import Netlist
        from sizeagent.variation import monte_carlo
        c2 = Circuit.load(card)
        mc = monte_carlo(c2, Netlist((Path(out) / "final_dut.cir").read_text()), n=mc_eval, seed=99, tol=0.0)
        s["mc_yield_final"] = mc["yield_all"]
    s["cost_trace"] = [h.cost for h in ag.history]
    s["best_cost_trace"] = [min(h.cost for h in ag.history[: i + 1]) for i in range(len(ag.history))]
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="heuristic")
    ap.add_argument("--model", default=None)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--iters", type=int, default=40)
    ap.add_argument("--circuits", nargs="*", default=["ota5t", "miller", "ldo"])
    ap.add_argument("--ablate-roles", action="store_true", help="also run without role priors")
    ap.add_argument("--yield-variant", action="store_true", help="also run with Monte-Carlo-in-the-loop (yield 0.9)")
    ap.add_argument("--mc-eval", type=int, default=0, help="Monte-Carlo samples to evaluate every final design")
    ap.add_argument("--only", nargs="*", default=None, help="run only these variants (roles, no_roles, roles+MC)")
    ap.add_argument("--out", default="results/bench")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    variants = [("roles", True, 0.0)] + ([("no_roles", False, 0.0)] if a.ablate_roles else []) \
        + ([("roles+MC", True, 0.9)] if a.yield_variant else [])
    if a.only:
        variants = [v for v in [("roles", True, 0.0), ("no_roles", False, 0.0), ("roles+MC", True, 0.9)] if v[0] in a.only]
    allres = {}
    for name in a.circuits:
        for vname, roles, yt in variants:
            runs = []
            for seed in range(a.seeds):
                s = run_one(f"circuits/{name}", a.backend, a.model, seed, a.iters,
                            out / f"{name}_{vname}_s{seed}", use_roles=roles, yield_target=yt, mc_eval=a.mc_eval)
                runs.append(s)
                print(f"{name:7s} {vname:8s} seed={seed} success={s['success']} iter={s['success_iter']} "
                      f"sims={s['total_sims']} pass={sum(s['final_pass'].values())}/{len(s['final_pass'])} "
                      f"mc_yield={s.get('mc_yield_final')}", flush=True)
            allres[f"{name}/{vname}"] = runs
    (out / "bench.json").write_text(json.dumps(allres, indent=1, default=str))

    # ------------------------------------------------------------- table
    rows = ["| circuit | variant | success | median iters to success | median ngspice runs | best strict pass | median MC yield of final design |",
            "|---|---|---|---|---|---|---|"]
    for key, runs in allres.items():
        name, v = key.split("/")
        succ = [r for r in runs if r["success"]]
        its = [r["success_iter"] for r in succ]
        sims = [r["total_sims"] for r in runs]
        bestp = max(sum(r["final_pass_strict"].values()) for r in runs)
        n = len(runs[0]["final_pass"])
        ys = [r.get("mc_yield_final") for r in runs if r.get("mc_yield_final") is not None]
        rows.append(f"| {name} | {v} | {len(succ)}/{len(runs)} | {st.median(its) if its else '-'} | "
                    f"{st.median(sims):.0f} | {bestp}/{n} | {f'{st.median(ys)*100:.0f} %' if ys else '-'} |")
    table = "\n".join(rows)
    (out / "summary.md").write_text(table + "\n")
    print(table)

    # ------------------------------------------------------------- figure
    fig, axes = plt.subplots(1, len(a.circuits), figsize=(4.3 * len(a.circuits), 3.2), squeeze=False)
    for ax, name in zip(axes[0], a.circuits):
        colors = {"roles": "#4C72B0", "no_roles": "#DD8452", "roles+MC": "#55A868"}
        labels = {"roles": "role-aware priors", "no_roles": "no role priors", "roles+MC": "role-aware + MC gate"}
        for vname, _, _ in variants:
            if vname == "roles+MC":
                continue  # cost is re-defined after guard-banding; omit from this plot
            runs = allres[f"{name}/{vname}"]
            for i, r in enumerate(runs):
                tr = r["best_cost_trace"]
                ax.plot(range(len(tr)), [max(x, 1e-3) for x in tr], color=colors[vname],
                        alpha=0.6, lw=1, label=labels[vname] if i == 0 else None)
        ax.set_yscale("log")
        ax.set_title(name, fontsize=10)
        ax.set_xlabel("iteration")
        ax.set_ylabel("best total violation")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "convergence.png", dpi=140)


if __name__ == "__main__":
    main()
