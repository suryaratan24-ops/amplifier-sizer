"""Spec-to-amplifier GUI: type the specs, the agent sizes every candidate topology with ngspice
in the loop, and the page reports the best topology with its transistor sizes.

Run from this folder:   python -m streamlit run gui.py
"""
from __future__ import annotations

import math
import os
import re
import sys
import time
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
# fall back to the README's default Windows install if NGSPICE is not set in this shell
_DEFAULT_NGSPICE = Path(r"C:\Spice64\bin\ngspice_con.exe")
if "NGSPICE" not in os.environ and _DEFAULT_NGSPICE.exists():
    os.environ["NGSPICE"] = str(_DEFAULT_NGSPICE)

from sizeagent.agent import Agent, AgentConfig  # noqa: E402
from sizeagent.circuit import Circuit  # noqa: E402
from sizeagent.gmid import generate_lut, load_lut, save_lut  # noqa: E402
from sizeagent.llm import make_backend  # noqa: E402
from sizeagent.netlist import ROLE_TEXT, Netlist  # noqa: E402
from sizeagent.spec import pass_check, violation  # noqa: E402
from sizeagent.tools import METRICS  # noqa: E402
from sizeagent.units import eng, fmt_value  # noqa: E402

# Candidate topologies: circuits/<folder> -> (display name, number of gain stages).
# To add an amplifier, create circuits/<folder>/{dut.cir, card.json} and add one line here.
TOPOLOGIES = {
    "ota5t": ("5-transistor OTA (single stage)", 1),
    "miller": ("Two-stage Miller op-amp", 2),
}

# first-order hand-check constants
GM_ID_BEST = 25.0            # optimistic gm/ID (1/V), close to weak inversion
SINGLE_STAGE_MAX_DB = 50.0   # gain = gm*ro/2; long-L 180 nm devices top out around here
MILLER_CC_RATIO = 0.22       # Cc >= 0.22*CL for ~60 deg phase margin (Allen & Holberg)

# Warm start, used when a topology does not converge from the default sizes: size once with a
# helper spec set, then re-size for the user's exact specs starting from that design. An output-swing
# spec steers the rule-based designer toward a balanced second stage (on this project's Miller:
# converges in ~20 iterations instead of sticking at 3/6); if the user already gave one, the
# CORE specs alone are sized first.
GUIDE_SWING = 1.2   # V
CORE = ("dc_gain", "unity_gain_bandwidth", "phase_margin", "power")

LABELS = {
    "dc_gain": "DC gain",
    "unity_gain_bandwidth": "Unity-gain bandwidth",
    "phase_margin": "Phase margin",
    "power": "Power",
    "cmrr": "CMRR",
    "output_swing": "Output swing",
    "input_offset": "Input offset",
}


# ---------------------------------------------------------------- engineering helpers
def show(metric: str, value) -> str:
    unit = METRICS[metric][0]
    if value is None or not math.isfinite(value):
        return "n/a"
    if unit == "dB":
        return f"{value:.1f} dB"
    if unit == "deg":
        return f"{value:.1f}°"
    return eng(value, unit)


def build_targets(specs: dict, vdd: float) -> dict:
    t = {
        "dc_gain": [">=", specs["gain"]],
        "unity_gain_bandwidth": [">=", specs["ugbw"]],
        "phase_margin": [">=", specs["pm"]],
        "power": ["<=", vdd * specs["i_max"]],
    }
    if specs.get("cmrr") is not None:
        t["cmrr"] = [">=", specs["cmrr"]]
    if specs.get("swing") is not None:
        t["output_swing"] = [">=", specs["swing"]]
    if specs.get("offset") is not None:
        t["input_offset"] = ["<=", specs["offset"]]
    return t


def hand_check(stages: int, specs: dict) -> tuple[bool, str]:
    """First-order feasibility check, so a topology that clearly cannot work is not simulated."""
    ugbw, cl, i_max = specs["ugbw"], specs["cl"], specs["i_max"]
    if stages == 1:
        if specs["gain"] > SINGLE_STAGE_MAX_DB:
            return False, (f"gain {specs['gain']:g} dB is above the ~{SINGLE_STAGE_MAX_DB:g} dB "
                           f"one stage (gm·ro/2) can give")
        gm1 = 2 * math.pi * ugbw * cl                     # UGBW = gm1 / (2π·CL)
        i_min = 2 * gm1 / GM_ID_BEST
    else:
        cc = MILLER_CC_RATIO * cl
        gm1 = 2 * math.pi * ugbw * cc                     # UGBW = gm1 / (2π·Cc)
        gm6 = 2.2 * gm1 * cl / cc                         # 2nd pole gm6/CL > 2.2·UGBW for 60° PM
        i_min = (2 * gm1 + gm6) / GM_ID_BEST
    text = f"needs gm1 ≈ {eng(gm1, 'S')}, so at least ≈ {eng(i_min, 'A')} supply current"
    if i_min > i_max:
        return False, text + f", more than your {eng(i_max, 'A')} limit"
    return True, text


class LiveAgent(Agent):
    """Agent that sends its progress lines to the page instead of the console."""

    def __init__(self, *args, on_line=None, **kw):
        super().__init__(*args, **kw)
        self.on_line = on_line

    def say(self, msg: str) -> None:
        if self.on_line:
            self.on_line(msg)


def guide_targets(targets: dict) -> dict:
    if "output_swing" not in targets:
        return {**targets, "output_swing": [">=", GUIDE_SWING]}
    return {m: t for m, t in targets.items() if m in CORE}


def size_topology(folder: str, specs: dict, iters: int, on_line=None, guide: bool = False,
                  start: Path | None = None) -> dict:
    """One agent run. guide sizes for the warm-start helper specs; start continues from an earlier netlist."""
    c = Circuit.load(ROOT / "circuits" / folder)
    if start:
        c.dut = Netlist(start.read_text())
    c.conditions["cl"] = specs["cl"]
    targets = build_targets(specs, c.vdd)
    c.targets = guide_targets(targets) if guide else targets
    c.targets_text = "Size this amplifier for: " + ", ".join(
        f"{LABELS[m]} {op} {show(m, v)}" for m, (op, v) in c.targets.items()) + f", with a {eng(specs['cl'], 'F')} load."
    lut_path = Path(c.model_path).with_suffix(f".gmid_{c.vdd:g}V.json")
    if lut_path.exists():
        lut = load_lut(lut_path)
    else:
        lut = generate_lut(c.model_path, c.vdd)
        save_lut(lut, lut_path)
    tag = "guide" if guide else ("warm" if start else "cold")
    out = ROOT / "runs" / f"gui_{folder}_{time.strftime('%Y%m%d-%H%M%S')}_{tag}"
    cfg = AgentConfig(max_iter=iters, accept="best", out_dir=str(out), verbose=False)
    s = LiveAgent(c, make_backend("heuristic", None, seed=0), cfg, lut, on_line=on_line).run()
    s["out_dir"] = str(out)
    s["roles"] = c.roles
    s["cost"] = sum(violation(m, s["final_metrics"].get(m, math.nan), op, t) for m, (op, t) in c.targets.items())
    return s


def pick_winner(runs: dict) -> tuple[str, str]:
    ok = [f for f, s in runs.items() if s["success"]]
    if ok:
        best = min(ok, key=lambda f: (runs[f]["final_metrics"].get("power", math.inf), TOPOLOGIES[f][1]))
        why = "meets every spec" + (" and uses the least power of those that do" if len(ok) > 1 else "")
        return best, why
    best = min(runs, key=lambda f: runs[f]["cost"])
    return best, "no topology met every spec; this one came closest"


def sizes_table(s: dict) -> list[dict]:
    net = Netlist((Path(s["out_dir"]) / "final_dut.cir").read_text())
    rows = []
    for e in net.elements:
        role = re.sub(r"\s*\([^)]*\)", "", ROLE_TEXT.get(s["roles"].get(e.name, ""), s["roles"].get(e.name, "")))
        if e.is_mos:
            rows.append({"Element": e.name, "Role": role, "Type": e.model.upper(),
                         "W": fmt_value(e.params["W"]) + "m", "L": fmt_value(e.params["L"]) + "m", "Value": ""})
        else:
            unit = {"R": "Ω", "C": "F", "V": "V"}.get(e.kind, "")
            rows.append({"Element": e.name, "Role": role, "Type": e.kind, "W": "", "L": "",
                         "Value": fmt_value(e.value) + unit})
    return rows


def spec_table(specs: dict, runs: dict) -> list[dict]:
    rows = []
    targets = next(iter(runs.values()))["targets"]
    for m, (op, t) in targets.items():
        row = {"Spec": LABELS[m], "Target": f"{'≥' if op == '>=' else '≤'} {show(m, t)}"}
        for f, s in runs.items():
            v = s["final_metrics"].get(m, math.nan)
            mark = "✅" if pass_check(v, op, t, 0.0) else ("⚠️" if pass_check(v, op, t, 0.05) else "❌")
            row[TOPOLOGIES[f][0]] = f"{mark} {show(m, v)}"
        rows.append(row)
    return rows


# ---------------------------------------------------------------- page
def spec_form() -> tuple[bool, dict, int, bool]:
    with st.form("specs"):
        st.subheader("1 · Your specifications")
        a, b, c = st.columns(3)
        gain = a.number_input("DC gain (dB) ≥", 0.0, 120.0, 60.0, 1.0)
        ugbw = b.number_input("Unity-gain bandwidth (MHz) ≥", 0.01, 500.0, 10.0, 0.5)
        pm = c.number_input("Phase margin (°) ≥", 0.0, 90.0, 60.0, 1.0)
        a, b, c = st.columns(3)
        i_max = a.number_input("Max supply current (µA) ≤", 0.1, 10000.0, 500.0, 10.0)
        cl = b.number_input("Load capacitance (pF)", 0.1, 100.0, 10.0, 0.5)
        c.number_input("Supply VDD (V)", value=1.8, disabled=True, help="Fixed by the PTM 180 nm models")

        st.markdown("**Optional specs** (tick to include)")
        a, b, c = st.columns(3)
        use_cmrr = a.checkbox("CMRR", value=True)
        cmrr = a.number_input("CMRR (dB) ≥", 0.0, 140.0, 70.0, 1.0)
        use_swing = b.checkbox("Output swing", value=True)
        swing = b.number_input("Output swing (V) ≥", 0.1, 1.8, 1.2, 0.05)
        use_off = c.checkbox("Input offset", value=True)
        offset = c.number_input("Input offset (mV) ≤", 0.01, 50.0, 2.0, 0.1)

        with st.expander("Search settings"):
            iters = st.slider("Max iterations per topology", 10, 80, 40)
            try_all = st.checkbox("Simulate every topology, even ones the hand check rules out")
        go = st.form_submit_button("Find best amplifier", type="primary")
    specs = {
        "gain": gain, "ugbw": ugbw * 1e6, "pm": pm, "i_max": i_max * 1e-6, "cl": cl * 1e-12,
        "cmrr": cmrr if use_cmrr else None,
        "swing": swing if use_swing else None,
        "offset": offset * 1e-3 if use_off else None,
    }
    return go, specs, iters, try_all


def run_search(specs: dict, iters: int, try_all: bool) -> dict:
    st.subheader("2 · Search")
    checks, runs = {}, {}
    for f, (label, stages) in TOPOLOGIES.items():
        ok, text = hand_check(stages, specs)
        checks[f] = (ok, text)
        st.markdown(f"{'🟢' if ok else '🔴'} **{label}**: {text}"
                    + ("" if ok else (" → simulating anyway" if try_all else " → skipped")))
    todo = [f for f in TOPOLOGIES if checks[f][0] or try_all]
    if not todo:
        st.error("The hand check rules out every topology. Relax the specs, or tick "
                 "'Simulate every topology' under Search settings to try anyway.")
        return {"specs": specs, "checks": checks, "runs": {}}

    for f in todo:
        label = TOPOLOGIES[f][0]
        with st.status(f"Sizing {label} with ngspice in the loop…", expanded=True) as box:
            lines: list[str] = []
            pane = st.empty()

            def on_line(msg, lines=lines, pane=pane):
                lines.append(msg)
                pane.code("\n".join(lines[-14:]), language=None)

            s = size_topology(f, specs, iters, on_line)
            if not s["success"]:
                helper = ", ".join(LABELS[m] for m in guide_targets(build_targets(specs, 1.8)))
                on_line(f"[warm start] stuck from the default sizes; sizing for {helper} first")
                g = size_topology(f, specs, iters, on_line, guide=True)
                on_line("[warm start] now your exact specs, starting from that design")
                warm = size_topology(f, specs, 2 * iters, on_line, start=Path(g["out_dir"]) / "final_dut.cir")
                warm["warm_start"] = True
                s = min((s, warm), key=lambda r: (not r["success"], r["cost"]))
            runs[f] = s
            n_pass = sum(s["final_pass"].values())
            how = " (with a warm start)" if s.get("warm_start") else ""
            if s["success"]:
                box.update(label=f"{label}: all specs met at iteration {s['success_iter']}{how} "
                                 f"({s['total_sims']} simulations)", state="complete", expanded=False)
            else:
                box.update(label=f"{label}: best design meets {n_pass}/{len(s['final_pass'])} specs "
                                 f"after {s['iterations']} iterations{how}", state="error", expanded=False)
    winner, why = pick_winner(runs)
    return {"specs": specs, "checks": checks, "runs": runs, "winner": winner, "why": why}


def show_result(res: dict) -> None:
    runs = res["runs"]
    if not runs:
        return
    st.subheader("3 · Result")
    w, s = res["winner"], runs[res["winner"]]
    (st.success if s["success"] else st.warning)(f"**Best topology: {TOPOLOGIES[w][0]}**: {res['why']}.")

    st.markdown("**Specs vs achieved**  (✅ met · ⚠️ within the 5 % EEsizer tolerance · ❌ missed)")
    st.dataframe(spec_table(res["specs"], runs), hide_index=True, width="stretch")

    st.markdown(f"**Sizes for the {TOPOLOGIES[w][0]}**")
    st.dataframe(sizes_table(s), hide_index=True, width="stretch")

    out = Path(s["out_dir"])
    netlist = (out / "final_dut.cir").read_text()
    a, b = st.columns([3, 2])
    with a:
        st.code(netlist, language=None)
    with b:
        st.download_button("Download sized netlist (.cir)", netlist, file_name=f"{w}_sized.cir")
        st.caption(f"All files for this run: `{out.relative_to(ROOT)}`")
    png = out / "convergence.png"
    if png.exists():
        st.image(str(png), caption="Each spec over the iterations (dashed line = target)")


def main() -> None:
    st.set_page_config(page_title="Amplifier Sizer", page_icon="📐", layout="wide")
    st.title("Amplifier Sizer")
    st.caption("Type the specs. The agent tries each amplifier topology, sizes it with ngspice in the loop, "
               "and shows the one that fits best.")
    go, specs, iters, try_all = spec_form()
    if go:
        st.session_state["result"] = run_search(specs, iters, try_all)
    if "result" in st.session_state:
        show_result(st.session_state["result"])


if __name__ == "__main__":
    main()
