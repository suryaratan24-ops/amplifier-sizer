# SizeAgent — an EEsizer re-implementation with role-aware prompting and Monte-Carlo-in-the-loop

A clean, modular Python re-implementation of **EEsizer: LLM-Based AI Agent for Sizing of Analog and Mixed-Signal
Circuits** (C. Liu, D. Chitnis, arXiv:2509.25510, 2025), plus three extensions:

1. **Role-aware ("layered") context**: every device is tagged with its *function* (input pair, tail source,
   mirror, 2nd-stage driver/source, LDO pass device, compensation C/R, decap) inferred from the netlist graph,
   with its gm/ID operating point and gm/ID-LUT what-if hints.
2. **Monte-Carlo-in-the-loop guard-banding**: after nominal convergence, a Pelgrom-mismatch Monte Carlo is run;
   if the parametric yield is below target, specs are guard-banded (|nominal−mean| + kσ; random offset is
   constrained through an analytic Pelgrom 3σ metric) and the loop continues.
3. **LDO benchmark from Li & Chan Carusone, ICCAD 2023** (RGCN + RL LDO sizing), ported to PTM 180 nm, with the
   paper's spec set (PSRR bands at both loads, PM at min/max load, I_Q, dropout, load regulation, minimise C_dec).

ngspice runs in the loop; the LLM can be GPT/o-series, Gemini (free tier works), Claude, any OpenAI-compatible
endpoint (Ollama/vLLM), a human/copy-paste "manual" backend, or an **offline rule-based expert** so that the whole
pipeline runs without an API key.

![agent loop](docs/agent_loop.png)

---

## 1. What EEsizer does (intuition first)

Think of a junior designer sitting next to a simulator. They (1) read the spec, (2) figure out what the circuit is,
(3) decide which simulations prove each spec, then (4) loop: *simulate → look at what failed → recall how each knob
moved each metric last time → nudge 1–3 knobs → simulate again*. EEsizer gives exactly this loop to an LLM:

| EEsizer stage | What the LLM does | Here |
|---|---|---|
| Task decomposition | splits the request into type / node / simulation / sizing questions | `prompts.TASK_DECOMPOSITION` |
| Target extraction | turns "gain above 40 dB …" into numbers | `prompts.TARGET_EXTRACTION` + regex cross-check |
| Circuit identification | names the topology and sub-blocks | `prompts.TYPE_IDENTIFY` + automatic role inference |
| Tool selection | **function calling**: which of dc/ac/tran and which analyses | `tools.TOOL_SCHEMA` (one `universal_circuit_tool`) |
| ReAct loop | observe → analyse → optimise (CoT) → size | `prompts.ANALYSIS / OPTIMISATION / SIZING` |
| Algorithm 1 | stop when every spec is within 5 % **and** no device has Vov < 0 | `spec.pass_check`, `Agent.region_ok` |

The LLM never edits connectivity: it returns JSON actions (`{"element":"M1","param":"W","value":"4u"}`), the agent
clamps them to bounds, propagates them to matched devices, and re-simulates.

## 2. Install

```bash
pip install numpy scipy matplotlib          # Python >= 3.9, no LLM SDK needed (plain HTTPS)
# ngspice >= 38:  Ubuntu: sudo apt install ngspice   |  Windows: install ngspice and
#   set NGSPICE=C:\Spice64\bin\ngspice_con.exe          (the console binary)
python -m pytest -q tests                    # 8 fast tests
```

## 3. Run

```bash
# offline (no key) - knowledge-guided rule-based backend
python -m sizeagent --circuit circuits/ota5t  --backend heuristic --iters 40
python -m sizeagent --circuit circuits/miller --backend heuristic --iters 40 --yield-target 0.9

# real LLMs (EEsizer default: continue from the latest design, 25 iterations)
export GEMINI_API_KEY=...;    python -m sizeagent --circuit circuits/ota5t --backend gemini --model gemini-2.5-flash
export OPENAI_API_KEY=...;    python -m sizeagent --circuit circuits/miller --backend openai --model o3
export ANTHROPIC_API_KEY=...; python -m sizeagent --circuit circuits/ldo --backend anthropic --model claude-sonnet-5-5
python -m sizeagent --circuit circuits/ota5t --backend ollama --model qwen2.5:14b      # local
python -m sizeagent --circuit circuits/ota5t --backend manual --exchange-dir xchg --single-call  # paste into any chat UI

# options: --single-call (1 LLM call/iteration)  --no-roles  --no-lut  --nl-targets  --accept best|always  --tol 0.05

python bench.py --backend heuristic --seeds 5 --iters 40 --ablate-roles --yield-variant --mc-eval 50
python -m sizeagent.variation --circuit circuits/ota5t --netlist runs/<run>/final_dut.cir --n 200
```

**GUI (specs in → best topology + sizes out):** `python -m streamlit run gui.py`. Enter gain, UGBW, phase
margin, max current and load; a first-order hand check skips topologies that cannot work, the agent sizes the rest
with ngspice in the loop, and the page shows the best amplifier with its W/L. Online (Streamlit Community Cloud):
`packages.txt` installs ngspice and `requirements.txt` the Python packages.

Every run writes `runs/<name>/`: `transcript.md` (all prompts and replies), `history.csv`, `convergence.png`,
`final_dut.cir`, `final_state.txt`, `summary.json`.

## 4. Repository map

```
sizeagent/
  netlist.py    parse/edit/render the DUT .subckt; graph-based role inference ("layer" view)
  circuit.py    circuit card: ports, test conditions, knobs + bounds, matched groups, targets
  benches.py    test benches: open-loop (DC loop via 1 TH/1 TF), buffer, inverting; LDO dc/loop/psrr/tran
  measure.py    metric extraction: gain, UGBW, PM, BW, CMRR, PSRR, offset, ICMR, swing, THD, delay, freq ...
  tools.py      tool registry (EEsizer Table III + LDO tools), function-calling schema, evaluator
  spec.py       targets, 5 % tolerance check, normalised violation, NL target parser
  prompts.py    all prompt templates
  llm.py        OpenAI-compatible / Gemini / Anthropic / Ollama / manual back-ends, robust JSON extraction
  heuristic.py  offline expert: role-knowledge priors + Broyden-updated sensitivities + trust-region steps
  agent.py      the EEsizer pipeline, acceptance policy, soft-objective phase, Monte-Carlo gate
  gmid.py       gm/ID LUT generation with ngspice + lookup + prompt hints
  variation.py  Monte Carlo (Pelgrom A_VT, W/L sigma)
circuits/       ota5t, miller, ldo   (card.json + dut.cir)
models/         PTM 180/130/90 nm (as distributed with the EEsizer repo)
```

## 5. Results (PTM 180 nm)

| circuit | variant | converged | median iterations | median ngspice runs | best strict pass | median MC yield* |
|---|---|---|---|---|---|---|
| ota5t | roles | 5/5 | 14 | 30 | 7/7 | 46 % (min 0 %) |
| ota5t | no_roles | 5/5 | 19 | 40 | 7/7 | 4 % (min 0 %) |
| ota5t | roles+MC | 5/5 | 30 | 182 | 7/7 | 100 % (min 98 %) |
| miller | roles | 5/5 | 22 | 69 | 9/9 | 98 % (min 0 %) |
| miller | no_roles | 0/5 | - | 123 | 4/9 | 0 % (min 0 %) |
| miller | roles+MC | 5/5 | 22 | 129 | 9/9 | 98 % (min 98 %) |
| ldo | roles | 0/5 | - | 123 | 6/8 | - |
| ldo | no_roles | 0/5 | - | 123 | 6/8 | - |

*50-sample Monte Carlo of each final design against the strict specs (Pelgrom A_VT = 5 mV*um, sigma_W,L = 5 nm). Offline backend, 5 seeds, 40-iteration budget (+12 per failed yield check). PTM 180 nm.

* **Role-aware priors matter**: on the two-stage Miller op-amp (9 specs, 13 knobs) the offline backend converges in
  5/5 runs with role knowledge and 0/5 without; on the 5T OTA it cuts the median from 19 to 14 iterations.
* **LLM in the loop**: Claude (interactive, `--backend manual --single-call`) sized the 5T OTA from the default
  W = 2 µm / L = 0.5 µm start to all 7 specs in **4 iterations / 8 LLM calls** — prompts, replies and plots are in
  `results/llm_demo_ota5t/`.
* **Nominal convergence ≠ robust design**: EEsizer's stopping rule (every spec within 5 %) accepts designs on the
  spec edge — median Monte-Carlo yield 46 % for the OTA, and one Miller run at 0 %. The Monte-Carlo gate
  (`--yield-target 0.9`) guard-bands the targets and adds a Pelgrom 3σ-offset metric; all 10 gated runs end at
  ≥ 98 % yield. The analytic Pelgrom σ agrees with Monte Carlo (0.29 mV vs 0.30 mV on the OTA).
* **ICCAD'23 LDO** (8 specs, 13 knobs, PTM port): 6/8 specs in 40 iterations (strict in 4/5 seeds). The two
  misses — PSRR above 1 MHz at 10 mA (−0.8 … −2.0 dB vs −5 dB) and load regulation (39–41 mV vs 36 mV) — are the
  same two the paper's RGCN-DDPG agent missed for LDO1 after 10 000 SKY130 simulations (−0.32 dB, 40 mV).
  Different PDK, so treat this as indicative.

Figures: `results/bench_heuristic/convergence.png` (best violation vs iteration), `results/ptm180_gmid_lut.png`,
per-run `convergence.png`. Monte-Carlo reports: `results/mc_*.md`.

## 6. Ideas this opens up (next steps)

* **Spectre / XT018 backend**: swap `spice.run_ngspice` for a Spectre/OCEAN runner; the rest is simulator-agnostic.
* **gm/ID action space**: let the LLM choose (gm/ID, L) per *role* and map to W with the LUT — the layered-LUT idea.
* **TIA benchmark**: add input-referred-noise and transimpedance-bandwidth tools (`.noise`) and a TIA card.
* **ML surrogates** (cf. the T-coil/ESD CNN paper): replace slow EM/PEX steps with learned models inside the loop.
* Compare against the RGCN-DDPG agent of ICCAD'23 on the same PDK and spec set.

## 7. Credits

EEsizer (arXiv:2509.25510, github.com/eelab-dev/EEsizer) for the method, prompts and PTM model files;
Li & Chan Carusone, ICCAD 2023 (github.com/ChrisZonghaoLi/sky130_ldo_rl) for the LDO benchmark.
