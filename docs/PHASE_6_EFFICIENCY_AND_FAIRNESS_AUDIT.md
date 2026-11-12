# Phase 6: Computational Efficiency Telemetry & Execution-Channel Fairness Audit (`v2.0.2`)

**Project**: *Evidence-Validated Strategy Adaptation for Memory-Augmented Agents Under Distribution Shift*  
**Author**: Dev Parth — Independent Researcher (*Independent Research in Artificial Intelligence*)  
**Release Version**: `v2.0.2` | **Date**: 9 November 2026  

---

## 1. Controlled $2 \times 2$ Execution-Channel Fairness Parity Audit

In default mode, EVSA invokes `VALIDATED` workflows as compiled deterministic operator tuples (`direct_macro`), whereas Baselines B–D condition `CalibratedSymbolicLLMBackbone.execute_guided_plan` (`src/benchmark/llm_backbone.py`) on retrieved exemplars (`stochastic_prompt`, weighted step adherence $\le 0.96$). To decouple epistemic memory selection from deterministic procedural execution, `experiments/run_complete_suite.py` evaluates all four cells of a controlled $2 \times 2$ design across $N = 20$ seeds (`experiments/processed/fairness_parity_summary.csv`):

| Controlled Execution Variant | Overall Held-Out SR (`%`) | Unseen Instances SR (`%`) | Zero-Shot Comp. $c=7$ (`%`) |
| :--- | :---: | :---: | :---: |
| `EVSA (Direct Macro Exec)` | **`54.50 ± 5.04`** | **`66.19 ± 7.11`** | **`31.13 ± 7.67`** |
| `EVSA (Stochastic Prompt Exec)` | **`49.62 ± 4.98`** | `58.78 ± 5.90` | **`31.31 ± 10.21`** |
| `Replay (Direct Macro Exec)` | `31.48 ± 2.45` | `41.97 ± 3.64` | `10.50 ± 3.59` |
| `Replay (Stochastic Prompt Exec)` | `27.58 ± 1.74` | `35.31 ± 2.96` | `12.12 ± 3.32` |

**Controlled Two-Factor Attribution**:
- Under identical stochastic prompt execution, EVSA outperforms Experience Replay by **`+22.04 pp`** ($\sim 82\%$ of the `+26.92 pp` nominal gap; $d_z = 4.09, p_{\mathrm{Holm}} = 1.57 \times 10^{-13}$).
- Under identical direct macro execution, EVSA outperforms Experience Replay by **`+23.02 pp`** ($d_z = 4.47, p_{\mathrm{Holm}} = 6.41 \times 10^{-14}$), and direct macro execution *degrades* Experience Replay on zero-shot $c=7$ (`12.12% -> 10.50%`) because rigid adherence amplifies surface-colliding plans.

---

## 2. Separation of Measured CPU/Rollout Telemetry vs. Logical Memory & Character-Based Token Projections (`Experiment 6`)

`experiments/processed/exp6_efficiency_summary.csv` distinguishes directly measured local runtime/rollout counts from fixed-record logical memory estimates and character-based LLM token cost projections (`$0.00` commercial API spend):
1. **Directly Measured Local CPU Latency & Rollout Counts (`480` Tasks/Seed)**:
   - Sub-millisecond Python/NumPy bookkeeping latency measured via `time.perf_counter()`: **`0.621 ± 0.065 ms/task`** (`403.45 ms` across 480 tasks on virtualized Xeon @ 2.60 GHz), negligible (`< 0.35%`) relative to real tool/LLM inference latencies (`200–2,000 ms`).
   - Directly counted environment rollouts across 480 tasks: **`590.45 ± 12.30`** for EVSA vs. `645.30–651.35` across baselines (saving `~55–61` rollouts/seed by solving adapted tasks on trial 1).
2. **Analytical Fixed-Record Logical Memory Estimate (`memory_bytes`)**:
   - Evaluated via deterministic per-record byte constants (`264 B` per active `StrategyRecord` + up to `128 * 48 B` capped tombstones in `StrategyRepository.memory_bytes()`, vs. `272 B` per baseline buffer entry—representing serialized logical storage rather than Python interpreter object/heap RSS): **`13.14 ± 1.92 KB`** for EVSA (`48.55` total strategy records across tiers), i.e., **`8.1x` smaller** than `Conventional Memory` (`106.09 KB`) and **`2.3x` smaller** than `Experience Replay` (`29.87 KB`).
3. **Character-Based Projected Token & USD Ledger (`TokenUsageLedger`)**:
   - Computed from serialized prompt/completion character lengths (`len(text) / 3.8` + framing constants, rather than a model-specific BPE tokenizer or live API billing) at `$2.50 / $10.00` per $10^6$ input/output tokens: EVSA projects **`46,758` total tokens** (**`$0.213`**, lowest projected cost among all five architectures) and **`0.117` SR points per `10k` projected tokens** (`1.83x` `Static` and `4.68x` `Reflection`).
