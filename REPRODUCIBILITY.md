# Reproducibility Guide (`EVSA v2.0.0`)

**Author**: Dev Parth  
**Affiliation**: Department of Computer Science and Engineering, Madan Mohan Malaviya University of Technology (MMMUT), Gorakhpur, Uttar Pradesh, India  
**Release Version**: `v2.0.0` | **Completion Date**: 7 November 2026  

---

## 1. Execution Environment & Hardware Specifications

All results in this repository were generated and verified in a deterministic CPU-only Linux environment without external commercial LLM API dependencies:
- **Operating System**: Linux `6.1.189+` (`x86_64`, `glibc 2.41`)
- **CPU**: Virtualized `Intel(R) Xeon(R) Processor @ 2.60GHz` (4 vCPUs)
- **Memory**: `8.0 GiB` RAM
- **Python**: `Python 3.13.16`
- **Core Scientific Stack**: `numpy`, `scipy`, `pandas`, `matplotlib`, `pytest`, `python-docx` (zero external network calls; Holm-Bonferroni correction and exact non-central-$t$ power analysis are implemented directly in `numpy`/`scipy` inside `src/evaluation/metrics_and_stats.py`)
- **LaTeX Engine**: `Tectonic 0.17.0` (self-contained XeTeX-compatible compiler)

---

## 2. Deterministic Seed Policy

Every experiment isolates stochasticity across independent NumPy `PCG64` (`np.random.default_rng`) streams derived deterministically from base seed $s$:
1. **Training Stream Seed**: `s`
2. **Held-Out Evaluation Probe Seed**: `s + 10000` (and `s + 20000` / `s + 55000` for distinct evaluation splits)
3. **Agent Internal Reasoning & Exploration Seed**: `s + k` (offset per experiment protocol)

- **Primary Multi-Seed Suite (`Experiments 1–6` & `Execution-Channel Parity`)**: $N = 20$ seeds `s in {1001, 1002, ..., 1020}`.
- **Extended Seed-Horizon Stability Check (`Experiment 1`)**: $N = 30$ seeds `s in {1001, 1002, ..., 1030}`.

All behavioral accuracy, contamination (`FDP`), `FAR`/`FPR`/`FRR`, and post-shift recovery metrics are **100% bit-for-bit deterministic** across runs on identical NumPy/SciPy builds. Sub-millisecond wall-clock CPU latency (`eval_time_ms_per_task`, `0.427 ± 0.049 ms/task` for EVSA) reflects local hypervisor scheduling while preserving cross-agent bookkeeping ratios.

---

## 3. Step-by-Step Reproduction Commands

From the repository root:

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Run the 11-Test Unit & Zero-Oracle-Leakage Suite (`~2.3 seconds`)
```bash
pytest -v tests/test_evsa_suite.py
```
Verifies:
1. `TaskInstance` and `RolloutResult` expose **zero** oracle ground-truth attributes (`optimal_plan`, `governing_archetype`, `true_regime`, `true_success`, `oracle_plan`).
2. Source-level inspection across all files in `src/agents/`, `src/benchmark/llm_backbone.py`, `src/memory/`, `src/retrieval/`, `src/validation/`, and `src/drift_detection/` confirms zero private `_optimal_plan` or oracle accesses.
3. Bit-for-bit benchmark reproducibility across identical seeds.
4. Bayesian Beta lower credible bound (`betaincinv`) monotonicity AND `w/o Confidence Estimation` returning the true unregularized empirical frequency (`0/1 -> 0.0`, `1/1 -> 1.0`, `2/3 -> 0.667`) rather than a hardcoded constant.
5. Discounted sequential heuristic (`SPRT`) promotion (`3/3` successes) vs. rejection (`[1, 0, 0]`).
6. Safe Least General Generalization (`Safe LGG`) requiring re-verification whenever an already-`VALIDATED` strategy broadens its preconditions (`|phi_new| < |phi_old|`).
7. Applicability retriever enforcing both `LCB >= tau_rel` (`0.45`) and strictly `VALIDATED`-only retrieval when `allow_candidates=False` (`learn=False`).
8. Per-domain residual CUSUM drift detection (`nu=0.12, h=1.45, cooldown=12`) and evidence-based selective quarantine preserving unfalsified strategies while leaving unshifted domains untouched.
9. Identical rollout budget parity (`B = 2`) across all 5 agents and `direct_macro` vs. `stochastic_prompt` execution parity toggle.
10. Experiment 4 decomposition of active validated strategies into clean optimal, creation-time false promotions, and corrupted LGG over-generalizations.
11. Exact non-central-$t$ statistical power (`MDES d_z = 0.660` at $N=20, 80\%$ power) and Holm-Bonferroni monotonicity.

### Step 3: Re-Run All Experiments, Sensitivity Controls & 74 Statistical Tests (`~2.6 minutes`)
```bash
python3 experiments/run_complete_suite.py
```
Outputs (`159.19s` verified runtime):
- Configuration files in `config/default_config.json` and `experiments/configs/*.json`
- Seed-level raw data in `experiments/raw/*.csv`
- Aggregated summaries (`Mean`, `SD`, `Median`, `IQR`, `95% CI`) in `experiments/processed/*.csv`, `tables/*.csv`, and `tables/summary_tables.md`
- Full paired parametric + Wilcoxon hypothesis test ledger (`74` tests, both per-family and simultaneous global-74 Holm-Bonferroni adjusted) in `experiments/statistical_analysis/paired_hypothesis_tests_holm.csv` and `statistical_digest.json`
- Execution manifest in `experiments/experiment_manifest.json`

### Step 4: Regenerate All 10 Publication Figures (`PNG` + `PDF`, `~9 seconds`)
```bash
python3 experiments/generate_all_figures.py
```
Outputs `fig01_system_architecture.{png,pdf}` through `fig10_exp6_efficiency.{png,pdf}` in `figures/`.

### Step 5: Compile the Manuscript (`PDF` + `DOCX`)
```bash
cd paper && tectonic -X compile research_paper.tex && cd ..
python3 paper/build_docx.py
```
Outputs `paper/research_paper.pdf` (`21` pages) and `paper/research_paper.docx`.
