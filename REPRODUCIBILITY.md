# Reproducibility Guide (`EVSA v2.0.2`)

**Author**: Dev Parth  
**Role**: Independent Researcher  
**Research Area**: Independent Research in Artificial Intelligence  
**Release Version**: `v2.0.2` | **Release Date**: 9 November 2026  

---

## 1. Cross-Platform Execution Environment & Pinned Dependencies

All results in this repository are generated deterministically on CPU without external commercial LLM API dependencies and run out-of-the-box on **Linux**, **Windows**, and **macOS**:
- **Reference Execution Environment**: Linux `6.1.189+` (`x86_64`), virtualized `Intel(R) Xeon(R) Processor @ 2.60GHz` (4 vCPUs, `8.0 GiB` RAM), `Python 3.13.16`
- **Cross-Platform CI Matrix (`.github/workflows/ci.yml`)**: Automated verification on `ubuntu-latest` (`Python 3.11`, `3.12`, `3.13`) and `windows-latest` (`Python 3.13`), using pure-Python/stdlib `detect_cpu_model()` (`experiments/run_complete_suite.py`) and explicit `encoding="utf-8"` file I/O
- **Pinned Scientific Stack (`requirements.txt`)**: `numpy==2.3.5`, `scipy==1.17.1`, `pandas==2.2.3`, `matplotlib==3.11.2`, `pytest==9.0.3`, `python-docx==1.1.2`
- **LaTeX Engine (Optional for PDF Recompilation)**: `Tectonic 0.17.0` (self-contained XeTeX-compatible compiler available on Linux, Windows, and macOS)

---

## 2. Deterministic Seed Policy & Hard Oracle Boundary

Every experiment isolates stochasticity across independent NumPy `PCG64` (`np.random.default_rng`) streams derived deterministically from base seed $s$:
1. **Training Stream Seed**: `s`
2. **Held-Out Evaluation Probe Seed**: `s + 10000` (and `s + 20000` / `s + 55000` for distinct evaluation splits)
3. **Agent Internal Reasoning & Exploration Seed**: `s + k` (offset per experiment protocol)

- **Primary Multi-Seed Suite (`Experiments 1–6` & `Execution-Channel Parity`)**: $N = 20$ seeds `s in {1001, 1002, ..., 1020}`.
- **Extended Seed-Horizon Stability Check (`Experiment 1`)**: $N = 30$ seeds `s in {1001, 1002, ..., 1030}`.
- **Hard Capability Isolation**: Every episode runs through `run_audited_episode` (`src/evaluation/audit.py`), passing a slot-restricted `AgentEnvironmentInterface` (`src/benchmark/agent_interface.py`) exposing only `evaluate_plan(task, executed_plan) -> RolloutResult`.

All behavioral accuracy, contamination (`FDP`), `FAR`/`FPR`/`FRR`, post-shift recovery, and direct per-family / Global-74 Holm-adjusted $p$-values are **100% bit-for-bit deterministic** across runs on identical NumPy/SciPy builds.

---

## 3. Step-by-Step Reproduction Commands (Linux, Windows PowerShell/CMD, or macOS)

From a clean clone of the repository root:

### Step 1: Install Pinned Dependencies
```bash
python -m pip install -r requirements.txt
```

### Step 2: Run the 16-Test Cross-Platform Scientific Integrity Suite (`~2.5 seconds`)
```bash
pytest -v tests/test_evsa_suite.py
```
Verifies all 16 automated checks:
1. `test_task_rollout_and_agent_outcome_have_zero_oracle_fields`: `TaskInstance`, `RolloutResult`, and `AgentEpisodeOutcome` (`src/benchmark/types.py`) expose zero oracle fields.
2. `test_hard_interface_boundary_separates_agent_from_oracle`: `AgentEnvironmentInterface` (`src/benchmark/agent_interface.py`) has no `__dict__` and every agent class in `src/agents/baselines.py` and `src/agents/evsa_agent.py` raises `TypeError` if passed a raw `NonStationaryTaskEnvironment`.
3. `test_zero_oracle_leakage_across_all_agent_and_backbone_modules`: Source inspection across `src/benchmark/agent_interface.py`, `src/benchmark/llm_backbone.py`, `src/validation/confidence.py`, `src/validation/sprt_validator.py`, `src/drift_detection/cusum_detector.py`, `src/retrieval/retriever.py`, `src/memory/repository.py`, `src/memory/hypothesis_generator.py`, `src/memory/memory_manager.py`, `src/agents/experimentation.py`, `src/agents/baselines.py`, and `src/agents/evsa_agent.py`.
4. `test_backbone_does_not_know_latent_xor_parity`: `CalibratedSymbolicLLMBackbone` (`src/benchmark/llm_backbone.py`) uses linear inclusive-OR heuristic on Step 3 rather than latent XOR parity.
5. `test_environment_determinism_and_regime_invariance`: Deterministic task sampling and stationary control domain invariance in `src/benchmark/environment.py`.
6. `test_confidence_estimator_bayesian_vs_empirical_ablation`: `ConfidenceEstimator` (`src/validation/confidence.py`) Beta quantile credible bounds vs. exact empirical frequency when `enabled=False`.
7. `test_discounted_sprt_promotion_and_quarantine_gates`: `EvidenceValidationEngine` (`src/validation/sprt_validator.py`) 3-trial promotion and negative-bound quarantine.
8. `test_safe_lgg_generalization_and_revalidation`: Safe LGG broadening in `src/memory/memory_manager.py` and `src/memory/repository.py` requiring confirming re-verification.
9. `test_retriever_tau_rel_gate_and_frozen_eval_validated_only`: `StrategyRetriever` (`src/retrieval/retriever.py`) `tau_rel = 0.45` gate and `ExperimentationEngine` (`src/agents/experimentation.py`) `VALIDATED`-only frozen evaluation (`learn=False`).
10. `test_cusum_cooldown_and_selective_shift_quarantine`: `DistributionShiftDetector` (`src/drift_detection/cusum_detector.py`) cooldown/warmup and selective domain quarantine in `src/memory/memory_manager.py`.
11. `test_budget_parity_and_contamination_audit`: Rollout budget parity (`B = 2`) and `audit_evsa_repository` (`src/evaluation/audit.py`) accounting.
12. `test_statistical_power_and_holm_bonferroni`: Exact non-central-$t$ MDES (`dz = 0.660` at $N=20, 80\%$ power) and direct per-family primary Holm adjustment in `src/evaluation/metrics_and_stats.py`.
13. `test_reproducible_global74_holm_ledger`: Bit-for-bit verification that `primary_pvalue_holm_global74` equals a single direct Holm pass over all 74 selected `primary_pvalue_raw` values (alongside separate `sensitivity_t_pvalue_holm_global74` and `sensitivity_wilcoxon_pvalue_holm_global74` passes) across `experiments/statistical_analysis/paired_hypothesis_tests_holm.csv` and `tables/table_statistical_tests_all.csv`.
14. `test_documentation_references_exact_source_symbols`: Verifies strictly 7 authoritative `docs/PHASE_*.md` files (zero duplicate outlines) and checks every `src/...` path and `test_...` symbol cited in `FINAL_AUDIT_REPORT.md` and `REPRODUCIBILITY.md`.
15. `test_independent_researcher_attribution_zero_college_mentions`: Scans repository docs, configs, LaTeX, and source files to guarantee consistent independent attribution and zero university/student mentions.
16. `test_cross_platform_cpu_detection_and_utf8_safety`: Verifies `detect_cpu_model()` across mocked Windows (`win32`), Linux, and macOS platforms with zero `lscpu` or `shell=True` calls.

### Step 3: Re-Run All Experiments, Sensitivity Controls & 74 Statistical Tests (`~3–4 minutes`)
```bash
python experiments/run_complete_suite.py
```

### Step 4: Regenerate All 10 Publication Figures (`PNG` + `PDF`, `~9 seconds`)
```bash
python experiments/generate_all_figures.py
```

### Step 5: Compile the 17-Section Manuscript (`PDF` + `DOCX`)
```bash
cd paper && tectonic -X compile research_paper.tex && cd ..
python paper/build_docx.py
```

### Step 6: Post-Push Remote Tag & GitHub Actions Verification
After pushing `main` and tag `v2.0.2` to GitHub (`git push origin main && git push origin v2.0.2`), verify live remote tag and CI check-runs via:
```bash
python experiments/verify_github_release.py --tag v2.0.2
```
