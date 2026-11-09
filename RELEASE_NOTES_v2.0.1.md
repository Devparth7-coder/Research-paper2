# Release Notes — EVSA `v2.0.1` (Scientific & Reproducibility Correction Release)

**Repository**: [https://github.com/Devparth7-coder/EVSA-Research-Paper](https://github.com/Devparth7-coder/EVSA-Research-Paper)  
**Author**: Dev Parth (Department of Computer Science and Engineering, MMMUT, Gorakhpur, Uttar Pradesh, India)  
**Tag**: `v2.0.1` | **License**: MIT  

---

## Summary of `v2.0.1`

`v2.0.1` is a comprehensive scientific, interface-isolation, statistical, bibliographic, and reproducibility correction release of **Evidence-Validated Strategy Adaptation (`EVSA`)** and **`ShiftBench-Sim v1.0`**, built directly on `main` (`4dd4c95`) and regenerated end-to-end from a clean clone.

### Key Scientific & Engineering Corrections in `v2.0.1`

1. **Reproducible Global-74 Holm-Bonferroni Ledger (`src/evaluation/metrics_and_stats.py`, `experiments/run_complete_suite.py`)**:
   - Added `apply_global_holm_bonferroni` directly into the statistical pipeline so `experiments/statistical_analysis/paired_hypothesis_tests_holm.csv`, `tables/table_statistical_tests_all.csv`, and `statistical_digest.json` deterministically export `t_pvalue_holm_global74`, `wilcoxon_pvalue_holm_global74`, `primary_pvalue_holm_global74`, and `significant_global74_at_05` (`62/74` per-family significant, `60/74` simultaneous global-74 significant; verified to `< 1e-15` by `test_reproducible_global74_holm_ledger`).
2. **Hard Capability Boundary for Oracle Isolation (`src/benchmark/agent_interface.py`, `src/evaluation/audit.py`)**:
   - Separated evaluator-only oracle state (`NonStationaryTaskEnvironment` + `run_audited_episode`) from all five agent implementations (`act_on_episode(..., env: AgentEnvironmentInterface) -> AgentEpisodeOutcome`) via a slot-restricted capability wrapper exposing solely `evaluate_plan(task, executed_plan) -> RolloutResult` (verified by `test_hard_interface_boundary_separates_agent_from_oracle`).
3. **Exact Source-Symbol Documentation Synchronization (`FINAL_AUDIT_REPORT.md`, `REPRODUCIBILITY.md`, `docs/PHASE_1..7_*.md`)**:
   - Rewrote `FINAL_AUDIT_REPORT.md` using exact `v2.0.1` file paths, class/method names, and test functions, added all seven phase dossiers in `docs/`, and added `test_documentation_references_exact_source_symbols` to prevent documentation drift.
4. **Verified Bibliographic & Related-Work Corrections (`paper/references.bib`, `paper/research_paper.tex`)**:
   - Corrected **AdaMEM** (`zhang2026adamem`, *AdaMEM: Test-Time Adaptive Memory for Language Agents*, ICML 2026; hybrid long-term trajectory + short-term synthesized strategy memory + `STEP-MFT`), **Dynamic Cheatsheet** (`suzgun2026dynamic`, EACL 2026 Long Papers, pp. 7080–7106, `doi:10.18653/v1/2026.eacl-long.333`), and **ReasoningBank** (`ouyang2026reasoningbank`, ICLR 2026, complete 17-author list including Chen-Yu Lee; self-judged trajectory distillation + `MaTTS`).
5. **Manuscript Scope & Claim Calibration (`paper/research_paper.{tex,pdf,docx}`)**:
   - Corrected section count references (`17` numbered sections across `21` typeset pages), refined causal language to controlled two-factor attribution and task-operator alignment, scoped retention claims precisely to unshifted stationary control domains (`82.21%`) vs. $\gamma_{\mathrm{shift}}=0.35$ discounting of unfalsified rules in alarmed shifted domains, and foregrounded the low-dimensional synthetic symbolic-procedural scope of `ShiftBench-Sim v1.0`.
6. **Pinned Dependencies, Tracked `.gitignore` & GitHub Actions CI**:
   - Pinned exact tested versions in `requirements.txt`, tracked `.gitignore`, and added `.github/workflows/ci.yml` (Python `3.11`, `3.12`, `3.13`).

---

## Canonical Repository Artifacts

- **Repository Root**: `https://github.com/Devparth7-coder/EVSA-Research-Paper`
- **LaTeX Manuscript**: `https://github.com/Devparth7-coder/EVSA-Research-Paper/blob/main/paper/research_paper.tex`
- **Raw Seed Logs**: `https://github.com/Devparth7-coder/EVSA-Research-Paper/tree/main/experiments/raw`
- **Statistical Ledger**: `https://github.com/Devparth7-coder/EVSA-Research-Paper/tree/main/experiments/statistical_analysis`
- **Source Package**: `https://github.com/Devparth7-coder/EVSA-Research-Paper/tree/main/src`
- **Test Suite**: `https://github.com/Devparth7-coder/EVSA-Research-Paper/tree/main/tests`
