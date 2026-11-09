# Phase 7: 17-Section Manuscript Compilation, CI Automation & Clean-Clone `v2.0.1` Release Audit

**Project**: *Evidence-Validated Strategy Adaptation for Memory-Augmented Agents Under Distribution Shift*  
**Author**: Dev Parth (Department of Computer Science and Engineering, MMMUT, Gorakhpur, Uttar Pradesh, India)  
**Release Version**: `v2.0.1` | **Date**: 7 November 2026  

---

## 1. Synchronized 17-Section Manuscript (`paper/research_paper.{tex,pdf,docx}`)

The publication manuscript (`paper/research_paper.tex`) comprises **17 numbered sections** across **21 typeset pages** (zero `Overfull \hbox` warnings, zero BibTeX warnings, 10 vector figures `fig01`–`fig10`, 9 booktabs tables, 1 pseudocode algorithm, and 30 verified BibTeX references in `paper/references.bib`):
1. `Introduction`
2. `Motivation and Problem Statement`
3. `Related Work`
4. `Research Questions and Hypotheses`
5. `Proposed EVSA Architecture`
6. `Mathematical Formulation`
7. `Benchmark Design: ShiftBench-Sim v1.0`
8. `Experimental Methodology`
9. `Baseline Implementations and Fairness Audit`
10. `Experimental Results`
11. `Statistical Analysis, Power, and Multiplicity Sensitivity`
12. `Ablation Studies`
13. `Computational Efficiency and Projected Token Economics`
14. `Discussion`
15. `Limitations and Threats to Validity`
16. `Conclusion and Future Work`
17. `Reproducibility, Data Availability, and Research Governance`

`paper/build_docx.py` programmatically parses `paper/research_paper.tex`, `experiments/statistical_analysis/statistical_digest.json`, `experiments/statistical_analysis/paired_hypothesis_tests_holm.csv`, and `paper/references.bib` to build `paper/research_paper.docx` in lockstep.

---

## 2. `v2.0.1` Correction Release Verification Checklist

1. **Reproducible Global-74 Holm Ledger (`Step 1`)**: `apply_global_holm_bonferroni` (`src/evaluation/metrics_and_stats.py`) integrated into `experiments/run_complete_suite.py`, verified by `test_reproducible_global74_holm_ledger`.
2. **Hard Capability Boundary (`Step 2`)**: `AgentEnvironmentInterface` (`src/benchmark/agent_interface.py`) + evaluator-only `run_audited_episode` (`src/evaluation/audit.py`), verified by `test_hard_interface_boundary_separates_agent_from_oracle` and `test_zero_oracle_leakage_across_all_agent_and_backbone_modules`.
3. **Exact Source-Symbol Audit Report (`Step 3`)**: `FINAL_AUDIT_REPORT.md` and `REPRODUCIBILITY.md` synchronized with exact `src/` paths and `tests/test_evsa_suite.py` test names, verified by `test_documentation_references_exact_source_symbols`.
4. **Bibliographic Accuracy (`Step 4`)**: Verified exact ICML 2026 title and hybrid memory + `STEP-MFT` mechanism for `zhang2026adamem`, EACL 2026 Long Papers (`pp. 7080–7106`, `doi:10.18653/v1/2026.eacl-long.333`) for `suzgun2026dynamic`, and complete 17-author ICLR 2026 list (including Chen-Yu Lee) + self-judged trajectory `MaTTS` description for `ouyang2026reasoningbank`.
5. **Manuscript Scope & Claim Calibration (`Step 5`)**: Accurate `17`-section count, controlled two-factor attribution phrasing, precise cross-domain stationary retention vs. shifted-domain $\gamma_{\mathrm{shift}}=0.35$ discounting phrasing, and explicit low-dimensional synthetic symbolic-procedural scope for `ShiftBench-Sim v1.0`.
6. **Phase Dossiers (`Step 6`)**: Complete `docs/PHASE_1_PROBLEM_AND_THEORY.md` through `docs/PHASE_7_MANUSCRIPT_AND_RELEASE_AUDIT.md`.
7. **Pinned Dependencies, `.gitignore` & GitHub Actions CI (`Step 7`)**: Exact versions pinned in `requirements.txt`, tracked `.gitignore`, and automated multi-Python test workflow in `.github/workflows/ci.yml`.
8. **Clean-Clone Full Regeneration & Tagging (`Step 8`)**: Full pipeline executed from a fresh `git clone` into an isolated directory, regenerating all CSVs, ledgers, figures, PDF, and DOCX, tagged `v2.0.1` on `main`.
