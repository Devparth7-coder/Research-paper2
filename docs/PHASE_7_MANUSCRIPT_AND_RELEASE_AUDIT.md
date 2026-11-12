# Phase 7: 17-Section Manuscript Compilation, Cross-Platform CI & `v2.0.2` Release Audit

**Project**: *Evidence-Validated Strategy Adaptation for Memory-Augmented Agents Under Distribution Shift*  
**Author**: Dev Parth — Independent Researcher (*Independent Research in Artificial Intelligence*)  
**Release Version**: `v2.0.2` | **Date**: 9 November 2026  

---

## 1. Synchronized 17-Section Manuscript (`paper/research_paper.{tex,pdf,docx}`)

The publication manuscript (`paper/research_paper.tex`) comprises **17 numbered sections** (zero `Overfull \hbox` warnings, zero BibTeX warnings, 10 vector figures `fig01`–`fig10`, 9 booktabs tables, 1 pseudocode algorithm, and 30 verified BibTeX references in `paper/references.bib`):
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

`paper/build_docx.py` programmatically parses `paper/research_paper.tex`, `experiments/statistical_analysis/statistical_digest.json`, `experiments/statistical_analysis/paired_hypothesis_tests_holm.csv`, and `paper/references.bib` (with explicit `encoding="utf-8"`) to build `paper/research_paper.docx` in lockstep.

---

## 2. `v2.0.2` Release Integrity Checklist

1. **Standardized Independent Attribution**: Removed all prior university/student mentions across 100% of repository files (`Dev Parth` / `Independent Researcher` / `Independent Research in Artificial Intelligence`), verified by `test_independent_researcher_attribution_zero_college_mentions`.
2. **Consolidated Authoritative Phase Documentation**: Retained strictly the 7 detailed `docs/PHASE_1_PROBLEM_AND_THEORY.md`..`PHASE_7_MANUSCRIPT_AND_RELEASE_AUDIT.md` dossiers and deleted empty duplicate outline files, verified by `test_documentation_references_exact_source_symbols`.
3. **Direct Single-Pass Primary Holm Ledger**: Pre-selected `primary_test` + `primary_pvalue_raw` per comparison and applied `_holm_adjust` once directly per family (`19` families, `62/74` sig.) and once directly across all 74 tests (`60/74` sig.), alongside labelled pure-$t$ (`60/74`) and pure-Wilcoxon (`57/74`) sensitivity families, verified by `test_reproducible_global74_holm_ledger`.
4. **Cross-Platform Windows + Linux Compatibility**: Replaced Linux-only `lscpu | grep ...` (`shell=True`) with pure-Python/stdlib `detect_cpu_model()`, enforced `encoding="utf-8"` on all text I/O, and structured `.github/workflows/ci.yml` into a fast 4-leg `unit-and-audit-tests` matrix (`ubuntu-latest` Python `3.11–3.13` + `windows-latest` Python `3.13` running `python -m pytest -v tests/test_evsa_suite.py`, `18` tests) plus a separate `reproducibility-and-docx-check` job—verified locally by `test_cross_platform_cpu_detection_and_utf8_safety` and remotely on `https://github.com/Devparth7-coder/EVSA` (`GitHub Actions run #43252668879` on `6316f48`, `5 / 5` jobs passing).
5. **Release Timestamp Alignment & Post-Push Tag Verification**: Aligned manuscript, `CITATION.cff`, release notes, and manifests to `9 November 2026` (`2026-11-09`), standardized repository references to `Devparth7-coder/EVSA`, and configured `experiments/verify_github_release.py` (`--repo Devparth7-coder/EVSA`) to verify remote tag resolution and CI check-runs once `v2.0.2` is pushed after release-material consistency checks.
