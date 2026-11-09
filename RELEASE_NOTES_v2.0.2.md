# Release Notes — `EVSA v2.0.2` (Independent Attribution & Release-Integrity Correction)

**Author**: Dev Parth  
**Role**: Independent Researcher  
**Research Area**: Independent Research in Artificial Intelligence  
**Release Date**: 9 November 2026 (`2026-11-09`)  

---

## Summary of Changes in `v2.0.2`

1. **Standardized Independent Attribution**:
   - Replaced all prior university/student affiliation strings across `README.md`, `CITATION.cff`, `config/default_config.json`, `FINAL_AUDIT_REPORT.md`, `REPRODUCIBILITY.md`, `TIMELINE.md`, `UNRESOLVED_LIMITATIONS.md`, `docs/PHASE_1..7_*.md`, `paper/research_paper.{tex,pdf,docx}`, `paper/build_docx.py`, `src/__init__.py`, and `experiments/run_complete_suite.py` with:
     - **Author**: `Dev Parth`
     - **Role**: `Independent Researcher`
     - **Research Area**: `Independent Research in Artificial Intelligence`
   - Enforced continuously via `test_independent_researcher_attribution_zero_college_mentions` in `tests/test_evsa_suite.py`.
2. **Consolidated Authoritative Phase Documentation (`docs/`)**:
   - Retained the seven detailed research dossiers (`docs/PHASE_1_PROBLEM_AND_THEORY.md` through `docs/PHASE_7_MANUSCRIPT_AND_RELEASE_AUDIT.md`) and removed the seven duplicate stub outlines (`docs/PHASE_1_LITERATURE.md` through `docs/PHASE_7_AUDIT_AND_RELEASE.md`) plus superseded release-note drafts after confirming zero inbound references (`test_documentation_references_exact_source_symbols`).
3. **Corrected Direct Single-Pass Primary Holm-Bonferroni Adjustment (`src/evaluation/metrics_and_stats.py`)**:
   - Pre-selects `primary_test` (`Wilcoxon signed-rank` when `shapiro_p < 0.05` [`8` tests], `Paired t-test` when `shapiro_p >= 0.05` [`66` tests]) and `primary_pvalue_raw` for each comparison before multiplicity adjustment.
   - Applies Holm's step-down procedure (`_holm_adjust`) **once directly** to `primary_pvalue_raw` within each of the **19 pre-defined metric families** (`primary_pvalue_holm`: **`62 / 74`** significant) and **once directly** to the 74-element Global-74 primary family (`primary_pvalue_holm_global74`: **`60 / 74`** significant), rather than splicing separately adjusted parametric and non-parametric vectors.
   - Exports explicitly labelled pure-test sensitivity families (`sensitivity_t_pvalue_holm_global74`: **`60 / 74`** significant; `sensitivity_wilcoxon_pvalue_holm_global74`: **`57 / 74`** significant), verified to `< 1e-15` by `test_reproducible_global74_holm_ledger`.
4. **Cross-Platform Windows + Linux Reproducibility**:
   - Replaced Linux-only `lscpu | grep ...` (`shell=True`) in `experiments/run_complete_suite.py` with cross-platform pure-Python/stdlib `detect_cpu_model()` (`winreg` / `PROCESSOR_IDENTIFIER` on Windows, `/proc/cpuinfo` on Linux, `sysctl` without `shell=True` on macOS, `platform.processor()` fallback) and explicit `encoding="utf-8"` file I/O (`test_cross_platform_cpu_detection_and_utf8_safety`).
   - Expanded `.github/workflows/ci.yml` to run on both `ubuntu-latest` (`Python 3.11`, `3.12`, `3.13`) and `windows-latest` (`Python 3.13`).
5. **Release Timestamp Alignment & Honest Remote Tag/CI Verification**:
   - Aligned manuscript, `CITATION.cff`, release notes, and regenerated manifests to **`9 November 2026` (`2026-11-09`)**.
   - Local clean-clone verification confirms **`16 / 16` passing tests** and full artifact regeneration. Remote GitHub tag creation and GitHub Actions check-run completion occur **only after** pushing both `main` and the annotated tag `v2.0.2` to `origin`.

---

## Publishing `v2.0.2` to GitHub & Verifying Remote Tag / CI Check-Runs

Because `git push origin main` alone does **not** push local Git tags, push both the branch and tag explicitly from an authenticated terminal (Linux, Windows PowerShell/CMD, or macOS):

```bash
git push origin main
git push origin v2.0.2
```

Once GitHub Actions completes on the pushed commit, verify the live remote tag ref, commit SHA match, and green CI check-runs directly via the GitHub REST API:

```bash
python experiments/verify_github_release.py --tag v2.0.2
```
(Add `--require-release` after creating a GitHub Release on tag `v2.0.2` if attaching release binaries such as `paper/research_paper.pdf` and `paper/research_paper.docx`.)
