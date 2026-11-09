# Phase 5: Statistical Hypothesis Testing, Power Analysis & Direct Global-74 Multiplicity Ledger (`v2.0.2`)

**Project**: *Evidence-Validated Strategy Adaptation for Memory-Augmented Agents Under Distribution Shift*  
**Author**: Dev Parth — Independent Researcher (*Independent Research in Artificial Intelligence*)  
**Release Version**: `v2.0.2` | **Date**: 9 November 2026  

---

## 1. Pre-Selected Primary Statistical Methodology (`src/evaluation/metrics_and_stats.py`)

Every comparative claim is evaluated across $N = 20$ paired random seeds (`1001..1020`) via `paired_statistical_comparison`:
1. **Parametric & Non-Parametric Dual Computation**: Two-sided paired Student's $t$-test (`scipy.stats.ttest_rel`) alongside exact two-sided Wilcoxon signed-rank test (`scipy.stats.wilcoxon`).
2. **Pre-Adjustment Primary Test Selection**: Paired seed differences $d_i = x_i - y_i$ are audited via Shapiro-Wilk (`scipy.stats.shapiro`) *prior* to multiplicity correction: whenever $p_{\mathrm{Shapiro}} < 0.05$ (`8` of `74` comparisons), exact Wilcoxon signed-rank serves as `primary_test`; otherwise (`66` of `74` comparisons), paired Student's $t$ serves as `primary_test`, recording `primary_pvalue_raw`.
3. **Standardized Effect Sizes & Bootstrap CIs**: Paired Cohen's $d_z = \bar{d} / s_d$, bias-corrected Hedges' $g_z$, rank-biserial correlation $r_{rb}$, Student-$t$ 95% CIs, and 4,000-resample BCa-style bootstrap 95% CIs.
4. **Exact Non-Central-$t$ Power Analysis (`compute_paired_power`, `minimum_detectable_dz`)**:
   - At $N = 20$ paired seeds and two-sided $\alpha = 0.05$, Minimum Detectable Effect Size is **`dz = 0.6604`** (`80%` power) and **`dz = 0.7645`** (`90%` power), verified by `test_statistical_power_and_holm_bonferroni`.

---

## 2. Direct Single-Pass Primary Holm vs. Sensitivity Families (`v2.0.2`)

In `v2.0.2`, `experiments/run_complete_suite.py` exports both multiplicity scopes—and separates the single-pass primary family from pure-test sensitivity families—directly into `experiments/statistical_analysis/paired_hypothesis_tests_holm.csv`, `tables/table_statistical_tests_all.csv`, and `experiments/statistical_analysis/statistical_digest.json`:
- **Direct Per-Family Primary Holm (`apply_holm_bonferroni`)**: Applies `_holm_adjust` **once directly** to `primary_pvalue_raw` within each of the **19 pre-defined experimental metric families** (`primary_pvalue_holm`, `significant_at_05`) $\implies$ **`62 / 74` tests significant at $\alpha = 0.05$** (pure-$t$ sensitivity: `62 / 74`; pure-Wilcoxon sensitivity: `62 / 74`).
- **Direct Simultaneous Global-74 Primary Holm (`apply_global_holm_bonferroni`)**: Applies `_holm_adjust` **once directly** to the 74-element `primary_pvalue_raw` vector (`primary_pvalue_holm_global74`, `significant_global74_at_05`) rather than splicing separately adjusted parametric and non-parametric vectors $\implies$ **`60 / 74` tests significant at $\alpha = 0.05$**.
- **Explicit Pure-Test Global-74 Sensitivity Families**: Separately exports `sensitivity_t_pvalue_holm_global74` (**`60 / 74`** significant) and `sensitivity_wilcoxon_pvalue_holm_global74` (**`57 / 74`** significant due to $N=20$ discrete rank flooring).

### Exact Accounting of the Two Global-74 Primary Multiplicity Borderlines
1. `RQ6_Contamination_eta_0.35` (`EVSA` vs. `Baseline B: Conventional Memory`, `Paired t-test` primary): raw $p = 0.00364$, per-family `primary_pvalue_holm = 0.00627` (**Significant**), Global-74 `primary_pvalue_holm_global74 = 0.05097` (**Borderline $> 0.05$**; vs. `Replay` `0.04695` and vs. `Reflection` `0.02579` remain $< 0.05$ globally).
2. `RQ3_Ablation_Overall_SR` (`EVSA Full` vs. `w/o Shift Detection`, `Paired t-test` primary): raw $p = 0.01160$, per-family `primary_pvalue_holm = 0.04639` (**Significant**), Global-74 `primary_pvalue_holm_global74 = 0.15075` (**Non-Significant**, while the post-shift half `RQ3_Ablation_PostShift_SR` remains significant under both scopes: `0.00370` per-family and `0.01749` Global-74).

Verified programmatically to $< 10^{-15}$ numerical tolerance by `test_reproducible_global74_holm_ledger` in `tests/test_evsa_suite.py`.
