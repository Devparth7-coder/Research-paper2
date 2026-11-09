# Phase 5: Statistical Hypothesis Testing, Power Analysis & Global-74 Multiplicity Ledger (`v2.0.1`)

**Project**: *Evidence-Validated Strategy Adaptation for Memory-Augmented Agents Under Distribution Shift*  
**Author**: Dev Parth (Department of Computer Science and Engineering, MMMUT, Gorakhpur, Uttar Pradesh, India)  
**Release Version**: `v2.0.1` | **Date**: 7 November 2026  

---

## 1. Statistical Methodology (`src/evaluation/metrics_and_stats.py`)

Every comparative claim is evaluated across $N = 20$ paired random seeds (`1001..1020`) via `paired_statistical_comparison`:
1. **Parametric & Non-Parametric Dual Testing**: Two-sided paired Student's $t$-test (`scipy.stats.ttest_rel`) alongside exact two-sided Wilcoxon signed-rank test (`scipy.stats.wilcoxon`).
2. **Normality-Guided Primary Selection**: Paired seed differences $d_i = x_i - y_i$ are audited via Shapiro-Wilk (`scipy.stats.shapiro`); whenever $p_{\mathrm{Shapiro}} < 0.05$, Wilcoxon serves as `primary_test`, otherwise Student's $t$.
3. **Standardized Effect Sizes & Bootstrap CIs**: Paired Cohen's $d_z = \bar{d} / s_d$, bias-corrected Hedges' $g_z$, rank-biserial correlation $r_{rb}$, Student-$t$ 95% CIs, and 4,000-resample BCa-style bootstrap 95% CIs.
4. **Exact Non-Central-$t$ Power Analysis (`compute_paired_power`, `minimum_detectable_dz`)**:
   - At $N = 20$ paired seeds and two-sided $\alpha = 0.05$, Minimum Detectable Effect Size is **`dz = 0.6604`** (`80%` power) and **`dz = 0.7645`** (`90%` power), verified by `test_statistical_power_and_holm_bonferroni`.

---

## 2. Reproducible Per-Family vs. Simultaneous Global-74 Holm-Bonferroni Ledgers

In `v2.0.1`, `experiments/run_complete_suite.py` exports both multiplicity scopes directly into `experiments/statistical_analysis/paired_hypothesis_tests_holm.csv`, `tables/table_statistical_tests_all.csv`, and `experiments/statistical_analysis/statistical_digest.json`:
- **Per-Family Holm-Bonferroni (`apply_holm_bonferroni`)**: Applied within each of the **16 pre-defined experimental metric families** across all $M = 74$ paired tests $\implies$ **`62 / 74` tests significant at $\alpha = 0.05$**.
- **Simultaneous Global-74 Holm-Bonferroni (`apply_global_holm_bonferroni`)**: Applied simultaneously across all $M = 74$ paired tests pooled together (`t_pvalue_holm_global74`, `wilcoxon_pvalue_holm_global74`, `primary_pvalue_holm_global74`, `significant_global74_at_05`) $\implies$ **`60 / 74` tests significant at $\alpha = 0.05$**.

### Exact Accounting of the Two Multiplicity-Scope Borderlines
1. `RQ6_Contamination_eta_0.35` (`EVSA` vs. `Baseline B: Conventional Memory`): raw $p_t = 0.00364$, per-family `t_pvalue_holm = 0.00627` (**Significant**), global-74 `t_pvalue_holm_global74 = 0.05098` (**Borderline $> 0.05$**; vs. `Replay` `0.04695` and vs. `Reflection` `0.02579` remain $< 0.05$ globally).
2. `RQ3_Ablation_Overall_SR` (`EVSA Full` vs. `w/o Shift Detection`): raw $p_t = 0.01160$, per-family `t_pvalue_holm = 0.04639` (**Significant**), global-74 `t_pvalue_holm_global74 = 0.15076` (**Non-Significant**, while the post-shift half `RQ3_Ablation_PostShift_SR` remains significant under both scopes: `0.00370` per-family and `0.01749` global-74).

Verified programmatically to $< 10^{-15}$ numerical tolerance by `test_reproducible_global74_holm_ledger` in `tests/test_evsa_suite.py`.
