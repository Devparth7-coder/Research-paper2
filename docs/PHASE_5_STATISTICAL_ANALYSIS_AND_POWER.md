# Phase 5: Statistical Hypothesis Testing, Power Analysis & Direct Global-74 Multiplicity Ledger (`v2.0.2`)

**Project**: *Evidence-Validated Strategy Adaptation for Memory-Augmented Agents Under Distribution Shift*  
**Author**: Dev Parth — Independent Researcher (*Independent Research in Artificial Intelligence*)  
**Release Version**: `v2.0.2` | **Date**: 9 November 2026  

---

## 1. Pre-Selected Primary Statistical Methodology (`src/evaluation/metrics_and_stats.py`)

Every comparative claim is evaluated across $N = 20$ paired random seeds (`1001..1020`) via `paired_statistical_comparison`:
1. **Parametric & Non-Parametric Dual Computation (Tie- and Zero-Aware Wilcoxon)**: Two-sided paired Student's $t$-test (`scipy.stats.ttest_rel`) alongside two-sided Wilcoxon signed-rank test (`scipy.stats.wilcoxon(nonzero_diff, zero_method="wilcox", alternative="two-sided", method="auto")` over non-zero paired differences $|d_i| > 10^{-12}$). In `SciPy 1.17.1`, `method="auto"` evaluates an **exact** null distribution only when non-zero paired differences have zero ties (`16 / 74` tests; `4 / 8` Wilcoxon-primary, conditional on $n_{\neq 0}$ when zeros are dropped) and switches to **tie-adjusted asymptotic normal** $p$-values (`correction=False`) when discrete episode counts produce ties at $n_{\neq 0} > 13$ (`58 / 74` tests; `4 / 8` Wilcoxon-primary) or exact sign-permutation when $n_{\neq 0} \le 13$. `paired_statistical_comparison` explicitly records `n_zero_diffs`, `wilcoxon_n_nonzero`, `wilcoxon_has_ties`, and `wilcoxon_method`.
2. **Pre-Adjustment Primary Test Selection & Preliminary-Test Conditioning Caveat**: In the preserved `v2.0.2` 74-test ledger, paired seed differences $d_i = x_i - y_i$ are audited via Shapiro-Wilk (`scipy.stats.shapiro`) *prior* to multiplicity correction: whenever $p_{\mathrm{Shapiro}} < 0.05$ (`8` of `74` comparisons), Wilcoxon signed-rank serves as `primary_test`; otherwise (`66` of `74` comparisons), paired Student's $t$ serves as `primary_test`, recording `primary_pvalue_raw`. Because selecting a test after a normality pretest on the same $N = 20$ sample introduces two-stage conditioning (so single-stage nominal Type-I error guarantees are approximate rather than exact, and pre-committing to a single unconditional primary test—such as a paired $t$-test or exact paired sign-flip permutation test with Wilcoxon sensitivity—is recommended for future confirmatory protocols), we verify sensitivity across unconditional pure-$t$ and pure-Wilcoxon families: **`0 / 74` per-family decisions** (`62 / 74` across all three) and **`0 / 74` Global-74 decisions vs. pure-$t$** (`60 / 74`) depend on the pretest (`57 / 74` under pure Global-74 Wilcoxon).
3. **Standardized Effect Sizes & Percentile Bootstrap CIs**: Paired Cohen's $d_z = \bar{d} / s_d$, bias-corrected Hedges' $g_z$, rank-biserial correlation $r_{rb}$, Student-$t$ 95% CIs, and **5,000-resample non-parametric percentile bootstrap 95% CIs** over seed-level means (`summarize_seed_distribution(..., n_boot=5000, boot_seed=2026)`).
4. **Paired-$t$ Exact Non-Central-$t$ Power Analysis (`compute_paired_power`, `minimum_detectable_dz`)**:
   - Evaluates the exact non-central $t_{19}$ distribution specifically for a two-sided **paired Student's $t$-test** under normal paired differences (not Wilcoxon or two-stage selector power): at $N = 20$ paired seeds and two-sided $\alpha = 0.05$, the paired-$t$ Minimum Detectable Effect Size is **`dz = 0.6604`** (`80%` paired-$t$ power) and **`dz = 0.7645`** (`90%` paired-$t$ power), verified by `test_statistical_power_and_holm_bonferroni`.

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

---

## 3. Planned Separate Statistical Revision (`v2.1.0` Protocol Decision)

To keep release `v2.0.2` strictly reproducible against its archived 74-row ledger (`paired_hypothesis_tests_holm.csv`), replacing the Shapiro-Wilk primary selector is scoped to a separate statistical revision (`v2.1.0`) governed by the following pre-committed protocol:
1. **Single Unconditional Primary Test**: Replace data-dependent `shapiro_p < 0.05` branching with an **exact two-sided paired sign-flip randomization test on the paired mean difference $\bar{d}$** across all $2^{20} = 1{,}048{,}576$ sign configurations (`1.907e-6` minimum two-sided exact $p$-value, zero Monte Carlo error, zero normality requirement, and zero discrete rank-tie approximation).
2. **Retained Secondary Columns**: Report unconditional paired Student's $t$-test (aligned directly with exact non-central-$t_{19}$ MDES `dz = 0.6604`) and tie-aware Wilcoxon signed-rank (`wilcoxon_method` explicitly logged) alongside the randomization primary column, relegating Shapiro-Wilk strictly to a descriptive normality diagnostic.
3. **Empirical Stability Preview on `1001..1020`**: Evaluated read-only on the saved `v2.0.2` raw seed logs, exact $2^{20}$ sign-flip randomization yields **`62 / 74` per-family Holm-significant tests** (`0` decision flips vs. `v2.0.2` primary, pure-$t$, and pure-Wilcoxon) and **`59 / 74` Global-74 Holm-significant tests** (`1` borderline difference vs. `v2.0.2` at row `52`, `RQ6_Contamination_eta_0.35` vs. `Experience Replay`: `0.0469 -> 0.0564`).
