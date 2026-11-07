"""
Evaluation, Statistical Inference, and Power Analysis Engine (Module 10).

Implements:
- Task accuracy, relative improvement (%), adaptation gain, recovery rate (%)
- Frequentist seed-level summary statistics (Mean, Median, SD, IQR, SEM, 95% t-CI,
  95% percentile bootstrap CI)
- Paired Student's t-test, exact Wilcoxon signed-rank test, Shapiro-Wilk diagnostic
- Paired Cohen's d_z, small-sample Hedges' g_z, rank-biserial correlation r_rb
- Exact non-central t statistical power & minimum detectable effect size (MDES)
- Holm-Bonferroni step-down family-wise error rate control (Holm, 1979)
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple
import numpy as np
from scipy import stats


def _holm_adjust(pvals: Sequence[float], alpha: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
    """Exact Holm (1979) step-down sequentially rejective FWER adjustment."""
    p = np.asarray(pvals, dtype=np.float64)
    m = p.size
    if m == 0:
        return np.array([], dtype=bool), np.array([], dtype=np.float64)
    order = np.argsort(p)
    sorted_p = p[order]
    factors = np.arange(m, 0, -1, dtype=np.float64)
    adj_sorted = np.minimum(1.0, np.maximum.accumulate(sorted_p * factors))
    adj = np.empty(m, dtype=np.float64)
    adj[order] = adj_sorted
    rej = adj <= alpha
    return rej, adj


def compute_accuracy(correct_or_successes: Sequence[int]) -> float:
    arr = np.asarray(correct_or_successes, dtype=np.float64)
    if arr.size == 0:
        return 0.0
    return float(np.sum(arr) / float(arr.size))


def compute_relative_improvement_pct(proposed_perf: float, baseline_perf: float) -> float:
    if abs(baseline_perf) < 1e-12:
        return 0.0
    return float(((proposed_perf - baseline_perf) / baseline_perf) * 100.0)


def compute_adaptation_gain(post_perf: float, pre_perf: float) -> float:
    return float(post_perf - pre_perf)


def compute_recovery_rate_pct(recovered_perf: float, original_perf: float) -> float:
    if abs(original_perf) < 1e-12:
        return 0.0
    return float((recovered_perf / original_perf) * 100.0)


def compute_paired_power(effect_size_dz: float, n_seeds: int = 20, alpha: float = 0.05) -> float:
    """Compute exact two-sided paired t-test power via non-central t distribution."""
    if n_seeds < 2 or abs(effect_size_dz) < 1e-12:
        return float(alpha)
    df = n_seeds - 1
    ncp = abs(effect_size_dz) * np.sqrt(n_seeds)
    if ncp >= 12.0:
        return 1.0
    t_crit = float(stats.t.ppf(1.0 - alpha / 2.0, df=df))
    upper = float(stats.nct.sf(t_crit, df=df, nc=ncp))
    lower = float(stats.nct.cdf(-t_crit, df=df, nc=ncp))
    if np.isnan(upper):
        upper = 1.0 if ncp > t_crit else 0.0
    if np.isnan(lower):
        lower = 0.0
    return float(np.clip(upper + lower, 0.0, 1.0))


def minimum_detectable_dz(n_seeds: int = 20, alpha: float = 0.05, target_power: float = 0.80) -> float:
    """Solve for minimum paired Cohen's d_z achieving target_power at two-sided alpha."""
    lo, hi = 0.0, 5.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if compute_paired_power(mid, n_seeds=n_seeds, alpha=alpha) < target_power:
            lo = mid
        else:
            hi = mid
    return float(0.5 * (lo + hi))


def summarize_seed_distribution(
    values: Sequence[float],
    ci_level: float = 0.95,
    n_boot: int = 5000,
    boot_seed: int = 2026,
) -> Dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    n = int(arr.size)
    if n == 0:
        return {
            "n": 0, "mean": 0.0, "median": 0.0, "sd": 0.0, "iqr": 0.0, "sem": 0.0,
            "min": 0.0, "max": 0.0, "ci95_low": 0.0, "ci95_high": 0.0,
            "boot_ci95_low": 0.0, "boot_ci95_high": 0.0,
        }
    mean = float(np.mean(arr))
    median = float(np.median(arr))
    q25, q75 = np.percentile(arr, [25.0, 75.0])
    iqr = float(q75 - q25)
    vmin, vmax = float(np.min(arr)), float(np.max(arr))
    if n == 1:
        return {
            "n": 1, "mean": mean, "median": median, "sd": 0.0, "iqr": 0.0, "sem": 0.0,
            "min": vmin, "max": vmax, "ci95_low": mean, "ci95_high": mean,
            "boot_ci95_low": mean, "boot_ci95_high": mean,
        }
    sd = float(np.std(arr, ddof=1))
    sem = float(sd / np.sqrt(n))
    t_crit = float(stats.t.ppf(0.5 + 0.5 * ci_level, df=n - 1))
    ci_low = mean - t_crit * sem
    ci_high = mean + t_crit * sem

    rng = np.random.default_rng(boot_seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    boot_means = np.mean(arr[idx], axis=1)
    alpha = 1.0 - ci_level
    b_low, b_high = np.percentile(boot_means, [100.0 * (alpha / 2.0), 100.0 * (1.0 - alpha / 2.0)])

    return {
        "n": n,
        "mean": mean,
        "median": median,
        "sd": sd,
        "iqr": iqr,
        "sem": sem,
        "min": vmin,
        "max": vmax,
        "ci95_low": float(ci_low),
        "ci95_high": float(ci_high),
        "boot_ci95_low": float(b_low),
        "boot_ci95_high": float(b_high),
    }


def paired_statistical_comparison(
    proposed_vals: Sequence[float],
    comparator_vals: Sequence[float],
    proposed_label: str,
    comparator_label: str,
) -> Dict[str, Any]:
    x = np.asarray(proposed_vals, dtype=np.float64)
    y = np.asarray(comparator_vals, dtype=np.float64)
    if x.shape != y.shape or x.size < 3:
        raise ValueError("Paired comparison requires equal-length arrays with n >= 3.")

    diff = x - y
    n = int(diff.size)
    mean_x = float(np.mean(x))
    mean_y = float(np.mean(y))
    mean_diff = float(np.mean(diff))
    median_diff = float(np.median(diff))
    sd_diff = float(np.std(diff, ddof=1))
    sem_diff = sd_diff / np.sqrt(n) if sd_diff > 0 else 0.0
    t_crit = float(stats.t.ppf(0.975, df=n - 1))
    diff_ci_low = mean_diff - t_crit * sem_diff
    diff_ci_high = mean_diff + t_crit * sem_diff

    if sd_diff > 1e-12:
        shapiro_stat, shapiro_p = stats.shapiro(diff)
        t_res = stats.ttest_rel(x, y)
        t_stat, t_p = float(t_res.statistic), float(t_res.pvalue)
    else:
        shapiro_stat, shapiro_p = 1.0, 1.0
        t_stat, t_p = 0.0, 1.0

    nonzero_diff = diff[np.abs(diff) > 1e-12]
    if nonzero_diff.size >= 3:
        w_res = stats.wilcoxon(nonzero_diff, zero_method="wilcox", alternative="two-sided")
        w_stat, w_p = float(w_res.statistic), float(w_res.pvalue)
        ranks = stats.rankdata(np.abs(nonzero_diff))
        pos_rank_sum = float(np.sum(ranks[nonzero_diff > 0]))
        neg_rank_sum = float(np.sum(ranks[nonzero_diff < 0]))
        rank_biserial = (pos_rank_sum - neg_rank_sum) / max(1.0, pos_rank_sum + neg_rank_sum)
    else:
        w_stat, w_p, rank_biserial = 0.0, 1.0, 0.0

    cohens_dz = mean_diff / sd_diff if sd_diff > 1e-12 else 0.0
    correction_j = 1.0 - (3.0 / (4.0 * (n - 1) - 1.0))
    hedges_gz = cohens_dz * correction_j
    achieved_power = compute_paired_power(cohens_dz, n_seeds=n, alpha=0.05)

    return {
        "proposed": proposed_label,
        "comparator": comparator_label,
        "n_seeds": n,
        "proposed_mean": mean_x,
        "comparator_mean": mean_y,
        "mean_diff": mean_diff,
        "median_diff": median_diff,
        "sd_diff": sd_diff,
        "diff_ci95_low": float(diff_ci_low),
        "diff_ci95_high": float(diff_ci_high),
        "relative_improvement_pct": compute_relative_improvement_pct(mean_x, mean_y),
        "shapiro_w": float(shapiro_stat),
        "shapiro_p": float(shapiro_p),
        "t_stat": t_stat,
        "t_pvalue": t_p,
        "wilcoxon_stat": w_stat,
        "wilcoxon_pvalue": w_p,
        "cohens_dz": float(cohens_dz),
        "hedges_gz": float(hedges_gz),
        "rank_biserial_r": float(rank_biserial),
        "achieved_power_05": float(achieved_power),
    }


def apply_holm_bonferroni(comparisons: List[Dict[str, Any]], alpha: float = 0.05) -> List[Dict[str, Any]]:
    if not comparisons:
        return comparisons
    t_pvals = [c["t_pvalue"] for c in comparisons]
    w_pvals = [c["wilcoxon_pvalue"] for c in comparisons]

    t_rej, t_adj = _holm_adjust(t_pvals, alpha=alpha)
    w_rej, w_adj = _holm_adjust(w_pvals, alpha=alpha)

    enriched: List[Dict[str, Any]] = []
    for idx, comp in enumerate(comparisons):
        row = dict(comp)
        row["t_pvalue_holm"] = float(t_adj[idx])
        row["t_significant_holm"] = bool(t_rej[idx])
        row["wilcoxon_pvalue_holm"] = float(w_adj[idx])
        row["wilcoxon_significant_holm"] = bool(w_rej[idx])
        use_nonparam = row["shapiro_p"] < 0.05
        row["primary_test"] = "Wilcoxon signed-rank" if use_nonparam else "Paired t-test"
        row["primary_pvalue_holm"] = row["wilcoxon_pvalue_holm"] if use_nonparam else row["t_pvalue_holm"]
        row["significant_at_05"] = row["wilcoxon_significant_holm"] if use_nonparam else row["t_significant_holm"]
        enriched.append(row)
    return enriched
