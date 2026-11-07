"""
Discounted Beta-Bernoulli Posterior & Credible-Bound Estimator (Module 6).

Computes exact Bayesian posterior parameters, posterior predictive mean, and
lower/upper credible quantiles (`betaincinv`) under exponential forgetting
factor gamma_base in (0, 1]. Note: intervals are Bayesian credible intervals,
distinct from frequentist coverage intervals under non-stationarity.

Ablation Mode (`enabled=False`, `w/o Confidence Estimation`):
Replaces Bayesian prior smoothing and lower/upper credible quantiles with the
unregularized discounted empirical success frequency (MLE; 0.50 prior to the
first trial), removing both small-sample lower-bound conservatism and upper-bound
uncertainty protection.
"""

from __future__ import annotations

from typing import Optional
import numpy as np
from scipy.special import betaincinv

from ..benchmark.types import StrategyRecord


class ConfidenceEstimator:
    """Maintains discounted Beta(alpha, beta) posteriors over strategy success probability."""

    def __init__(
        self,
        prior_alpha: float = 1.0,
        prior_beta: float = 1.0,
        gamma_base: float = 0.985,
        forgetting_gamma: Optional[float] = None,
        credible_delta: float = 0.10,
        enabled: bool = True,
    ) -> None:
        self.prior_alpha = prior_alpha
        self.prior_beta = prior_beta
        self.gamma_base = gamma_base if forgetting_gamma is None else forgetting_gamma
        self.credible_delta = credible_delta
        self.enabled = enabled

    def _empirical_frequency(self, record: StrategyRecord) -> float:
        """Unregularized discounted empirical mean (MLE) without Beta prior shrinkage."""
        disc_succ = max(0.0, record.alpha - self.prior_alpha)
        disc_fail = max(0.0, record.beta - self.prior_beta)
        disc_total = disc_succ + disc_fail
        if disc_total <= 1e-6 or record.n_trials == 0:
            return 0.50
        return float(np.clip(disc_succ / disc_total, 0.0, 1.0))

    def update_posterior(
        self,
        record: StrategyRecord,
        observed_success: int,
        episode_idx: int = 0,
    ) -> None:
        """Apply exponential forgetting toward Beta(alpha_0, beta_0) and incorporate new Bernoulli trial."""
        excess_a = max(0.0, record.alpha - self.prior_alpha)
        excess_b = max(0.0, record.beta - self.prior_beta)
        record.alpha = self.prior_alpha + self.gamma_base * excess_a + float(observed_success)
        record.beta = self.prior_beta + self.gamma_base * excess_b + float(1 - observed_success)
        record.n_trials += 1
        record.n_successes += int(observed_success)
        record.last_updated_episode = episode_idx
        if int(observed_success) == 1:
            record.consecutive_failures = 0
        else:
            record.consecutive_failures += 1

    def posterior_mean(self, record: StrategyRecord) -> float:
        if not self.enabled:
            return self._empirical_frequency(record)
        return float(record.alpha / (record.alpha + record.beta))

    def lower_credible_bound(self, record: StrategyRecord) -> float:
        """Exact delta-quantile of Beta(alpha_h, beta_h) posterior (or raw empirical mean if disabled)."""
        if not self.enabled:
            return self._empirical_frequency(record)
        return float(betaincinv(record.alpha, record.beta, self.credible_delta))

    def upper_credible_bound(self, record: StrategyRecord) -> float:
        """Exact (1 - delta)-quantile of Beta(alpha_h, beta_h) for Bayes-UCB candidate exploration."""
        if not self.enabled:
            return self._empirical_frequency(record)
        return float(betaincinv(record.alpha, record.beta, 1.0 - self.credible_delta))
