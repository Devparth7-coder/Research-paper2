"""
Discounted Sequential Log-Likelihood Ratio Validation Gate (Module 5).

Adapts Wald's Sequential Probability Ratio Test (SPRT) log-likelihood increments
with exponential forgetting (gamma_sprt = 0.985), clipping [-6, 6], a minimum
replication count (min_trials = 3), and a discounted Beta lower credible bound
threshold (tau_lcb = 0.48).

Important statistical qualification: Because log-odds are exponentially
discounted, clipped, pooled under LGG merges, and evaluated across multiple
adaptively selected candidate strategies without family-wise alpha-spending,
classical i.i.d. Wald Type-I/Type-II error inequalities do not strictly hold as
formal guarantees; instead, (alpha_fa, beta_fr) serve as calibration parameters
whose empirical false-promotion and false-rejection rates are audited directly.
"""

from __future__ import annotations

import math
import numpy as np

from ..benchmark.types import StrategyRecord, StrategyStatus
from .confidence import ConfidenceEstimator


class EvidenceValidationEngine:
    """Discounted SPRT-inspired sequential promotion and quarantine gate."""

    def __init__(
        self,
        p0: float = 0.35,
        p1: float = 0.80,
        alpha_fa: float = 0.11,
        beta_fr: float = 0.15,
        min_trials: int = 3,
        tau_lcb: float = 0.48,
        tau_invalidate_lcb: float = 0.45,
        gamma_sprt: float = 0.985,
        validation_enabled: bool = True,
        invalidation_enabled: bool = True,
    ) -> None:
        self.p0 = p0
        self.p1 = p1
        self.alpha_fa = alpha_fa
        self.beta_fr = beta_fr
        self.min_trials = min_trials
        self.tau_lcb = tau_lcb
        self.tau_invalidate_lcb = tau_invalidate_lcb
        self.gamma_sprt = gamma_sprt
        self.validation_enabled = validation_enabled
        self.invalidation_enabled = invalidation_enabled

        self.upper_bound = math.log((1.0 - beta_fr) / alpha_fa)
        self.lower_bound = math.log(beta_fr / (1.0 - alpha_fa))
        self._pos_step = math.log(p1 / p0)
        self._neg_step = math.log((1.0 - p1) / (1.0 - p0))

    def step_evidence(
        self,
        record: StrategyRecord,
        observed_success: int,
        confidence_estimator: ConfidenceEstimator,
        episode_idx: int,
    ) -> StrategyStatus:
        inc = self._pos_step if int(observed_success) == 1 else self._neg_step
        record.sprt_llr = float(np.clip(self.gamma_sprt * record.sprt_llr + inc, -6.0, 6.0))
        record.last_updated_episode = episode_idx
        lcb = confidence_estimator.lower_credible_bound(record)

        if not self.validation_enabled:
            if record.status == StrategyStatus.CANDIDATE and int(observed_success) == 1:
                record.status = StrategyStatus.VALIDATED
                record.promoted_episode = episode_idx
            elif (
                self.invalidation_enabled
                and record.consecutive_failures >= 2
                and lcb < self.tau_invalidate_lcb
            ):
                record.status = StrategyStatus.QUARANTINED
                record.invalidated_episode = episode_idx
            return record.status

        if record.status == StrategyStatus.CANDIDATE:
            if (
                record.n_trials >= self.min_trials
                and record.sprt_llr >= self.upper_bound
                and lcb >= self.tau_lcb
            ):
                record.status = StrategyStatus.VALIDATED
                record.promoted_episode = episode_idx
            elif record.sprt_llr <= self.lower_bound or (
                record.n_trials >= 2 and record.n_successes == 0
            ):
                record.status = StrategyStatus.QUARANTINED
                record.invalidated_episode = episode_idx

        elif record.status == StrategyStatus.VALIDATED and self.invalidation_enabled:
            if record.sprt_llr <= self.lower_bound or (
                record.consecutive_failures >= 2 and lcb < self.tau_invalidate_lcb
            ):
                record.status = StrategyStatus.QUARANTINED
                record.invalidated_episode = episode_idx

        return record.status
