"""
Applicability-Constrained Confidence-Weighted Strategy Retriever (Module 7).

Filters VALIDATED strategies by both symbolic applicability phi_h(psi_t) = 1
and lower credible reliability LCB_delta(h) >= tau_rel (0.45), ranking matches via:
  U(h, q_t) = w_sem * max(0, cos(x_t, c_h)) + w_conf * LCB_delta(h) + w_tier + b_spec * |phi_h|
and ranking CANDIDATE strategies via Bayes-UCB exploration utility:
  UCB(h, q_t) = 0.85 * UCB_delta(h) + 0.15 * max(0, cos(x_t, c_h)) + b_spec * |phi_h|.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple
import numpy as np

from ..benchmark.types import StrategyRecord, StrategyStatus, TaskInstance
from ..memory.repository import StrategyRepository
from ..validation.confidence import ConfidenceEstimator


class StrategyRetriever:
    """Retrieves applicable validated and candidate strategies ranked by composite utility."""

    def __init__(
        self,
        confidence_estimator: ConfidenceEstimator,
        use_applicability: bool = True,
        tau_rel: float = 0.45,
        w_sem: float = 0.28,
        w_conf: float = 0.52,
        w_tier: float = 0.20,
        specificity_bonus: float = 0.015,
    ) -> None:
        self.confidence_estimator = confidence_estimator
        self.use_applicability = use_applicability
        self.tau_rel = tau_rel
        self.w_sem = w_sem
        self.w_conf = w_conf
        self.w_tier = w_tier
        self.specificity_bonus = specificity_bonus

    def retrieve(
        self,
        repo: StrategyRepository,
        task: TaskInstance,
    ) -> Tuple[List[Tuple[float, StrategyRecord]], List[Tuple[float, StrategyRecord]], int]:
        """Return `(validated_ranked, candidate_ranked, num_records_evaluated)`."""
        domain_recs = repo.domain_strategies(
            task.domain, statuses=(StrategyStatus.VALIDATED, StrategyStatus.CANDIDATE)
        )
        eval_count = 0
        val_scored: List[Tuple[float, StrategyRecord]] = []
        cand_scored: List[Tuple[float, StrategyRecord]] = []

        for rec in domain_recs:
            eval_count += 1
            if self.use_applicability and not rec.matches_preconditions(task.structural_predicates):
                continue

            cos_sim = float(np.clip(np.dot(task.query_embedding, rec.embedding), -1.0, 1.0))
            pos_sim = max(0.0, cos_sim)
            spec_term = self.specificity_bonus * float(len(rec.preconditions))

            if rec.status == StrategyStatus.VALIDATED:
                lcb = self.confidence_estimator.lower_credible_bound(rec)
                if lcb < self.tau_rel:
                    continue
                score = self.w_sem * pos_sim + self.w_conf * lcb + self.w_tier + spec_term
                val_scored.append((score, rec))
            elif rec.status == StrategyStatus.CANDIDATE:
                ucb = self.confidence_estimator.upper_credible_bound(rec)
                score = 0.85 * ucb + 0.15 * pos_sim + spec_term
                cand_scored.append((score, rec))

        val_scored.sort(key=lambda item: item[0], reverse=True)
        cand_scored.sort(key=lambda item: item[0], reverse=True)
        return val_scored, cand_scored, eval_count

    def detect_contradictions(
        self,
        validated_matches: Sequence[Tuple[float, StrategyRecord]],
    ) -> bool:
        """Return True if two applicable VALIDATED strategies prescribe conflicting plans."""
        if len(validated_matches) < 2:
            return False
        lead_plan = validated_matches[0][1].operator_plan
        return any(r.operator_plan != lead_plan for _, r in validated_matches[1:])
