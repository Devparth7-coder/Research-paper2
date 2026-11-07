"""
Empirical Precondition Induction & Compositional Hypothesis Generator (Module 2).

Learns marginal step-predicate outcome contrasts from observed rollouts without
ever accessing ground-truth optimal plans or latent archetype indices.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np

from ..benchmark.environment import ACTION_DIMS
from ..benchmark.llm_backbone import CalibratedSymbolicLLMBackbone
from ..benchmark.types import RolloutResult, StrategyStatus, TaskInstance
from ..validation.confidence import ConfidenceEstimator
from .repository import StrategyRepository


class HypothesisGenerator:
    """Empirical precondition salience estimator and compositional plan synthesizer."""

    def __init__(self, use_applicability: bool = True, seed: int = 42) -> None:
        self.use_applicability = use_applicability
        self.rng = np.random.default_rng(seed)
        self.step_stats: Dict[Tuple[str, int, int, int], np.ndarray] = {}

    def record_observation(
        self,
        task: TaskInstance,
        res: RolloutResult,
        discount: float = 0.995,
    ) -> None:
        n_preds = len(task.structural_predicates)
        for p_idx in range(n_preds):
            p_val = int(task.structural_predicates[p_idx])
            for s_idx, op in enumerate(res.executed_plan):
                key = (task.domain, p_idx, p_val, s_idx)
                mat = self.step_stats.get(key)
                if mat is None:
                    mat = np.zeros((ACTION_DIMS[s_idx], 2), dtype=np.float64)
                    self.step_stats[key] = mat
                else:
                    mat *= discount
                if res.observed_success == 1:
                    mat[int(op), 0] += 1.0
                elif res.first_failed_step == s_idx and res.failed_operator == int(op):
                    mat[int(op), 1] += 1.25
                elif res.first_failed_step is not None and s_idx < res.first_failed_step:
                    mat[int(op), 0] += 0.55

    def discount_domain_statistics(self, domain: str, factor: float = 0.15) -> None:
        for key, mat in self.step_stats.items():
            if key[0] == domain:
                mat[:, 0] *= factor
                mat[:, 1] *= 0.50

    def _estimate_step_governing_predicate(self, domain: str, step_idx: int, n_preds: int) -> Optional[int]:
        best_p: Optional[int] = None
        best_diff = 0.12
        for p_idx in range(n_preds):
            m0 = self.step_stats.get((domain, p_idx, 0, step_idx))
            m1 = self.step_stats.get((domain, p_idx, 1, step_idx))
            if m0 is not None and m1 is not None:
                n0 = m0.sum(axis=1)
                n1 = m1.sum(axis=1)
                valid = (n0 >= 2.0) & (n1 >= 2.0)
                if np.any(valid):
                    r0 = (m0[valid, 0] + 0.5) / (n0[valid] + 1.0)
                    r1 = (m1[valid, 0] + 0.5) / (n1[valid] + 1.0)
                    diff = float(np.max(np.abs(r0 - r1)))
                    if diff > best_diff:
                        best_diff = diff
                        best_p = p_idx
        return best_p

    def infer_salient_preconditions(self, task: TaskInstance) -> Tuple[Tuple[int, int], ...]:
        if not self.use_applicability:
            return ()

        n_preds = len(task.structural_predicates)
        contrasts: List[Tuple[float, int]] = []
        total_obs = 0.0

        for p_idx in range(n_preds):
            max_diff = 0.0
            for s_idx in range(len(ACTION_DIMS)):
                m0 = self.step_stats.get((task.domain, p_idx, 0, s_idx))
                m1 = self.step_stats.get((task.domain, p_idx, 1, s_idx))
                if m0 is not None and m1 is not None:
                    n0 = m0.sum(axis=1)
                    n1 = m1.sum(axis=1)
                    total_obs += float(n0.sum() + n1.sum())
                    valid = (n0 >= 2.5) & (n1 >= 2.5)
                    if np.any(valid):
                        rate0 = (m0[valid, 0] + 0.5) / (n0[valid] + 1.0)
                        rate1 = (m1[valid, 0] + 0.5) / (n1[valid] + 1.0)
                        diff = float(np.max(np.abs(rate0 - rate1)))
                        if diff > max_diff:
                            max_diff = diff
            contrasts.append((max_diff, p_idx))

        if total_obs < 80.0:
            return tuple((idx, int(task.structural_predicates[idx])) for idx in range(n_preds))

        selected = [idx for diff, idx in contrasts if diff >= 0.14]
        if len(selected) < 2:
            contrasts.sort(key=lambda x: x[0], reverse=True)
            selected = sorted(idx for _, idx in contrasts[:3])
        else:
            selected = sorted(selected)
        return tuple((idx, int(task.structural_predicates[idx])) for idx in selected)

    def synthesize_compositional_plan(
        self,
        task: TaskInstance,
        repo: StrategyRepository,
        confidence_estimator: ConfidenceEstimator,
        backbone: CalibratedSymbolicLLMBackbone,
        avoid_ops: Optional[Dict[int, List[int]]] = None,
        quarantined_plans: Optional[Sequence[Tuple[int, ...]]] = None,
    ) -> Tuple[Tuple[int, ...], int, int]:
        val_domain = repo.domain_strategies(task.domain, statuses=(StrategyStatus.VALIDATED,))
        n_preds = len(task.structural_predicates)
        step_votes: List[np.ndarray] = [np.zeros(d, dtype=np.float64) for d in ACTION_DIMS]
        merged_avoid: Dict[int, List[int]] = {
            k: list(v) for k, v in (avoid_ops.items() if avoid_ops else [])
        }

        for s_idx in range(len(ACTION_DIMS)):
            gov_p = (
                self._estimate_step_governing_predicate(task.domain, s_idx, n_preds)
                if self.use_applicability
                else None
            )

            for rec in val_domain:
                lcb = confidence_estimator.lower_credible_bound(rec)
                rec_dict = dict(rec.preconditions)
                if self.use_applicability and gov_p is not None and gov_p in rec_dict:
                    if rec_dict[gov_p] == int(task.structural_predicates[gov_p]):
                        step_votes[s_idx][int(rec.operator_plan[s_idx])] += 0.90 * lcb
                elif self.use_applicability and rec.preconditions:
                    matches = sum(1 for idx, val in rec.preconditions if task.structural_predicates[idx] == val)
                    if matches == len(rec.preconditions):
                        step_votes[s_idx][int(rec.operator_plan[s_idx])] += 0.80 * lcb
                elif not self.use_applicability:
                    sim = max(0.0, float(np.dot(rec.embedding, task.query_embedding)))
                    step_votes[s_idx][int(rec.operator_plan[s_idx])] += 0.35 * sim * lcb

            p_indices = [gov_p] if (self.use_applicability and gov_p is not None) else list(range(n_preds))
            for p_idx in p_indices:
                if p_idx is None:
                    continue
                p_val = int(task.structural_predicates[p_idx])
                mat = self.step_stats.get((task.domain, p_idx, p_val, s_idx))
                if mat is not None:
                    succ = mat[:, 0]
                    fail = mat[:, 1]
                    tot = succ + fail
                    rates = (succ + 0.4) / (tot + 1.2)
                    weight = 0.55 if (gov_p is not None and p_idx == gov_p) else 0.15
                    step_votes[s_idx] += weight * np.minimum(tot / 3.0, 1.5) * (rates - 0.35)

                    if self.use_applicability and gov_p is not None and p_idx == gov_p:
                        for op_i in range(ACTION_DIMS[s_idx]):
                            if fail[op_i] >= 2.2 and rates[op_i] < 0.22:
                                if op_i not in merged_avoid.setdefault(s_idx, []):
                                    merged_avoid[s_idx].append(op_i)

        has_strong_signal = any(float(v.max()) > 0.32 for v in step_votes)
        rec_plans: List[Tuple[Tuple[int, ...], float]] = []
        if has_strong_signal:
            comp_plan = []
            for s_idx in range(len(ACTION_DIMS)):
                scores = step_votes[s_idx].copy()
                if s_idx in merged_avoid:
                    for bad in merged_avoid[s_idx]:
                        if 0 <= bad < ACTION_DIMS[s_idx]:
                            scores[bad] -= 2.0
                if float(scores.max()) > 0.20:
                    comp_plan.append(int(np.argmax(scores)))
                else:
                    base_p = backbone._base_step_probs(task, s_idx).copy()
                    for bad in merged_avoid.get(s_idx, []):
                        if 0 <= bad < ACTION_DIMS[s_idx]:
                            base_p[bad] *= 0.05
                    base_p /= base_p.sum()
                    comp_plan.append(int(self.rng.choice(ACTION_DIMS[s_idx], p=base_p)))
            rec_plans.append((tuple(comp_plan), 0.80))

        plan, p_tok, c_tok = backbone.generate_plan(
            task,
            system_role="evsa_hypothesis_generator",
            recommended_plans=rec_plans if rec_plans else None,
            avoid_operators=merged_avoid if merged_avoid else None,
        )

        if quarantined_plans and plan in quarantined_plans:
            mut_list = list(plan)
            s_mut = int(self.rng.integers(0, len(ACTION_DIMS)))
            alts = [a for a in range(ACTION_DIMS[s_mut]) if a != mut_list[s_mut]]
            if avoid_ops and s_mut in avoid_ops:
                filtered = [a for a in alts if a not in avoid_ops[s_mut]]
                if filtered:
                    alts = filtered
            mut_list[s_mut] = int(self.rng.choice(alts))
            plan = tuple(mut_list)

        return plan, p_tok, c_tok
