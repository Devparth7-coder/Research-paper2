"""
Proposed Evidence-Validated Strategy Adaptation Agent (EVSAAgent).

Integrates all ten EVSA modules under equivalent rollout budgets (B = 2 during
adaptation, B = 1 during held-out evaluation), supports five leave-one-out
ablations, and includes an optional stochastic-prompt execution mode for
controlled fairness sensitivity analysis.

Strict Zero-Oracle Guarantee:
No method on `EVSAAgent` ever inspects ground-truth workflow plans, latent
archetypes, hidden regime IDs, or uncorrupted rollout outcomes.
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional
import numpy as np

from ..benchmark.agent_interface import AgentEnvironmentInterface
from ..benchmark.types import AgentEpisodeOutcome, RolloutResult, StrategyStatus, TaskInstance
from ..drift_detection.cusum_detector import DistributionShiftDetector
from ..memory.hypothesis_generator import HypothesisGenerator
from ..memory.memory_manager import AdaptiveMemoryManager
from ..memory.repository import StrategyRepository
from ..retrieval.retriever import StrategyRetriever
from ..validation.confidence import ConfidenceEstimator
from ..validation.sprt_validator import EvidenceValidationEngine
from .baselines import BaseExperimentalAgent, _require_agent_interface
from .experimentation import ExperimentationEngine


class EVSAAgent(BaseExperimentalAgent):
    """Proposed Agent E: Evidence-Validated Strategy Adaptation (EVSA)."""

    def __init__(
        self,
        seed: int = 42,
        name: str = "Proposed Agent E: EVSA",
        use_evidence_validation: bool = True,
        use_confidence_estimation: bool = True,
        use_applicability_conditions: bool = True,
        use_shift_detection: bool = True,
        use_memory_invalidation: bool = True,
        min_validation_trials: int = 3,
        lcb_delta: float = 0.10,
        tau_lcb: float = 0.48,
        tau_rel: float = 0.45,
        stochastic_prompt_execution: bool = False,
    ) -> None:
        super().__init__(seed=seed)
        self.name = name
        self.use_evidence_validation = use_evidence_validation
        self.use_confidence_estimation = use_confidence_estimation
        self.use_applicability_conditions = use_applicability_conditions
        self.use_shift_detection = use_shift_detection
        self.use_memory_invalidation = use_memory_invalidation
        self.stochastic_prompt_execution = stochastic_prompt_execution

        self.repo = StrategyRepository(max_active_per_domain=48)
        self.confidence = ConfidenceEstimator(
            credible_delta=lcb_delta,
            enabled=use_confidence_estimation,
        )
        self.validator = EvidenceValidationEngine(
            min_trials=min_validation_trials,
            tau_lcb=tau_lcb,
            validation_enabled=use_evidence_validation,
            invalidation_enabled=use_memory_invalidation,
        )
        self.retriever = StrategyRetriever(
            confidence_estimator=self.confidence,
            use_applicability=use_applicability_conditions,
            tau_rel=tau_rel,
        )
        self.hypothesis_gen = HypothesisGenerator(
            use_applicability=use_applicability_conditions,
            seed=seed + 17,
        )
        self.memory_mgr = AdaptiveMemoryManager(
            confidence_estimator=self.confidence,
            use_applicability=use_applicability_conditions,
            invalidation_enabled=use_memory_invalidation,
        )
        self.shift_detector = DistributionShiftDetector(
            enabled=use_shift_detection,
        )
        self.exp_engine = ExperimentationEngine()

    def act_on_episode(
        self,
        task: TaskInstance,
        env: AgentEnvironmentInterface,
        max_trials: int = 2,
        learn: bool = True,
    ) -> AgentEpisodeOutcome:
        _require_agent_interface(env)
        t0 = time.perf_counter()
        p_tok_ep, c_tok_ep = 0, 0
        total_evals = 0
        shift_fired_ep = False

        local_avoid: Dict[int, List[int]] = {}
        tried_strategy_ids: set[str] = set()
        trials_used = 0
        best_res: Optional[RolloutResult] = None
        best_trial_idx = 0

        while trials_used < max_trials:
            trial_zero_idx = trials_used
            trials_used += 1
            val_matches, cand_matches, n_eval = self.retriever.retrieve(self.repo, task)
            total_evals += n_eval

            val_avail = [(u, r) for u, r in val_matches if r.strategy_id not in tried_strategy_ids]
            cand_avail = [(u, r) for u, r in cand_matches if r.strategy_id not in tried_strategy_ids]

            if learn and self.retriever.detect_contradictions(val_avail):
                self.memory_mgr.resolve_contradictions(self.repo, val_avail, task.episode_idx)
                cp, cc = self.backbone.account_auxiliary_call(
                    "contradiction_check",
                    f"Domain={task.domain} Candidates={len(val_avail)}",
                    "ResolvedStrategyConflict",
                )
                p_tok_ep += cp
                c_tok_ep += cc

            chosen_rec, used_validated = self.exp_engine.select_strategy_for_trial(
                val_avail, cand_avail, trials_used, learn=learn
            )

            if chosen_rec is not None:
                tried_strategy_ids.add(chosen_rec.strategy_id)
                if self.stochastic_prompt_execution:
                    w_guide = 0.95 if used_validated else 0.80
                    plan, rp, rc = self.backbone.generate_plan(
                        task,
                        system_role="evsa_prompt_conditioned",
                        recommended_plans=[(chosen_rec.operator_plan, w_guide)],
                        avoid_operators=local_avoid if local_avoid else None,
                    )
                else:
                    plan = chosen_rec.operator_plan
                    rp, rc = self.backbone.account_auxiliary_call(
                        "evsa_execute_strategy",
                        f"Task={task.domain} Pred={task.structural_predicates} "
                        f"Strategy={chosen_rec.strategy_id} Status={chosen_rec.status.value} Plan={list(plan)}",
                        f"ActionPlan={list(plan)}",
                    )
                p_tok_ep += rp
                c_tok_ep += rc
            else:
                quar_matching = [
                    r.operator_plan
                    for r in self.repo.domain_strategies(task.domain, statuses=(StrategyStatus.QUARANTINED,))
                    if (
                        not self.use_applicability_conditions
                        or r.matches_preconditions(task.structural_predicates)
                    )
                ]
                plan, gp, gc = self.hypothesis_gen.synthesize_compositional_plan(
                    task=task,
                    repo=self.repo,
                    confidence_estimator=self.confidence,
                    backbone=self.backbone,
                    avoid_ops=local_avoid if local_avoid else None,
                    quarantined_plans=quar_matching,
                )
                p_tok_ep += gp
                c_tok_ep += gc

            res = env.evaluate_plan(task, plan)
            if best_res is None or res.observed_reward > best_res.observed_reward:
                best_res = res
                best_trial_idx = trial_zero_idx

            if learn:
                self.hypothesis_gen.record_observation(task, res)

                if chosen_rec is not None:
                    # 1. Compute pre-outcome predictive probability BEFORE updating posterior
                    pred_p = self.confidence.posterior_mean(chosen_rec)

                    # 2. Update posterior and sequential validation state first so
                    #    `consecutive_failures`, `alpha`, `beta`, and `sprt_llr` reflect this trial
                    prev_status = chosen_rec.status
                    self.confidence.update_posterior(chosen_rec, res.observed_success, task.episode_idx)
                    if res.observed_success == 1:
                        emb_mix = 0.8 * chosen_rec.embedding + 0.2 * task.query_embedding
                        chosen_rec.embedding = emb_mix / (np.linalg.norm(emb_mix) + 1e-12)
                    new_status = self.validator.step_evidence(
                        chosen_rec, res.observed_success, self.confidence, task.episode_idx
                    )
                    self.memory_mgr.record_transition_counters(
                        self.repo, prev_status, new_status, chosen_rec
                    )

                    # 3. Feed out-of-sample residual to CUSUM detector if a VALIDATED strategy ran;
                    #    if alarm fires, selective quarantine inspects updated failure/LCB state cleanly.
                    if used_validated:
                        drift = self.shift_detector.update(
                            task.domain, pred_p, res.observed_success, task.episode_idx
                        )
                        if drift:
                            shift_fired_ep = True
                            self.memory_mgr.apply_domain_shift_discount(
                                self.repo,
                                domain=task.domain,
                                gamma_shift=self.shift_detector.shift_discount_gamma,
                                tau_lcb=self.validator.tau_lcb,
                                episode_idx=task.episode_idx,
                            )
                            self.hypothesis_gen.discount_domain_statistics(task.domain, factor=0.25)

                elif res.observed_success == 1:
                    preconds = self.hypothesis_gen.infer_salient_preconditions(task)
                    hp, hc = self.backbone.account_auxiliary_call(
                        "evsa_induce_hypothesis",
                        f"Task={task.domain} Pred={task.structural_predicates} Plan={list(res.executed_plan)}",
                        f"CandidatePreconds={list(preconds)} Plan={list(res.executed_plan)}",
                    )
                    p_tok_ep += hp
                    c_tok_ep += hc
                    self.memory_mgr.admit_or_merge_candidate(
                        repo=self.repo,
                        task=task,
                        preconditions=preconds,
                        plan=res.executed_plan,
                        observed_success=1,
                        episode_idx=task.episode_idx,
                        validation_engine=self.validator,
                    )

            if res.observed_success == 1:
                best_res = res
                best_trial_idx = trial_zero_idx
                break
            else:
                if res.first_failed_step is not None and res.failed_operator is not None:
                    local_avoid.setdefault(res.first_failed_step, []).append(res.failed_operator)

        if learn:
            self.memory_mgr.prune_capacity(self.repo, task.domain)

        dt_ms = (time.perf_counter() - t0) * 1000.0
        assert best_res is not None
        return AgentEpisodeOutcome(
            best_trial_idx=best_trial_idx,
            trials_used=trials_used,
            strategy_evaluations=total_evals,
            execution_time_ms=dt_ms,
            active_memory_items=len(self.repo.active_strategies()),
            memory_bytes=self.repo.memory_bytes(),
            prompt_tokens=p_tok_ep,
            completion_tokens=c_tok_ep,
            shift_detected=shift_fired_ep,
        )

    def active_memory_count(self) -> int:
        return len(self.repo.active_strategies())

    def validated_memory_count(self) -> int:
        return len(self.repo.validated_strategies())

    def memory_footprint_bytes(self) -> int:
        return self.repo.memory_bytes()
