"""
Adaptive Memory Manager (Module 8).

Implements:
1. Safe Least-General-Generalization (LGG; Plotkin, 1970) precondition
   intersection (`K_min = 3`): when LGG expands the precondition domain of an
   already-VALIDATED strategy, its sequential log-odds are conservatively
   clamped below `upper_bound` and the strategy steps back to CANDIDATE until
   at least one independent trial confirms reliability on the broader subspace.
2. Evidence-based selective post-shift quarantine (`consecutive_failures >= 1`
   or `lcb < tau_quarantine_shift`), replacing blind age-based wipeouts.
3. Contradiction resolution and bounded per-domain capacity pruning.
4. Strict zero-oracle isolation (zero access to ground-truth plans or regimes).
"""

from __future__ import annotations

from typing import Sequence, Tuple
import numpy as np

from ..benchmark.types import StrategyRecord, StrategyStatus, TaskInstance
from ..validation.confidence import ConfidenceEstimator
from ..validation.sprt_validator import EvidenceValidationEngine
from .repository import StrategyRepository, make_strategy_id


class AdaptiveMemoryManager:
    """Lifecycle manager for candidate admission, safe LGG generalization, and quarantine."""

    def __init__(
        self,
        confidence_estimator: ConfidenceEstimator,
        use_applicability: bool = True,
        invalidation_enabled: bool = True,
        min_lgg_literals: int = 3,
        lgg_discount_gamma: float = 0.80,
        tau_quarantine_shift: float = 0.58,
    ) -> None:
        self.confidence_estimator = confidence_estimator
        self.use_applicability = use_applicability
        self.invalidation_enabled = invalidation_enabled
        self.min_lgg_literals = min_lgg_literals
        self.lgg_discount_gamma = lgg_discount_gamma
        self.tau_quarantine_shift = tau_quarantine_shift

    def record_transition_counters(
        self,
        repo: StrategyRepository,
        prev_status: StrategyStatus,
        new_status: StrategyStatus,
        record: StrategyRecord | None = None,
    ) -> None:
        if prev_status != StrategyStatus.VALIDATED and new_status == StrategyStatus.VALIDATED:
            repo.total_promoted += 1
        if prev_status != StrategyStatus.QUARANTINED and new_status == StrategyStatus.QUARANTINED:
            repo.total_quarantined += 1
            if record is not None:
                record.quarantine_count += 1

    def admit_or_merge_candidate(
        self,
        repo: StrategyRepository,
        task: TaskInstance,
        preconditions: Tuple[Tuple[int, int], ...],
        plan: Tuple[int, ...],
        observed_success: int,
        episode_idx: int,
        validation_engine: EvidenceValidationEngine,
    ) -> StrategyRecord:
        sid = make_strategy_id(task.domain, preconditions, plan)
        existing = repo.get(sid)
        if existing is not None and existing.status != StrategyStatus.QUARANTINED:
            prev_status = existing.status
            self.confidence_estimator.update_posterior(existing, observed_success, episode_idx)
            new_status = validation_engine.step_evidence(
                existing, observed_success, self.confidence_estimator, episode_idx
            )
            self.record_transition_counters(repo, prev_status, new_status, existing)
            return existing

        # Tombstone policy: a QUARANTINED record is never resurrected by non-positive evidence.
        if existing is not None and existing.status == StrategyStatus.QUARANTINED and int(observed_success) != 1:
            existing.last_updated_episode = episode_idx
            return existing

        if self.use_applicability and int(observed_success) == 1:
            pred_dict = dict(preconditions)
            for peer in repo.domain_strategies(
                task.domain, statuses=(StrategyStatus.CANDIDATE, StrategyStatus.VALIDATED)
            ):
                if peer.operator_plan != plan:
                    continue
                peer_dict = dict(peer.preconditions)
                shared_keys = set(pred_dict.keys()) & set(peer_dict.keys())
                diffs = [k for k in shared_keys if pred_dict[k] != peer_dict[k]]
                lgg_preconds = tuple(
                    sorted((k, pred_dict[k]) for k in shared_keys if pred_dict[k] == peer_dict[k])
                )
                broadened = len(lgg_preconds) < len(peer.preconditions)
                if 1 <= len(diffs) <= 2 and len(lgg_preconds) >= self.min_lgg_literals:
                    new_sid = make_strategy_id(task.domain, lgg_preconds, plan)
                    target_existing = repo.get(new_sid)
                    was_validated = peer.status == StrategyStatus.VALIDATED
                    prior_invalidated_ep = (
                        target_existing.invalidated_episode
                        if (target_existing is not None and target_existing.status == StrategyStatus.QUARANTINED)
                        else peer.invalidated_episode
                    )
                    prior_quarantine_cnt = (
                        peer.quarantine_count
                        + (
                            target_existing.quarantine_count
                            if (target_existing is not None and target_existing.status == StrategyStatus.QUARANTINED)
                            else 0
                        )
                    )
                    if peer.strategy_id in repo.records:
                        del repo.records[peer.strategy_id]
                    peer.strategy_id = new_sid
                    peer.preconditions = lgg_preconds
                    peer.invalidated_episode = prior_invalidated_ep
                    peer.quarantine_count = prior_quarantine_cnt
                    peer.lgg_merge_count += 1
                    repo.lgg_merges += 1
                    emb_mix = 0.65 * peer.embedding + 0.35 * task.query_embedding
                    peer.embedding = emb_mix / (np.linalg.norm(emb_mix) + 1e-12)
                    repo.add(peer)

                    self.confidence_estimator.update_posterior(peer, observed_success, episode_idx)

                    if (
                        (was_validated and broadened)
                        or (target_existing is not None and target_existing.status == StrategyStatus.QUARANTINED)
                    ) and validation_engine.validation_enabled:
                        # Safe LGG / tombstone-broadened merge: discount excess pseudo-counts
                        # and require confirming validation on the broadened subspace.
                        peer.alpha = 1.0 + self.lgg_discount_gamma * max(0.0, peer.alpha - 1.0)
                        peer.beta = 1.0 + self.lgg_discount_gamma * max(0.0, peer.beta - 1.0)
                        peer.status = StrategyStatus.CANDIDATE
                        peer.sprt_llr = min(
                            self.lgg_discount_gamma * peer.sprt_llr,
                            validation_engine.upper_bound - 0.15,
                        )
                        peer.last_updated_episode = episode_idx
                        repo.lgg_revalidations_triggered += 1
                        return peer

                    prev_status = peer.status
                    new_status = validation_engine.step_evidence(
                        peer, observed_success, self.confidence_estimator, episode_idx
                    )
                    self.record_transition_counters(repo, prev_status, new_status, peer)
                    return peer

        # Probationary in-place re-admission when new positive evidence arrives for a
        # previously QUARANTINED deterministic strategy ID: preserves quarantine history
        # (`invalidated_episode`, `quarantine_count`) and avoids duplicate record objects
        # while requiring a fresh full multi-trial sequential validation cycle.
        if existing is not None and existing.status == StrategyStatus.QUARANTINED:
            repo.candidates_created += 1
            existing.embedding = task.query_embedding.copy()
            existing.alpha = 1.0
            existing.beta = 1.0
            existing.n_trials = 0
            existing.n_successes = 0
            existing.sprt_llr = 0.0
            existing.consecutive_failures = 0
            existing.status = StrategyStatus.CANDIDATE
            existing.last_updated_episode = episode_idx
            existing.created_task_id = task.task_id
            self.confidence_estimator.update_posterior(existing, observed_success, episode_idx)
            prev_status = existing.status
            new_status = validation_engine.step_evidence(
                existing, observed_success, self.confidence_estimator, episode_idx
            )
            self.record_transition_counters(repo, prev_status, new_status, existing)
            return existing

        repo.candidates_created += 1
        rec = StrategyRecord(
            strategy_id=sid,
            domain=task.domain,
            preconditions=preconditions,
            embedding=task.query_embedding.copy(),
            operator_plan=plan,
            alpha=1.0,
            beta=1.0,
            n_trials=0,
            n_successes=0,
            sprt_llr=0.0,
            consecutive_failures=0,
            status=StrategyStatus.CANDIDATE,
            created_episode=episode_idx,
            last_updated_episode=episode_idx,
            created_task_id=task.task_id,
        )
        self.confidence_estimator.update_posterior(rec, observed_success, episode_idx)
        prev_status = rec.status
        new_status = validation_engine.step_evidence(
            rec, observed_success, self.confidence_estimator, episode_idx
        )
        repo.add(rec)
        self.record_transition_counters(repo, prev_status, new_status, rec)
        return rec

    def apply_domain_shift_discount(
        self,
        repo: StrategyRepository,
        domain: str,
        gamma_shift: float,
        tau_lcb: float,
        episode_idx: int,
    ) -> int:
        """
        Evidence-based selective quarantine upon a per-domain CUSUM shift alarm:
        - Immediately QUARANTINE strategies in `domain` exhibiting empirical failure
          (`consecutive_failures >= 1`) or weakened reliability (`lcb < tau_quarantine_shift`).
        - Discount historical pseudo-counts of remaining unfalsified strategies in
          `domain` and demote VALIDATED strategies to CANDIDATE for re-verification.
        """
        _ = tau_lcb
        if not self.invalidation_enabled:
            return 0
        affected = 0
        for rec in repo.domain_strategies(
            domain, statuses=(StrategyStatus.VALIDATED, StrategyStatus.CANDIDATE)
        ):
            lcb = self.confidence_estimator.lower_credible_bound(rec)
            if rec.consecutive_failures >= 1 or lcb < self.tau_quarantine_shift:
                rec.status = StrategyStatus.QUARANTINED
                rec.invalidated_episode = episode_idx
                rec.quarantine_count += 1
                repo.total_quarantined += 1
                affected += 1
            else:
                rec.alpha = 1.0 + gamma_shift * max(0.0, rec.alpha - 1.0)
                rec.beta = 1.0 + gamma_shift * max(0.0, rec.beta - 1.0)
                rec.sprt_llr = min(0.50, rec.sprt_llr * gamma_shift)
                if rec.status == StrategyStatus.VALIDATED:
                    rec.status = StrategyStatus.CANDIDATE
                    affected += 1
        return affected

    def resolve_contradictions(
        self,
        repo: StrategyRepository,
        validated_matches: Sequence[Tuple[float, StrategyRecord]],
        episode_idx: int,
    ) -> None:
        if not self.invalidation_enabled or len(validated_matches) < 2:
            return
        leader = validated_matches[0][1]
        leader_lcb = self.confidence_estimator.lower_credible_bound(leader)
        for _, challenger in validated_matches[1:]:
            if challenger.operator_plan != leader.operator_plan:
                chal_lcb = self.confidence_estimator.lower_credible_bound(challenger)
                if leader_lcb - chal_lcb > 0.14:
                    challenger.status = StrategyStatus.CANDIDATE
                    challenger.last_updated_episode = episode_idx
                elif challenger.consecutive_failures > leader.consecutive_failures:
                    challenger.status = StrategyStatus.QUARANTINED
                    challenger.invalidated_episode = episode_idx
                    challenger.quarantine_count += 1
                    repo.total_quarantined += 1

    def prune_capacity(self, repo: StrategyRepository, domain: str) -> None:
        active = repo.domain_strategies(
            domain, statuses=(StrategyStatus.VALIDATED, StrategyStatus.CANDIDATE)
        )
        if len(active) <= repo.max_active_per_domain:
            return
        scored = sorted(
            active,
            key=lambda r: (
                1 if r.status == StrategyStatus.VALIDATED else 0,
                self.confidence_estimator.lower_credible_bound(r),
                r.last_updated_episode,
            ),
        )
        to_remove = len(active) - repo.max_active_per_domain
        for r in scored[:to_remove]:
            if r.status == StrategyStatus.CANDIDATE:
                r.status = StrategyStatus.QUARANTINED
                r.quarantine_count += 1
                repo.total_quarantined += 1
