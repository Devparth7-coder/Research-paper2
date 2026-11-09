"""
ShiftBench-Sim v1.0: Non-Stationary Multi-Domain Procedural Task Environment.

Models sequential 4-step workflow decision-making across four domains with
governing structural preconditions, nuisance binary attributes, high-weight
surface topic distractors, stochastic outcomes, concept & covariate shifts,
and controllable evaluator feedback corruption.

Strict Oracle Isolation:
All ground-truth workflow mappings, latent archetype IDs, regime indices, and
uncorrupted rollout outcomes are stored exclusively inside private environment
audit tables (`_task_meta` and `_rollout_audit_log`), never on `TaskInstance`
or `RolloutResult`.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np

from .agent_interface import AgentEnvironmentInterface
from .types import (
    AuditOutcome,
    EpisodeTelemetry,
    MemoryEntry,
    RolloutResult,
    StrategyRecord,
    TaskInstance,
)

DOMAINS: Tuple[str, ...] = (
    "api_orchestration",
    "system_diagnosis",
    "data_pipeline",
    "resource_planning",
)

ACTION_DIMS: Tuple[int, ...] = (4, 5, 4, 3)  # H = 4 steps, 240 workflows/domain
NUM_PREDICATES: int = 5                      # 3 governing + 2 nuisance predicates
NUM_TOPICS: int = 8                          # Surface distractor scenarios


class NonStationaryTaskEnvironment:
    """Executable procedural MDP benchmark with controllable distribution shifts."""

    def __init__(
        self,
        seed: int = 42,
        p_opt: float = 0.92,
        p_sub3: float = 0.28,
        p_sub2: float = 0.09,
        p_sub1: float = 0.02,
        feedback_corruption_rate: float = 0.0,
    ) -> None:
        self.base_seed = seed
        self.rng = np.random.default_rng(seed)
        self.p_opt = p_opt
        self.p_sub3 = p_sub3
        self.p_sub2 = p_sub2
        self.p_sub1 = p_sub1
        self.feedback_corruption_rate = feedback_corruption_rate

        topic_rng = np.random.default_rng(20261107)
        raw_topics = topic_rng.normal(0.0, 1.0, size=(NUM_TOPICS, 10))
        norms = np.linalg.norm(raw_topics, axis=1, keepdims=True) + 1e-12
        self.topic_vectors = raw_topics / norms

        self.regime_tables: Dict[int, Dict[Tuple[str, int], Tuple[int, ...]]] = {}
        self._build_regime_tables()

        # Private oracle lookup: task_id -> (regime_id, governing_archetype, optimal_plan)
        self._task_meta: Dict[str, Tuple[int, int, Tuple[int, ...]]] = {}
        self._rollout_audit_log: List[AuditOutcome] = []

    def _build_regime_tables(self) -> None:
        """Construct structured compositional operator mappings for Regimes 0, 1, and 2."""
        table_0: Dict[Tuple[str, int], Tuple[int, ...]] = {}
        table_1: Dict[Tuple[str, int], Tuple[int, ...]] = {}
        table_2: Dict[Tuple[str, int], Tuple[int, ...]] = {}

        for d_idx, domain in enumerate(DOMAINS):
            s0_map = [int((d_idx + b * 2) % ACTION_DIMS[0]) for b in (0, 1)]
            s1_map = [int((d_idx * 2 + b * 3 + 1) % ACTION_DIMS[1]) for b in (0, 1)]
            s2_map = [int((d_idx + b * 2 + 1) % ACTION_DIMS[2]) for b in (0, 1)]
            s3_map = [int((d_idx + x) % ACTION_DIMS[3]) for x in (0, 1)]

            s0_shift2 = [int((v + 2) % ACTION_DIMS[0]) for v in s0_map]
            s1_shift1 = [int((v + 2) % ACTION_DIMS[1]) for v in s1_map]
            s2_shift1 = [int((v + 2) % ACTION_DIMS[2]) for v in s2_map]
            s2_shift2 = [int((v + 1) % ACTION_DIMS[2]) for v in s2_map]

            for arch in range(8):
                b0 = (arch >> 2) & 1
                b1 = (arch >> 1) & 1
                b2 = arch & 1
                bxor = b0 ^ b1

                plan_0 = (s0_map[b0], s1_map[b1], s2_map[b2], s3_map[bxor])
                if domain != "resource_planning":
                    plan_1 = (s0_map[b0], s1_shift1[b1], s2_shift1[b2], s3_map[bxor])
                else:
                    plan_1 = plan_0

                plan_2 = (
                    s0_shift2[b0],
                    s1_shift1[b1] if domain != "resource_planning" else s1_map[b1],
                    s2_shift2[b2],
                    s3_map[bxor],
                )

                table_0[(domain, arch)] = plan_0
                table_1[(domain, arch)] = plan_1
                table_2[(domain, arch)] = plan_2

        self.regime_tables[0] = table_0
        self.regime_tables[1] = table_1
        self.regime_tables[2] = table_2

    def get_optimal_plan(self, domain: str, archetype: int, regime_id: int = 0) -> Tuple[int, ...]:
        """Return ground-truth optimal H-step plan for post-hoc evaluation auditing only."""
        return self.regime_tables[regime_id][(domain, archetype)]

    def register_task_metadata(
        self,
        task_id: str,
        regime_id: int,
        governing_archetype: int,
        optimal_plan: Tuple[int, ...],
    ) -> None:
        """Register private oracle metadata for a task stream in an execution environment."""
        self._task_meta[task_id] = (regime_id, governing_archetype, optimal_plan)

    def copy_metadata_from(self, other: "NonStationaryTaskEnvironment") -> None:
        """Copy private oracle task registry from a stream-generator environment."""
        self._task_meta.update(other._task_meta)

    def get_task_oracle(self, task_id: str) -> Tuple[int, int, Tuple[int, ...]]:
        """Retrieve (regime_id, governing_archetype, optimal_plan) for post-hoc evaluation."""
        return self._task_meta[task_id]

    def resolve_oracle_for_task(self, task: TaskInstance, default_regime: int = 0) -> Tuple[int, int, Tuple[int, ...]]:
        """Resolve private oracle metadata for `task` without storing it on `TaskInstance`."""
        if task.task_id in self._task_meta:
            return self._task_meta[task.task_id]
        arch = (
            (int(task.structural_predicates[0]) << 2)
            | (int(task.structural_predicates[1]) << 1)
            | int(task.structural_predicates[2])
        )
        opt = self.get_optimal_plan(task.domain, arch, regime_id=default_regime)
        self._task_meta[task.task_id] = (default_regime, arch, opt)
        return default_regime, arch, opt

    def pop_episode_audits(self) -> List[AuditOutcome]:
        """Drain and return AuditOutcome records accumulated during the just-completed episode."""
        out = list(self._rollout_audit_log)
        self._rollout_audit_log.clear()
        return out

    def make_agent_interface(
        self,
        corruption_override: Optional[float] = None,
        rollout_observer=None,
    ) -> AgentEnvironmentInterface:
        """
        Construct a capability-isolated `AgentEnvironmentInterface` handle that exposes
        ONLY `evaluate_plan(task, plan) -> RolloutResult` to agents without exposing
        `self` or any evaluator-only oracle methods/tables.
        """
        def _step(task_obj: TaskInstance, plan_tuple: Tuple[int, ...]) -> RolloutResult:
            res = self.evaluate_plan(task_obj, plan_tuple, corruption_override=corruption_override)
            if rollout_observer is not None:
                rollout_observer(task_obj, plan_tuple, res)
            return res

        return AgentEnvironmentInterface(_step)

    def _make_query_embedding(
        self,
        domain: str,
        predicates: Tuple[int, ...],
        topic_id: int,
        rng: np.random.Generator,
    ) -> Tuple[np.ndarray, np.ndarray]:
        d_idx = DOMAINS.index(domain)
        dom_vec = np.zeros(4, dtype=np.float64)
        dom_vec[d_idx] = 1.0

        pred_signed = np.array([2.0 * p - 1.0 for p in predicates], dtype=np.float64)
        inter = np.array([pred_signed[0] * pred_signed[1]], dtype=np.float64)
        struct_vec = np.concatenate([pred_signed, inter]) + rng.normal(0.0, 0.25, size=6)

        topic_vec = self.topic_vectors[topic_id] + rng.normal(0.0, 0.08, size=10)
        context_features = np.concatenate([dom_vec, struct_vec, topic_vec[:2]])

        raw_emb = np.concatenate([0.50 * dom_vec, 0.36 * struct_vec, 0.82 * topic_vec])
        emb_norm = raw_emb / (np.linalg.norm(raw_emb) + 1e-12)
        return context_features, emb_norm

    def sample_task_stream(
        self,
        num_episodes: int,
        regime_schedule: Optional[Sequence[int]] = None,
        allowed_archetypes: Optional[Sequence[int]] = None,
        covariate_shift_at: Optional[int] = None,
        unseen_flag: bool = False,
        stream_seed: Optional[int] = None,
        stream_tag: str = "s",
    ) -> List[TaskInstance]:
        used_seed = self.base_seed if stream_seed is None else stream_seed
        rng = np.random.default_rng(used_seed)
        archs = list(range(8)) if allowed_archetypes is None else list(allowed_archetypes)
        tasks: List[TaskInstance] = []

        for ep in range(num_episodes):
            regime_id = 0 if regime_schedule is None else int(regime_schedule[ep])
            domain = DOMAINS[int(rng.integers(0, len(DOMAINS)))]

            if covariate_shift_at is not None and ep < covariate_shift_at:
                if rng.random() < 0.85:
                    arch = int(rng.choice([a for a in archs if a < 4] or archs))
                else:
                    arch = int(rng.choice(archs))
                topic_id = int(rng.integers(0, 4))
            elif covariate_shift_at is not None and ep >= covariate_shift_at:
                if rng.random() < 0.85:
                    arch = int(rng.choice([a for a in archs if a >= 4] or archs))
                else:
                    arch = int(rng.choice(archs))
                topic_id = int(rng.integers(0, 4))
            else:
                arch = int(rng.choice(archs))
                topic_id = int(rng.integers(0, NUM_TOPICS))

            b0 = (arch >> 2) & 1
            b1 = (arch >> 1) & 1
            b2 = arch & 1
            nuis0 = int(rng.integers(0, 2))
            nuis1 = int(rng.integers(0, 2))
            predicates = (b0, b1, b2, nuis0, nuis1)

            ctx_feat, q_emb = self._make_query_embedding(domain, predicates, topic_id, rng)
            opt_plan = self.get_optimal_plan(domain, arch, regime_id=regime_id)

            # Observable task_id encodes NO regime_id and NO archetype index
            tid = f"task_{stream_tag}_{used_seed}_{ep:05d}_{domain[:3]}"
            self.register_task_metadata(tid, regime_id, arch, opt_plan)

            tasks.append(
                TaskInstance(
                    task_id=tid,
                    episode_idx=ep,
                    domain=domain,
                    structural_predicates=predicates,
                    surface_topic_id=topic_id,
                    context_features=ctx_feat,
                    query_embedding=q_emb,
                    is_unseen_variant=unseen_flag or (arch == 7),
                )
            )
        return tasks

    def evaluate_plan(
        self,
        task: TaskInstance,
        executed_plan: Tuple[int, ...],
        corruption_override: Optional[float] = None,
    ) -> RolloutResult:
        regime_id, arch, opt = self.resolve_oracle_for_task(task)
        matches = tuple(int(a) == int(b) for a, b in zip(executed_plan, opt))
        n_match = sum(matches)
        alignment = n_match / float(len(ACTION_DIMS))

        first_fail: Optional[int] = None
        failed_op: Optional[int] = None
        for idx, ok in enumerate(matches):
            if not ok:
                first_fail = idx
                failed_op = int(executed_plan[idx])
                break

        if n_match == 4:
            p_succ = self.p_opt
        elif n_match == 3:
            p_succ = self.p_sub3
        elif n_match == 2:
            p_succ = self.p_sub2
        else:
            p_succ = self.p_sub1

        true_success = int(self.rng.random() < p_succ)
        true_reward = float(np.clip(0.35 * alignment + 0.65 * true_success, 0.0, 1.0))

        eta = self.feedback_corruption_rate if corruption_override is None else corruption_override
        corrupted = bool(eta > 0.0 and (self.rng.random() < eta))

        if not corrupted:
            obs_success = true_success
            obs_reward = true_reward
            obs_first_fail = first_fail
            obs_failed_op = failed_op
        else:
            obs_success = 1 - true_success
            if obs_success == 1:
                obs_reward = float(self.rng.uniform(0.82, 0.97))
                obs_first_fail = None
                obs_failed_op = None
            else:
                obs_reward = float(self.rng.uniform(0.08, 0.32))
                obs_first_fail = int(self.rng.integers(0, len(ACTION_DIMS)))
                obs_failed_op = int(executed_plan[obs_first_fail])

        plan_clean = tuple(int(x) for x in executed_plan)
        self._rollout_audit_log.append(
            AuditOutcome(
                task_id=task.task_id,
                domain=task.domain,
                regime_id=regime_id,
                governing_archetype=arch,
                optimal_plan=opt,
                executed_plan=plan_clean,
                alignment_score=alignment,
                true_success=true_success,
                true_reward=true_reward,
                observed_success=obs_success,
                observed_reward=obs_reward,
                feedback_corrupted=corrupted,
            )
        )

        return RolloutResult(
            executed_plan=plan_clean,
            observed_success=obs_success,
            observed_reward=obs_reward,
            first_failed_step=obs_first_fail,
            failed_operator=obs_failed_op,
        )

    def _audit_evsa_records(self, validated_records: Sequence[StrategyRecord], regime_id: int) -> float:
        if not validated_records:
            return 0.0
        bad = 0
        for rec in validated_records:
            is_bad = False
            checked = 0
            for arch in range(8):
                b0 = (arch >> 2) & 1
                b1 = (arch >> 1) & 1
                b2 = arch & 1
                ok = True
                for idx, req in rec.preconditions:
                    if idx == 0 and b0 != req:
                        ok = False
                    elif idx == 1 and b1 != req:
                        ok = False
                    elif idx == 2 and b2 != req:
                        ok = False
                if ok:
                    checked += 1
                    if rec.operator_plan != self.get_optimal_plan(rec.domain, arch, regime_id=regime_id):
                        is_bad = True
                        break
            if checked == 0 or is_bad:
                bad += 1
        return bad / float(len(validated_records))

    def _audit_memory_entries(
        self,
        entries: Sequence[MemoryEntry],
        regime_id: int,
        include_reflection_avoid: bool = False,
    ) -> float:
        if not include_reflection_avoid:
            pos_entries = [e for e in entries if e.observed_success == 1]
            if not pos_entries:
                return 0.0
            bad = 0
            for e in pos_entries:
                arch = (int(e.predicates[0]) << 2) | (int(e.predicates[1]) << 1) | int(e.predicates[2])
                opt = self.get_optimal_plan(e.domain, arch, regime_id=regime_id)
                if e.executed_plan != opt:
                    bad += 1
            return bad / float(len(pos_entries))

        if not entries:
            return 0.0
        pos_ratio = self._audit_memory_entries(entries, regime_id, include_reflection_avoid=False)
        bad_refs = 0
        for m in entries:
            arch = (int(m.predicates[0]) << 2) | (int(m.predicates[1]) << 1) | int(m.predicates[2])
            opt = self.get_optimal_plan(m.domain, arch, regime_id=regime_id)
            if m.observed_success == 1 and m.executed_plan != opt:
                bad_refs += 1
            elif any(opt[s] in ops for s, ops in m.reflection_avoid_map.items()):
                bad_refs += 1
        return max(pos_ratio, bad_refs / float(len(entries)))

    def finalize_episode_telemetry(
        self,
        experiment_name: str,
        agent_name: str,
        seed: int,
        task: TaskInstance,
        best_trial_idx: int,
        trials_used: int,
        strategy_evaluations: int,
        execution_time_ms: float,
        active_memory_items: int,
        memory_bytes: int,
        prompt_tokens: int,
        completion_tokens: int,
        shift_detected: bool = False,
        validated_strategies: Optional[Sequence[StrategyRecord]] = None,
        memory_entries: Optional[Sequence[MemoryEntry]] = None,
        include_reflection_avoid: bool = False,
    ) -> EpisodeTelemetry:
        """Assemble ground-truth telemetry strictly inside the environment harness."""
        audits = self.pop_episode_audits()
        assert audits, "No rollout audits recorded during episode."
        idx = min(max(0, best_trial_idx), len(audits) - 1)
        chosen_audit = audits[idx]
        regime_id = chosen_audit.regime_id

        if validated_strategies is not None:
            contam = self._audit_evsa_records(validated_strategies, regime_id)
        elif memory_entries is not None:
            contam = self._audit_memory_entries(
                memory_entries, regime_id, include_reflection_avoid=include_reflection_avoid
            )
        else:
            contam = 0.0

        return EpisodeTelemetry(
            experiment_name=experiment_name,
            agent_name=agent_name,
            seed=seed,
            episode_idx=task.episode_idx,
            domain=task.domain,
            regime_id=regime_id,
            true_success=chosen_audit.true_success,
            observed_success=chosen_audit.observed_success,
            true_reward=chosen_audit.true_reward,
            observed_reward=chosen_audit.observed_reward,
            alignment_score=chosen_audit.alignment_score,
            trials_used=trials_used,
            strategy_evaluations=strategy_evaluations,
            execution_time_ms=execution_time_ms,
            active_memory_items=active_memory_items,
            memory_bytes=memory_bytes,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            shift_detected=shift_detected,
            contaminated_active_ratio=contam,
        )
