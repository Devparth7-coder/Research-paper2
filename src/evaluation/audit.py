"""
Evaluator-Only Episode Harness & Post-Hoc Contamination Audit (`src/evaluation/audit.py`).

Located strictly in the external `evaluation` package so no agent, memory,
retrieval, validation, or drift-detection module ever imports `NonStationaryTaskEnvironment`
or `EpisodeTelemetry` or accesses ground-truth workflow tables.
"""

from __future__ import annotations

from typing import Callable, Dict, Optional, Sequence, Tuple

from ..agents.baselines import (
    BaseExperimentalAgent,
    ConventionalMemoryAgent,
    ExperienceReplayAgent,
    ReflectionAgent,
)
from ..agents.evsa_agent import EVSAAgent
from ..benchmark.environment import NonStationaryTaskEnvironment
from ..benchmark.types import EpisodeTelemetry, RolloutResult, StrategyStatus, TaskInstance
from ..memory.repository import StrategyRepository


def run_audited_episode(
    agent: BaseExperimentalAgent,
    task: TaskInstance,
    oracle_env: NonStationaryTaskEnvironment,
    experiment_name: str,
    max_trials: int = 2,
    learn: bool = True,
    corruption_override: Optional[float] = None,
    rollout_observer: Optional[Callable[[TaskInstance, Tuple[int, ...], RolloutResult], None]] = None,
) -> EpisodeTelemetry:
    """
    Execute one task episode across the hard capability boundary:
    1. Construct a capability-isolated `AgentEnvironmentInterface` handle from `oracle_env`.
    2. Invoke `agent.act_on_episode(...)` (which receives ONLY observable `RolloutResult`s
       and returns strictly non-oracle `AgentEpisodeOutcome`).
    3. Join agent-side resource telemetry with evaluator-only ground-truth audit logs.
    """
    agent_iface = oracle_env.make_agent_interface(
        corruption_override=corruption_override,
        rollout_observer=rollout_observer,
    )
    outcome = agent.act_on_episode(task, agent_iface, max_trials=max_trials, learn=learn)

    val_strategies = None
    mem_entries = None
    include_ref_avoid = False
    if isinstance(agent, EVSAAgent):
        val_strategies = agent.repo.validated_strategies()
    elif isinstance(agent, ConventionalMemoryAgent):
        mem_entries = agent.memory
    elif isinstance(agent, ReflectionAgent):
        mem_entries = agent._all_entries()
        include_ref_avoid = True
    elif isinstance(agent, ExperienceReplayAgent):
        mem_entries = agent.replay_buffer

    return oracle_env.finalize_episode_telemetry(
        experiment_name=experiment_name,
        agent_name=agent.name,
        seed=agent.seed,
        task=task,
        best_trial_idx=outcome.best_trial_idx,
        trials_used=outcome.trials_used,
        strategy_evaluations=outcome.strategy_evaluations,
        execution_time_ms=outcome.execution_time_ms,
        active_memory_items=outcome.active_memory_items,
        memory_bytes=outcome.memory_bytes,
        prompt_tokens=outcome.prompt_tokens,
        completion_tokens=outcome.completion_tokens,
        shift_detected=outcome.shift_detected,
        validated_strategies=val_strategies,
        memory_entries=mem_entries,
        include_reflection_avoid=include_ref_avoid,
    )


def is_record_optimal_at_creation(
    created_task_id: str,
    domain: str,
    operator_plan: Tuple[int, ...],
    env: NonStationaryTaskEnvironment,
    default_regime: int = 0,
) -> bool:
    """Check post-hoc whether `operator_plan` matched the oracle plan of `created_task_id`."""
    if created_task_id and created_task_id in env._task_meta:
        _, _, opt = env.get_task_oracle(created_task_id)
        return operator_plan == opt
    return any(
        operator_plan == env.get_optimal_plan(domain, arch, regime_id=default_regime)
        for arch in range(8)
    )


def audit_evsa_repository(
    repo: StrategyRepository,
    env: NonStationaryTaskEnvironment,
    current_regime: int = 0,
) -> Dict[str, float]:
    """
    Post-hoc decomposition of EVSA repository quality:
    - `n_validated`, `clean_validated`, `false_promoted_active`, `lgg_contaminated_active`
    - `total_false_promotions`, `subopt_candidates`, `opt_records_tested`, `false_rejects`
    """
    vals = repo.validated_strategies()
    n_val = len(vals)
    clean = 0
    fp_active = 0
    lgg_bad = 0

    for rec in vals:
        was_opt = is_record_optimal_at_creation(
            rec.created_task_id, rec.domain, rec.operator_plan, env, default_regime=current_regime
        )
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
                if rec.operator_plan != env.get_optimal_plan(rec.domain, arch, regime_id=current_regime):
                    is_bad = True
                    break
        if checked == 0 or is_bad:
            if not was_opt:
                fp_active += 1
            else:
                lgg_bad += 1
        else:
            clean += 1

    all_recs = list(repo.records.values())
    subopt_cands = 0
    total_fp = 0
    opt_tested = 0
    false_rejects = 0

    for rec in all_recs:
        was_opt = is_record_optimal_at_creation(
            rec.created_task_id, rec.domain, rec.operator_plan, env, default_regime=current_regime
        )
        if not was_opt:
            subopt_cands += 1
            if rec.promoted_episode is not None:
                total_fp += 1
        elif rec.n_trials >= 3:
            opt_tested += 1
            if rec.status != StrategyStatus.VALIDATED:
                false_rejects += 1

    contam_rate = float((fp_active + lgg_bad) / float(n_val)) if n_val > 0 else 0.0
    return {
        "n_validated": n_val,
        "clean_validated": clean,
        "false_promoted_active": fp_active,
        "lgg_contaminated_active": lgg_bad,
        "contamination_rate": contam_rate,
        "subopt_candidates": subopt_cands,
        "total_false_promotions": total_fp,
        "opt_records_tested": opt_tested,
        "false_rejects": false_rejects,
    }


def count_false_admitted_plans(
    admitted_history: Sequence[Tuple[str, Tuple[int, ...]]],
    env: NonStationaryTaskEnvironment,
) -> int:
    """Count post-hoc how many admitted `(task_id, plan)` pairs were non-optimal."""
    bad = 0
    for task_id, plan in admitted_history:
        _, _, opt = env.get_task_oracle(task_id)
        if plan != opt:
            bad += 1
    return bad
