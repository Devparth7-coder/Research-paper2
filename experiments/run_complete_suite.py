#!/usr/bin/env python3
"""
Master Reproducible Execution, Recalculation, Sensitivity, and Statistical Pipeline
for EVSA v2.0.2 (Author: Dev Parth, Independent Researcher).

Executes:
1. Experiments 1--6 across N=20 independent random seeds (1001..1020) from scratch.
2. Empirical CUSUM residual trajectory logging for ShiftBench-Sim domains.
3. Experiment 4 denominator & safe-LGG contamination decomposition audit.
4. Fairness & sensitivity studies (Execution-Channel Parity at N=20; Seed-Horizon
   Stability at N=30).
5. Full parametric & non-parametric statistical inference with single-pass primary
   Holm-Bonferroni multiple-testing control (per-family and Global-74) plus separate
   labelled parametric/non-parametric sensitivity families and non-central-t power.
6. Programmatic CSV summary export to tables/.
"""

from __future__ import annotations

from dataclasses import replace
import datetime as dt
import json
import logging
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import src as evsa_pkg
from src.agents import (
    BaseExperimentalAgent,
    ConventionalMemoryAgent,
    EVSAAgent,
    ExperienceReplayAgent,
    ReflectionAgent,
    StaticAgent,
)
from src.benchmark import (
    DOMAINS,
    ESTIMATED_INPUT_COST_PER_1M_USD,
    ESTIMATED_OUTPUT_COST_PER_1M_USD,
    EpisodeTelemetry,
    NonStationaryTaskEnvironment,
    StrategyStatus,
    TaskInstance,
)
from src.evaluation import (
    apply_global_holm_bonferroni,
    apply_holm_bonferroni,
    audit_evsa_repository,
    compute_accuracy,
    compute_adaptation_gain,
    compute_paired_power,
    compute_recovery_rate_pct,
    compute_relative_improvement_pct,
    count_false_admitted_plans,
    minimum_detectable_dz,
    paired_statistical_comparison,
    run_audited_episode,
    summarize_seed_distribution,
)


def detect_cpu_model(
    target_platform: Optional[str] = None,
    cpuinfo_path: Path = Path("/proc/cpuinfo"),
) -> str:
    """
    Cross-platform CPU model detector (Windows, Linux, macOS) with zero UNIX shell pipelines.
    """
    plat = (target_platform or sys.platform).lower()

    # 1. Windows: read Windows Registry ProcessorNameString or PROCESSOR_IDENTIFIER
    if plat.startswith("win"):
        try:
            import winreg  # type: ignore[import-not-found]
            with winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE,
                r"HARDWARE\DESCRIPTION\System\CentralProcessor\0",
            ) as key:
                val, _ = winreg.QueryValueEx(key, "ProcessorNameString")
                if isinstance(val, str) and val.strip():
                    return val.strip()
        except Exception:
            pass
        env_cpu = os.environ.get("PROCESSOR_IDENTIFIER", "").strip()
        if env_cpu:
            return env_cpu

    # 2. Linux: parse /proc/cpuinfo directly in pure Python
    if plat.startswith("linux") and cpuinfo_path.exists():
        try:
            for line in cpuinfo_path.read_text(encoding="utf-8", errors="ignore").splitlines():
                if ":" in line and line.split(":", 1)[0].strip().lower() in {"model name", "hardware", "cpu"}:
                    val = line.split(":", 1)[1].strip()
                    if val:
                        return val
        except Exception:
            pass

    # 3. macOS: query sysctl directly via argument list
    if plat == "darwin":
        try:
            res = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
                timeout=3,
                check=False,
            )
            if res.returncode == 0 and res.stdout.strip():
                return res.stdout.strip()
        except Exception:
            pass

    return (platform.processor() or platform.machine() or "Unknown CPU").strip()


def write_experiment_configs() -> None:
    """Write explicit JSON configuration files to config/ and experiments/configs/."""
    global_cfg = {
        "project": "Evidence-Validated Strategy Adaptation for Memory-Augmented Agents Under Distribution Shift",
        "author": "Dev Parth",
        "role": "Independent Researcher",
        "affiliation": "Independent Research in Artificial Intelligence",
        "release_date": "2026-11-09",
        "version": evsa_pkg.__version__,
        "benchmark": {
            "name": "ShiftBench-Sim v1.0",
            "domains": list(DOMAINS),
            "step_horizon_H": 4,
            "action_cardinalities": [4, 5, 4, 3],
            "num_predicates": 5,
            "governing_predicates": 3,
            "nuisance_predicates": 2,
            "surface_topics": 8,
            "embedding_dim": 20,
            "embedding_weights": {"domain": 0.50, "structural": 0.36, "surface_topic": 0.82},
            "outcome_probabilities": {"p_opt": 0.92, "p_sub3": 0.28, "p_sub2": 0.09, "p_sub1": 0.02},
            "zero_oracle_exposure": True,
        },
        "backbone": {
            "type": "CalibratedSymbolicLLMBackbone",
            "base_step_competence": 0.67,
            "step3_prior_mode": "linear_inclusive_or",
            "adherence_range": [0.55, 0.96],
            "estimated_pricing_usd_per_1m_tokens": {
                "input": ESTIMATED_INPUT_COST_PER_1M_USD,
                "output": ESTIMATED_OUTPUT_COST_PER_1M_USD,
                "actual_commercial_api_cost_usd": 0.0,
            },
        },
        "evsa_hyperparameters": {
            "sprt_p0": 0.35,
            "sprt_p1": 0.80,
            "nominal_alpha_fa": 0.11,
            "nominal_beta_fr": 0.15,
            "gamma_sprt": 0.985,
            "min_trials": 3,
            "beta_prior": [1.0, 1.0],
            "gamma_beta": 0.985,
            "credible_delta": 0.10,
            "tau_lcb_promote": 0.48,
            "tau_rel_retrieve": 0.45,
            "tau_lcb_invalidate": 0.45,
            "min_lgg_literals": 3,
            "lgg_discount_gamma": 0.80,
            "salience_contrast_threshold": 0.14,
            "retrieval_weights": {"w_sem": 0.28, "w_conf": 0.52, "w_tier": 0.20, "specificity_bonus": 0.015},
            "cusum_slack_nu": 0.12,
            "cusum_threshold_h": 1.45,
            "cusum_min_obs_since_reset": 3,
            "cusum_cooldown_episodes": 12,
            "shift_discount_gamma": 0.35,
            "tau_quarantine_shift": 0.58,
            "max_active_per_domain": 48,
        },
    }
    (ROOT / "config" / "default_config.json").write_text(json.dumps(global_cfg, indent=2), encoding="utf-8")

    exp_cfgs = {
        "exp1_config.json": {
            "experiment": "Experiment 1: Unseen Task Generalization",
            "seeds": list(range(1001, 1021)),
            "train_episodes": 240,
            "train_archetypes": [0, 1, 2, 3, 4, 5, 6],
            "train_max_trials": 2,
            "eval_unseen_instance_episodes": 160,
            "eval_unseen_compositional_episodes": 80,
            "eval_compositional_archetype": [7],
            "eval_max_trials": 1,
        },
        "exp2_config.json": {
            "experiment": "Experiment 2: Continual Learning Across Episodes",
            "seeds": list(range(1001, 1021)),
            "num_blocks": 8,
            "episodes_per_block": 40,
            "total_train_episodes": 320,
            "heldout_probe_episodes": 96,
        },
        "exp3_config.json": {
            "experiment": "Experiment 3: Multi-Regime Distribution Shift Adaptation",
            "seeds": list(range(1001, 1021)),
            "phase1_stationary_episodes": [0, 160],
            "phase2_concept_shift_episodes": [160, 320],
            "phase3_compound_shift_episodes": [320, 420],
            "unshifted_control_domain": "resource_planning",
        },
        "exp4_config.json": {
            "experiment": "Experiment 4: Memory Reliability Under Misleading Feedback",
            "seeds": list(range(1001, 1021)),
            "corruption_rates_eta": [0.10, 0.25, 0.35],
            "noisy_episodes": 200,
            "clean_recovery_episodes": 100,
        },
        "exp5_config.json": {
            "experiment": "Experiment 5: Component Ablation Studies",
            "seeds": list(range(1001, 1021)),
            "noisy_pre_shift_episodes": 180,
            "noisy_pre_shift_eta": 0.15,
            "post_shift_episodes": 180,
            "heldout_unseen_episodes": 96,
        },
        "exp6_config.json": {
            "experiment": "Experiment 6: Computational, Latency, Memory, and Token Efficiency",
            "seeds": list(range(1001, 1021)),
            "instrumented_tasks_per_seed": 480,
        },
        "sensitivity_config.json": {
            "experiment": "Fairness & Seed-Horizon Sensitivity Audits",
            "fairness_execution_parity_seeds": list(range(1001, 1021)),
            "extended_seed_stability_seeds": list(range(1001, 1031)),
        },
    }
    for fname, payload in exp_cfgs.items():
        (ROOT / "experiments" / "configs" / fname).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def build_core_agents(seed: int) -> List[BaseExperimentalAgent]:
    return [
        StaticAgent(seed=seed),
        ConventionalMemoryAgent(seed=seed),
        ReflectionAgent(seed=seed),
        ExperienceReplayAgent(seed=seed),
        EVSAAgent(seed=seed),
    ]


def build_ablation_agents(seed: int) -> List[EVSAAgent]:
    return [
        EVSAAgent(seed=seed, name="EVSA (Full)"),
        EVSAAgent(seed=seed, name="w/o Evidence Validation", use_evidence_validation=False),
        EVSAAgent(seed=seed, name="w/o Confidence Estimation", use_confidence_estimation=False),
        EVSAAgent(seed=seed, name="w/o Applicability Conditions", use_applicability_conditions=False),
        EVSAAgent(seed=seed, name="w/o Shift Detection", use_shift_detection=False),
        EVSAAgent(seed=seed, name="w/o Memory Invalidation", use_memory_invalidation=False),
    ]


def evaluate_frozen_agent(
    agent: BaseExperimentalAgent,
    eval_tasks: Sequence[TaskInstance],
    eval_seed: int,
    exp_name: str,
    source_env: Optional[NonStationaryTaskEnvironment] = None,
) -> List[EpisodeTelemetry]:
    eval_env = NonStationaryTaskEnvironment(seed=eval_seed, feedback_corruption_rate=0.0)
    if source_env is not None:
        eval_env.copy_metadata_from(source_env)
    return [run_audited_episode(agent, t, eval_env, experiment_name=exp_name, max_trials=1, learn=False) for t in eval_tasks]


def run_experiment_1(seeds: Sequence[int], logger: logging.Logger) -> Tuple[pd.DataFrame, pd.DataFrame]:
    logger.info("Running Experiment 1 across %d seeds...", len(seeds))
    seed_rows: List[Dict[str, Any]] = []
    split_rows: List[Dict[str, Any]] = []

    for seed in seeds:
        stream_gen = NonStationaryTaskEnvironment(seed=seed)
        train_stream = stream_gen.sample_task_stream(
            num_episodes=240, allowed_archetypes=[0, 1, 2, 3, 4, 5, 6],
            stream_seed=seed * 10 + 1, stream_tag="e1tr",
        )
        eval_unseen_inst = stream_gen.sample_task_stream(
            num_episodes=160, allowed_archetypes=[0, 1, 2, 3, 4, 5, 6],
            unseen_flag=True, stream_seed=seed * 10 + 2, stream_tag="e1in",
        )
        eval_unseen_comp = stream_gen.sample_task_stream(
            num_episodes=80, allowed_archetypes=[7],
            unseen_flag=True, stream_seed=seed * 10 + 3, stream_tag="e1co",
        )

        for agent in build_core_agents(seed=seed):
            train_env = NonStationaryTaskEnvironment(seed=seed + 1000)
            train_env.copy_metadata_from(stream_gen)
            t0 = time.perf_counter()
            train_tels = [run_audited_episode(agent, t, train_env, "exp1_train", max_trials=2, learn=True) for t in train_stream]
            train_wall_ms = (time.perf_counter() - t0) * 1000.0

            t1 = time.perf_counter()
            eval_inst_tels = evaluate_frozen_agent(
                agent, eval_unseen_inst, seed + 2000, "exp1_eval_inst", source_env=stream_gen
            )
            eval_comp_tels = evaluate_frozen_agent(
                agent, eval_unseen_comp, seed + 3000, "exp1_eval_comp", source_env=stream_gen
            )
            eval_wall_ms = (time.perf_counter() - t1) * 1000.0

            all_eval = eval_inst_tels + eval_comp_tels
            total_p_tok = sum(r.prompt_tokens for r in train_tels) + sum(r.prompt_tokens for r in all_eval)
            total_c_tok = sum(r.completion_tokens for r in train_tels) + sum(r.completion_tokens for r in all_eval)
            false_drift_alarms = sum(int(r.shift_detected) for r in train_tels)

            seed_rows.append({
                "experiment": "exp1_baseline",
                "seed": seed,
                "agent": agent.name,
                "benchmark_tasks": len(all_eval),
                "task_success_rate": compute_accuracy([r.true_success for r in all_eval]),
                "unseen_instance_sr": compute_accuracy([r.true_success for r in eval_inst_tels]),
                "unseen_compositional_sr": compute_accuracy([r.true_success for r in eval_comp_tels]),
                "exact_plan_accuracy": compute_accuracy([int(r.alignment_score == 1.0) for r in all_eval]),
                "step_alignment_accuracy": float(np.mean([r.alignment_score for r in all_eval])),
                "average_reward": float(np.mean([r.true_reward for r in all_eval])),
                "eval_time_ms_per_task": float(np.mean([r.execution_time_ms for r in all_eval])),
                "total_wall_time_ms": train_wall_ms + eval_wall_ms,
                "memory_kb": agent.memory_footprint_bytes() / 1024.0,
                "active_memory_items": agent.active_memory_count(),
                "total_rollouts": sum(r.trials_used for r in train_tels) + len(all_eval),
                "total_strategy_evaluations": sum(r.strategy_evaluations for r in train_tels) + sum(
                    r.strategy_evaluations for r in all_eval
                ),
                "prompt_tokens": total_p_tok,
                "completion_tokens": total_c_tok,
                "total_tokens": total_p_tok + total_c_tok,
                "estimated_api_cost_usd": (
                    (total_p_tok / 1e6) * ESTIMATED_INPUT_COST_PER_1M_USD
                    + (total_c_tok / 1e6) * ESTIMATED_OUTPUT_COST_PER_1M_USD
                ),
                "stationary_false_drift_alarms": false_drift_alarms,
            })

            for split_label, sub_tels in [
                ("Unseen_Instances", eval_inst_tels),
                ("Unseen_Compositional_Archetype", eval_comp_tels),
            ]:
                split_rows.append({
                    "seed": seed,
                    "agent": agent.name,
                    "split": split_label,
                    "success_rate": compute_accuracy([r.true_success for r in sub_tels]),
                    "exact_plan_accuracy": compute_accuracy([int(r.alignment_score == 1.0) for r in sub_tels]),
                    "average_reward": float(np.mean([r.true_reward for r in sub_tels])),
                })

    return pd.DataFrame(seed_rows), pd.DataFrame(split_rows)


def run_experiment_2(seeds: Sequence[int], logger: logging.Logger) -> Tuple[pd.DataFrame, pd.DataFrame]:
    logger.info("Running Experiment 2 across %d seeds...", len(seeds))
    num_blocks, block_size = 8, 40
    curve_rows: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []

    for seed in seeds:
        stream_gen = NonStationaryTaskEnvironment(seed=seed + 50)
        learn_tasks = stream_gen.sample_task_stream(
            num_episodes=num_blocks * block_size, stream_seed=seed * 20 + 1, stream_tag="e2tr"
        )
        probe_tasks = stream_gen.sample_task_stream(
            num_episodes=96, unseen_flag=True, stream_seed=seed * 20 + 2, stream_tag="e2pr"
        )

        for agent in build_core_agents(seed=seed):
            train_env = NonStationaryTaskEnvironment(seed=seed + 4000)
            train_env.copy_metadata_from(stream_gen)
            init_tels = evaluate_frozen_agent(
                agent, probe_tasks, seed + 5000, "exp2_probe_0", source_env=stream_gen
            )
            init_sr = compute_accuracy([r.true_success for r in init_tels])
            curve_rows.append({
                "seed": seed, "agent": agent.name, "learning_block": 0, "tasks_seen": 0,
                "eval_success_rate": init_sr,
                "eval_reward": float(np.mean([r.true_reward for r in init_tels])),
                "active_memory_items": agent.active_memory_count(),
                "memory_kb": agent.memory_footprint_bytes() / 1024.0,
            })

            block_srs = [init_sr]
            stationary_alarms = 0
            for b in range(1, num_blocks + 1):
                chunk = learn_tasks[(b - 1) * block_size : b * block_size]
                for task in chunk:
                    tel = run_audited_episode(agent, task, train_env, "exp2_learn", max_trials=2, learn=True)
                    stationary_alarms += int(tel.shift_detected)
                probe_tels = evaluate_frozen_agent(
                    agent, probe_tasks, seed + 5000 + b, f"exp2_probe_{b}", source_env=stream_gen
                )
                b_sr = compute_accuracy([r.true_success for r in probe_tels])
                block_srs.append(b_sr)
                curve_rows.append({
                    "seed": seed, "agent": agent.name, "learning_block": b, "tasks_seen": b * block_size,
                    "eval_success_rate": b_sr,
                    "eval_reward": float(np.mean([r.true_reward for r in probe_tels])),
                    "active_memory_items": agent.active_memory_count(),
                    "memory_kb": agent.memory_footprint_bytes() / 1024.0,
                })

            if isinstance(agent, EVSAAgent):
                retained = len(agent.repo.validated_strategies())
                candidates_pending = len([r for r in agent.repo.records.values() if r.status == StrategyStatus.CANDIDATE])
                rejected = agent.repo.total_quarantined
            elif isinstance(agent, ExperienceReplayAgent):
                retained, candidates_pending, rejected = len(agent.replay_buffer), 0, agent.rejected_strategies
            elif isinstance(agent, ConventionalMemoryAgent):
                retained = len([m for m in agent.memory if m.observed_success == 1])
                candidates_pending = 0
                rejected = len([m for m in agent.memory if m.observed_success == 0])
            elif isinstance(agent, ReflectionAgent):
                retained = agent.accepted_count
                candidates_pending = 0
                rejected = len([m for m in agent._all_entries() if m.observed_success == 0])
            else:
                retained, candidates_pending, rejected = 0, 0, 0

            final_sr = block_srs[-1]
            summary_rows.append({
                "seed": seed,
                "agent": agent.name,
                "initial_sr": init_sr,
                "block1_sr": block_srs[1],
                "final_sr": final_sr,
                "total_improvement_abs": final_sr - init_sr,
                "total_improvement_pct": compute_relative_improvement_pct(final_sr, init_sr),
                "auc_sr": float(np.mean(block_srs[1:])),
                "strategies_retained": retained,
                "candidates_pending": candidates_pending,
                "strategies_rejected": rejected,
                "stationary_false_alarms": stationary_alarms,
            })

    return pd.DataFrame(curve_rows), pd.DataFrame(summary_rows)


def run_experiment_3(
    seeds: Sequence[int],
    logger: logging.Logger,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    logger.info("Running Experiment 3 across %d seeds...", len(seeds))
    T1, T2, T_END = 160, 320, 420
    window_rows: List[Dict[str, Any]] = []
    shift_summary_rows: List[Dict[str, Any]] = []
    cusum_trace_rows: List[Dict[str, Any]] = []

    for seed in seeds:
        gen = NonStationaryTaskEnvironment(seed=seed + 77)
        p1_tasks = gen.sample_task_stream(
            T1, regime_schedule=[0] * T1, allowed_archetypes=[0, 1, 2, 3, 4, 5, 6],
            stream_seed=seed * 30 + 1, stream_tag="s",
        )
        p2_tasks = gen.sample_task_stream(
            T2 - T1, regime_schedule=[1] * (T2 - T1), allowed_archetypes=[0, 1, 2, 3, 4, 5, 6],
            stream_seed=seed * 30 + 2, stream_tag="s",
        )
        p3_tasks = gen.sample_task_stream(
            T_END - T2, regime_schedule=[2] * (T_END - T2), allowed_archetypes=[4, 5, 6, 7],
            unseen_flag=True, stream_seed=seed * 30 + 3, stream_tag="s",
        )

        full_stream: List[TaskInstance] = [
            replace(t, episode_idx=idx) for idx, t in enumerate(p1_tasks + p2_tasks + p3_tasks)
        ]

        for agent in build_core_agents(seed=seed):
            run_env = NonStationaryTaskEnvironment(seed=seed + 6000)
            run_env.copy_metadata_from(gen)
            ep_tels: List[EpisodeTelemetry] = []
            pre_shift_validated_ids: set[str] = set()

            for idx, task in enumerate(full_stream):
                if idx == T1 and isinstance(agent, EVSAAgent):
                    pre_shift_validated_ids = {r.strategy_id for r in agent.repo.validated_strategies()}
                tel = run_audited_episode(agent, task, run_env, "exp3_shift", max_trials=2, learn=True)
                ep_tels.append(tel)

                if isinstance(agent, EVSAAgent):
                    for d_name in DOMAINS:
                        cusum_trace_rows.append({
                            "seed": seed,
                            "episode_idx": idx,
                            "domain": d_name,
                            "cusum_stat": agent.shift_detector.cusum_stat.get(d_name, 0.0),
                            "alarm_fired": int(tel.shift_detected and task.domain == d_name),
                        })

            win_size = 20
            for w_start in range(0, T_END, win_size):
                sub = ep_tels[w_start : w_start + win_size]
                window_rows.append({
                    "seed": seed,
                    "agent": agent.name,
                    "window_start": w_start,
                    "window_mid": w_start + win_size // 2,
                    "regime_id": sub[0].regime_id,
                    "success_rate": compute_accuracy([r.true_success for r in sub]),
                    "average_reward": float(np.mean([r.true_reward for r in sub])),
                    "contaminated_ratio": float(np.mean([r.contaminated_active_ratio for r in sub])),
                    "shift_alarms": sum(int(r.shift_detected) for r in sub),
                })

            pre_shift_sr = compute_accuracy([r.true_success for r in ep_tels[T1 - 40 : T1]])
            shock_sr = compute_accuracy([r.true_success for r in ep_tels[T1 : T1 + 25]])
            recovered_sr = compute_accuracy([r.true_success for r in ep_tels[T2 - 50 : T2]])
            phase3_sr = compute_accuracy([r.true_success for r in ep_tels[T_END - 40 : T_END]])

            unseen_arch7 = [
                ep_tels[i].true_success
                for i in range(T2, T_END)
                if gen.get_task_oracle(full_stream[i].task_id)[1] == 7
            ]
            unseen_post_shift_sr = compute_accuracy(unseen_arch7) if unseen_arch7 else phase3_sr

            unshifted_dom = [
                ep_tels[i].true_success for i in range(T1, T2) if full_stream[i].domain == "resource_planning"
            ]
            unshifted_retention_sr = compute_accuracy(unshifted_dom)

            target_sr = 0.85 * pre_shift_sr
            adapt_episodes = T2 - T1
            p2_succ = [r.true_success for r in ep_tels[T1:T2]]
            for offset in range(15, len(p2_succ)):
                if float(np.mean(p2_succ[offset - 15 : offset + 1])) >= target_sr:
                    adapt_episodes = offset + 1
                    break

            mean_alarm_delay = np.nan
            unshifted_false_alarms_p2 = 0
            if isinstance(agent, EVSAAgent):
                post_val_ids = {
                    r.strategy_id for r in agent.repo.records.values()
                    if r.promoted_episode is not None and r.promoted_episode >= T1
                }
                new_strategies_discovered = len(post_val_ids - pre_shift_validated_ids)
                delays = []
                for s_dom in ("api_orchestration", "system_diagnosis", "data_pipeline"):
                    alarms_dom = [ep for ep, dm in agent.shift_detector.alarm_episodes if dm == s_dom and T1 <= ep < T2]
                    if alarms_dom:
                        delays.append(min(alarms_dom) - T1)
                if delays:
                    mean_alarm_delay = float(np.mean(delays))
                unshifted_false_alarms_p2 = len([
                    ep for ep, dm in agent.shift_detector.alarm_episodes
                    if dm == "resource_planning" and T1 <= ep < T2
                ])
            elif isinstance(agent, ExperienceReplayAgent):
                new_strategies_discovered = len([m for m in agent.replay_buffer if m.episode_idx >= T1])
            elif isinstance(agent, ConventionalMemoryAgent):
                new_strategies_discovered = len([
                    m for m in agent.memory if m.episode_idx >= T1 and m.observed_success == 1
                ])
            elif isinstance(agent, ReflectionAgent):
                new_strategies_discovered = len([
                    m for m in agent._all_entries() if m.episode_idx >= T1 and m.observed_success == 1
                ])
            else:
                new_strategies_discovered = 0

            shift_summary_rows.append({
                "seed": seed,
                "agent": agent.name,
                "pre_shift_sr": pre_shift_sr,
                "immediate_post_shift_sr": shock_sr,
                "performance_degradation": pre_shift_sr - shock_sr,
                "recovered_post_shift_sr": recovered_sr,
                "adaptation_gain": compute_adaptation_gain(recovered_sr, shock_sr),
                "recovery_rate_pct": compute_recovery_rate_pct(recovered_sr, pre_shift_sr),
                "adaptation_speed_episodes": adapt_episodes,
                "unshifted_domain_sr": unshifted_retention_sr,
                "phase3_compound_shift_sr": phase3_sr,
                "unseen_tasks_post_shift_sr": unseen_post_shift_sr,
                "new_strategies_discovered": new_strategies_discovered,
                "mean_alarm_delay_episodes": mean_alarm_delay,
                "unshifted_false_alarms_p2": unshifted_false_alarms_p2,
            })

    return pd.DataFrame(window_rows), pd.DataFrame(shift_summary_rows), pd.DataFrame(cusum_trace_rows)


def run_experiment_4(seeds: Sequence[int], logger: logging.Logger) -> pd.DataFrame:
    corruption_levels = [0.10, 0.25, 0.35]
    logger.info("Running Experiment 4 across %d seeds and eta=%s...", len(seeds), corruption_levels)
    rows: List[Dict[str, Any]] = []

    for eta in corruption_levels:
        for seed in seeds:
            gen = NonStationaryTaskEnvironment(seed=seed + 90)
            noisy_stream = gen.sample_task_stream(
                200, stream_seed=seed * 40 + int(eta * 100), stream_tag="e4n"
            )
            clean_recovery_stream = gen.sample_task_stream(
                100, stream_seed=seed * 40 + int(eta * 100) + 1, stream_tag="e4c"
            )
            combined_stream: List[TaskInstance] = [
                replace(t, episode_idx=idx) for idx, t in enumerate(noisy_stream + clean_recovery_stream)
            ]

            for agent in build_core_agents(seed=seed):
                noisy_env = NonStationaryTaskEnvironment(seed=seed + 7000, feedback_corruption_rate=eta)
                noisy_env.copy_metadata_from(gen)
                clean_env = NonStationaryTaskEnvironment(seed=seed + 8000, feedback_corruption_rate=0.0)
                clean_env.copy_metadata_from(gen)

                rollout_audit: List[Tuple[bool, int]] = []

                def observe_rollout(task_obj: TaskInstance, plan_tuple: Tuple[int, ...], res_obj) -> None:
                    _, _, opt_p = gen.get_task_oracle(task_obj.task_id)
                    rollout_audit.append((plan_tuple == opt_p, res_obj.observed_success))

                noisy_tels = [
                    run_audited_episode(
                        agent,
                        task,
                        noisy_env,
                        f"exp4_eta_{eta}",
                        max_trials=2,
                        learn=True,
                        rollout_observer=observe_rollout,
                    )
                    for task in combined_stream[:200]
                ]
                contam_at_200 = noisy_tels[-1].contaminated_active_ratio
                noisy_late_sr = compute_accuracy([r.true_success for r in noisy_tels[120:200]])

                subopt_attempts = sum(1 for is_opt, _ in rollout_audit if not is_opt)
                opt_attempts = sum(1 for is_opt, _ in rollout_audit if is_opt)

                cand_fpr = np.nan
                n_val_200 = 0
                clean_val_200 = 0
                fp_active_200 = 0
                lgg_bad_200 = 0
                lgg_merges_200 = 0

                if isinstance(agent, EVSAAgent):
                    bd = audit_evsa_repository(agent.repo, gen, current_regime=0)
                    false_accepts = int(bd["total_false_promotions"])
                    subopt_cands = int(bd["subopt_candidates"])
                    opt_tested = int(bd["opt_records_tested"])
                    false_rejects = int(bd["false_rejects"])
                    far = false_accepts / float(max(1, subopt_attempts))
                    cand_fpr = false_accepts / float(max(1, subopt_cands))
                    frr = false_rejects / float(max(1, opt_tested)) if opt_tested > 0 else 0.0
                    n_val_200 = int(bd["n_validated"])
                    clean_val_200 = int(bd["clean_validated"])
                    fp_active_200 = int(bd["false_promoted_active"])
                    lgg_bad_200 = int(bd["lgg_contaminated_active"])
                    lgg_merges_200 = int(agent.repo.lgg_merges)
                elif isinstance(agent, ExperienceReplayAgent):
                    false_accepts = count_false_admitted_plans(agent.admitted_history, gen)
                    opt_rejected = sum(1 for is_opt, obs in rollout_audit if is_opt and obs == 0)
                    far = false_accepts / float(max(1, subopt_attempts))
                    cand_fpr = far
                    frr = opt_rejected / float(max(1, opt_attempts))
                    n_val_200 = len(agent.replay_buffer)
                elif isinstance(agent, ConventionalMemoryAgent):
                    false_pos_stored = len([
                        m for m in agent.memory
                        if m.observed_success == 1
                        and m.executed_plan != gen.get_task_oracle(m.task_id)[2]
                    ])
                    opt_rejected = sum(1 for is_opt, obs in rollout_audit if is_opt and obs == 0)
                    far = false_pos_stored / float(max(1, subopt_attempts))
                    cand_fpr = far
                    frr = opt_rejected / float(max(1, opt_attempts))
                    n_val_200 = len([m for m in agent.memory if m.observed_success == 1])
                elif isinstance(agent, ReflectionAgent):
                    false_accepts = count_false_admitted_plans(agent.accepted_history, gen)
                    opt_rejected = sum(1 for is_opt, obs in rollout_audit if is_opt and obs == 0)
                    far = false_accepts / float(max(1, subopt_attempts))
                    cand_fpr = far
                    frr = opt_rejected / float(max(1, opt_attempts))
                    n_val_200 = len(agent._all_entries())
                else:
                    far, cand_fpr, frr = 0.0, 0.0, 0.0

                rec_tels = [
                    run_audited_episode(agent, task, clean_env, f"exp4_rec_{eta}", max_trials=2, learn=True)
                    for task in combined_stream[200:]
                ]
                rows.append({
                    "seed": seed,
                    "corruption_rate_eta": eta,
                    "agent": agent.name,
                    "subopt_rollouts": subopt_attempts,
                    "false_acceptance_rate": far,
                    "candidate_false_promotion_rate": cand_fpr,
                    "false_rejection_rate": frr,
                    "active_validated_end_noisy": n_val_200,
                    "clean_validated_end_noisy": clean_val_200,
                    "false_promoted_active_noisy": fp_active_200,
                    "lgg_contaminated_active_noisy": lgg_bad_200,
                    "lgg_merges_noisy": lgg_merges_200,
                    "memory_contamination_rate": contam_at_200,
                    "post_clean_contamination_rate": rec_tels[-1].contaminated_active_ratio,
                    "noisy_phase_sr": noisy_late_sr,
                    "recovery_after_misleading_sr": compute_accuracy([r.true_success for r in rec_tels[50:]]),
                })

    return pd.DataFrame(rows)


def run_experiment_5(seeds: Sequence[int], logger: logging.Logger) -> pd.DataFrame:
    logger.info("Running Experiment 5 across %d seeds...", len(seeds))
    rows: List[Dict[str, Any]] = []

    for seed in seeds:
        gen = NonStationaryTaskEnvironment(seed=seed + 123)
        p1 = gen.sample_task_stream(
            180, regime_schedule=[0] * 180, allowed_archetypes=[0, 1, 2, 3, 4, 5, 6],
            stream_seed=seed * 50 + 1, stream_tag="e5a",
        )
        p2 = gen.sample_task_stream(
            180, regime_schedule=[1] * 180, allowed_archetypes=[0, 1, 2, 3, 4, 5, 6],
            stream_seed=seed * 50 + 2, stream_tag="e5b",
        )
        unseen_eval = gen.sample_task_stream(
            96, regime_schedule=[1] * 96, allowed_archetypes=[0, 1, 2, 3, 4, 5, 6, 7],
            unseen_flag=True, stream_seed=seed * 50 + 3, stream_tag="e5ev",
        )
        stream = [replace(t, episode_idx=idx) for idx, t in enumerate(p1 + p2)]

        for variant in build_ablation_agents(seed=seed):
            env_noisy = NonStationaryTaskEnvironment(seed=seed + 9100, feedback_corruption_rate=0.15)
            env_noisy.copy_metadata_from(gen)
            env_shift = NonStationaryTaskEnvironment(seed=seed + 9200, feedback_corruption_rate=0.0)
            env_shift.copy_metadata_from(gen)
            tels = [
                run_audited_episode(
                    variant,
                    task,
                    env_noisy if idx < 180 else env_shift,
                    "exp5_ablation",
                    max_trials=2,
                    learn=True,
                )
                for idx, task in enumerate(stream)
            ]
            eval_tels = evaluate_frozen_agent(variant, unseen_eval, seed + 9300, "exp5_eval", source_env=gen)
            rows.append({
                "seed": seed,
                "variant": variant.name,
                "overall_stream_sr": compute_accuracy([r.true_success for r in tels]),
                "noisy_pre_shift_sr": compute_accuracy([r.true_success for r in tels[120:180]]),
                "post_shift_recovered_sr": compute_accuracy([r.true_success for r in tels[300:360]]),
                "heldout_unseen_sr": compute_accuracy([r.true_success for r in eval_tels]),
                "pre_shift_contamination": tels[179].contaminated_active_ratio,
                "post_shift_contamination": tels[-1].contaminated_active_ratio,
                "active_memory_items": variant.active_memory_count(),
                "validated_strategies": variant.validated_memory_count(),
                "memory_kb": variant.memory_footprint_bytes() / 1024.0,
            })

    return pd.DataFrame(rows)


def run_fairness_and_seed_sensitivity(logger: logging.Logger) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Execute controlled fairness & statistical power sensitivity experiments:
    1. Execution-Channel Parity (N=20 seeds, 1001..1020): compares EVSA and Replay
       under both deterministic macro execution and stochastic prompt conditioning.
    2. Seed-Horizon Stability (N=30 seeds, 1001..1030): verifies stability of
       Experiment 1 unseen generalization beyond N=20 seeds.
    """
    logger.info("Running Fairness Execution-Channel Parity & N=30 Seed Stability Checks...")
    fairness_rows: List[Dict[str, Any]] = []
    seeds_20 = list(range(1001, 1021))

    for seed in seeds_20:
        gen = NonStationaryTaskEnvironment(seed=seed)
        train_s = gen.sample_task_stream(
            240, allowed_archetypes=[0, 1, 2, 3, 4, 5, 6], stream_seed=seed * 10 + 1, stream_tag="ftr"
        )
        eval_i = gen.sample_task_stream(
            160, allowed_archetypes=[0, 1, 2, 3, 4, 5, 6], unseen_flag=True, stream_seed=seed * 10 + 2, stream_tag="fi"
        )
        eval_c = gen.sample_task_stream(
            80, allowed_archetypes=[7], unseen_flag=True, stream_seed=seed * 10 + 3, stream_tag="fc"
        )

        test_agents: List[BaseExperimentalAgent] = [
            EVSAAgent(seed=seed, name="EVSA (Direct Macro Exec)", stochastic_prompt_execution=False),
            EVSAAgent(seed=seed, name="EVSA (Stochastic Prompt Exec)", stochastic_prompt_execution=True),
            ExperienceReplayAgent(seed=seed, direct_macro_execution=False, custom_name="Replay (Stochastic Prompt Exec)"),
            ExperienceReplayAgent(seed=seed, direct_macro_execution=True, custom_name="Replay (Direct Macro Exec)"),
        ]
        for ag in test_agents:
            tr_env = NonStationaryTaskEnvironment(seed=seed + 1000)
            tr_env.copy_metadata_from(gen)
            for t in train_s:
                run_audited_episode(ag, t, tr_env, "fair_tr", max_trials=2, learn=True)
            ev_i = evaluate_frozen_agent(ag, eval_i, seed + 2000, "fair_ev_i", source_env=gen)
            ev_c = evaluate_frozen_agent(ag, eval_c, seed + 3000, "fair_ev_c", source_env=gen)
            all_ev = ev_i + ev_c
            fairness_rows.append({
                "seed": seed,
                "agent": ag.name,
                "overall_sr": compute_accuracy([r.true_success for r in all_ev]),
                "unseen_instance_sr": compute_accuracy([r.true_success for r in ev_i]),
                "unseen_compositional_sr": compute_accuracy([r.true_success for r in ev_c]),
            })

    extra_seeds = list(range(1021, 1031))
    df_extra_main, _ = run_experiment_1(extra_seeds, logger)
    return pd.DataFrame(fairness_rows), df_extra_main


def summarize_grouped(df: pd.DataFrame, group_cols: List[str], metric_cols: List[str]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for keys, sub in df.groupby(group_cols, sort=False):
        row: Dict[str, Any] = {}
        if isinstance(keys, tuple):
            for c, k in zip(group_cols, keys):
                row[c] = k
        else:
            row[group_cols[0]] = keys
        row["n_seeds"] = len(sub)
        for m in metric_cols:
            s = summarize_seed_distribution(sub[m].dropna().to_numpy())
            for stat_k in ("mean", "median", "sd", "iqr", "ci95_low", "ci95_high", "boot_ci95_low", "boot_ci95_high"):
                row[f"{m}_{stat_k}"] = s[stat_k]
        rows.append(row)
    return pd.DataFrame(rows)


def export_markdown_summary_tables(tbl_dir: Path) -> None:
    """Export human-readable Markdown summary tables directly from CSV artifacts."""
    specs = [
        ("table_exp1_generalization.csv", [
            "agent", "task_success_rate_mean", "unseen_instance_sr_mean",
            "unseen_compositional_sr_mean", "exact_plan_accuracy_mean",
            "step_alignment_accuracy_mean", "average_reward_mean", "eval_time_ms_per_task_mean",
        ], None),
        ("table_exp2_continual_learning.csv", [
            "agent", "initial_sr_mean", "block1_sr_mean", "final_sr_mean",
            "total_improvement_abs_mean", "total_improvement_pct_mean",
            "auc_sr_mean", "strategies_retained_mean",
        ], None),
        ("table_exp3_distribution_shift.csv", [
            "agent", "pre_shift_sr_mean", "immediate_post_shift_sr_mean",
            "performance_degradation_mean", "recovered_post_shift_sr_mean",
            "adaptation_gain_mean", "recovery_rate_pct_mean", "adaptation_speed_episodes_mean",
        ], None),
        ("table_exp4_feedback_corruption.csv", [
            "corruption_rate_eta", "agent", "subopt_rollouts_mean",
            "false_acceptance_rate_mean", "candidate_false_promotion_rate_mean",
            "false_rejection_rate_mean", "memory_contamination_rate_mean",
            "recovery_after_misleading_sr_mean",
        ], None),
        ("table_exp5_ablation.csv", [
            "variant", "overall_stream_sr_mean", "noisy_pre_shift_sr_mean",
            "post_shift_recovered_sr_mean", "heldout_unseen_sr_mean",
            "pre_shift_contamination_mean", "post_shift_contamination_mean", "validated_strategies_mean",
        ], None),
        ("table_exp6_efficiency.csv", [
            "agent", "task_success_rate_mean", "total_wall_time_ms_mean",
            "eval_time_ms_per_task_mean", "memory_kb_mean", "active_memory_items_mean",
            "total_rollouts_mean", "total_tokens_mean",
        ], None),
        ("table_sensitivity_30seeds.csv", [
            "agent", "task_success_rate_mean", "unseen_instance_sr_mean", "unseen_compositional_sr_mean",
        ], None),
        ("table_sensitivity_fairness.csv", [
            "agent", "overall_sr_mean", "unseen_instance_sr_mean", "unseen_compositional_sr_mean",
        ], None),
        ("table_statistical_tests_all.csv", [
            "proposed", "comparator", "cohens_dz", "family", "metric",
            "primary_test", "primary_pvalue_raw", "primary_pvalue_holm",
            "primary_pvalue_holm_global74", "sensitivity_t_pvalue_holm_global74",
            "significant_at_05", "significant_global74_at_05",
        ], None),
    ]
    md_parts = [f"# EVSA v{evsa_pkg.__version__} Verified Publication Tables (Recalculated from Raw Seed Data)\n"]
    for csv_name, cols, max_rows in specs:
        df_c = pd.read_csv(tbl_dir / csv_name)[cols]
        if max_rows is not None:
            df_c = df_c.head(max_rows)
        md_parts.append(f"## `{csv_name}`\n")
        header = "| " + " | ".join(cols) + " |"
        sep = "| " + " | ".join(["---"] * len(cols)) + " |"
        rows_txt = [header, sep]
        for _, r in df_c.iterrows():
            vals = []
            for c in cols:
                v = r[c]
                if isinstance(v, float):
                    vals.append(f"{v:.4g}")
                else:
                    vals.append(str(v))
            rows_txt.append("| " + " | ".join(vals) + " |")
        md_parts.append("\n".join(rows_txt) + "\n")
    (tbl_dir / "summary_tables.md").write_text("\n".join(md_parts), encoding="utf-8")


def main() -> None:
    write_experiment_configs()
    exp_dir = ROOT / "experiments"
    log_path = exp_dir / "logs" / "experiment_execution.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(log_path, mode="w", encoding="utf-8"), logging.StreamHandler(sys.stdout)],
    )
    logger = logging.getLogger("evsa_audit")

    seeds = list(range(1001, 1021))
    t0 = time.perf_counter()

    df_exp1_main, df_exp1_splits = run_experiment_1(seeds, logger)
    df_exp2_curves, df_exp2_summary = run_experiment_2(seeds, logger)
    df_exp3_windows, df_exp3_summary, df_exp3_cusum = run_experiment_3(seeds, logger)
    df_exp4 = run_experiment_4(seeds, logger)
    df_exp5 = run_experiment_5(seeds, logger)
    df_fairness, df_exp1_extra10 = run_fairness_and_seed_sensitivity(logger)

    df_exp6 = df_exp1_main[[
        "seed", "agent", "benchmark_tasks", "task_success_rate", "total_wall_time_ms",
        "eval_time_ms_per_task", "memory_kb", "active_memory_items", "total_rollouts",
        "total_strategy_evaluations", "prompt_tokens", "completion_tokens",
        "total_tokens", "estimated_api_cost_usd",
    ]].copy()
    df_exp6["sr_per_10k_tokens"] = df_exp6["task_success_rate"] / (df_exp6["total_tokens"] / 10000.0)

    df_exp1_30seeds = pd.concat([df_exp1_main, df_exp1_extra10], ignore_index=True)

    raw_dir = exp_dir / "raw"
    df_exp1_main.to_csv(raw_dir / "exp1_baseline_raw_seeds.csv", index=False)
    df_exp1_splits.to_csv(raw_dir / "exp1_unseen_splits_raw_seeds.csv", index=False)
    df_exp1_30seeds.to_csv(raw_dir / "exp1_30seed_stability_raw.csv", index=False)
    df_exp2_curves.to_csv(raw_dir / "exp2_learning_curves_raw.csv", index=False)
    df_exp2_summary.to_csv(raw_dir / "exp2_learning_summary_raw.csv", index=False)
    df_exp3_windows.to_csv(raw_dir / "exp3_shift_windows_raw.csv", index=False)
    df_exp3_summary.to_csv(raw_dir / "exp3_shift_summary_raw.csv", index=False)
    df_exp3_cusum.to_csv(raw_dir / "exp3_domain_cusum_traces_raw.csv", index=False)
    df_exp4.to_csv(raw_dir / "exp4_misleading_feedback_raw.csv", index=False)
    df_exp5.to_csv(raw_dir / "exp5_ablation_raw.csv", index=False)
    df_exp6.to_csv(raw_dir / "exp6_computational_efficiency_raw.csv", index=False)
    df_fairness.to_csv(raw_dir / "sensitivity_execution_parity_raw.csv", index=False)

    proc_dir = exp_dir / "processed"
    s1 = summarize_grouped(df_exp1_main, ["agent"], [
        "task_success_rate", "unseen_instance_sr", "unseen_compositional_sr",
        "exact_plan_accuracy", "step_alignment_accuracy", "average_reward",
        "eval_time_ms_per_task", "memory_kb", "stationary_false_drift_alarms",
    ])
    evsa_sr1 = float(s1.loc[s1["agent"] == "Proposed Agent E: EVSA", "task_success_rate_mean"].iloc[0])
    s1["evsa_rel_improvement_over_agent_pct"] = s1["task_success_rate_mean"].apply(
        lambda v: compute_relative_improvement_pct(evsa_sr1, float(v))
    )
    s1.to_csv(proc_dir / "exp1_baseline_summary.csv", index=False)

    s1_30 = summarize_grouped(df_exp1_30seeds, ["agent"], [
        "task_success_rate", "unseen_instance_sr", "unseen_compositional_sr",
    ])
    s1_30.to_csv(proc_dir / "exp1_30seed_stability_summary.csv", index=False)

    s2_curves = summarize_grouped(df_exp2_curves, ["agent", "learning_block", "tasks_seen"], [
        "eval_success_rate", "eval_reward", "active_memory_items", "memory_kb",
    ])
    s2_curves.to_csv(proc_dir / "exp2_learning_curves_summary.csv", index=False)

    s2_sum = summarize_grouped(df_exp2_summary, ["agent"], [
        "initial_sr", "block1_sr", "final_sr", "total_improvement_abs",
        "total_improvement_pct", "auc_sr", "strategies_retained",
        "candidates_pending", "strategies_rejected", "stationary_false_alarms",
    ])
    s2_sum.to_csv(proc_dir / "exp2_learning_summary.csv", index=False)

    s3_win = summarize_grouped(df_exp3_windows, ["agent", "window_start", "window_mid", "regime_id"], [
        "success_rate", "average_reward", "contaminated_ratio", "shift_alarms",
    ])
    s3_win.to_csv(proc_dir / "exp3_shift_windows_summary.csv", index=False)

    s3_sum = summarize_grouped(df_exp3_summary, ["agent"], [
        "pre_shift_sr", "immediate_post_shift_sr", "performance_degradation",
        "recovered_post_shift_sr", "adaptation_gain", "recovery_rate_pct",
        "adaptation_speed_episodes", "unshifted_domain_sr", "phase3_compound_shift_sr",
        "unseen_tasks_post_shift_sr", "new_strategies_discovered",
        "mean_alarm_delay_episodes", "unshifted_false_alarms_p2",
    ])
    s3_sum.to_csv(proc_dir / "exp3_shift_summary.csv", index=False)

    s3_cusum = summarize_grouped(df_exp3_cusum, ["episode_idx", "domain"], ["cusum_stat", "alarm_fired"])
    s3_cusum.to_csv(proc_dir / "exp3_domain_cusum_summary.csv", index=False)

    s4 = summarize_grouped(df_exp4, ["corruption_rate_eta", "agent"], [
        "subopt_rollouts", "false_acceptance_rate", "candidate_false_promotion_rate",
        "false_rejection_rate", "active_validated_end_noisy", "clean_validated_end_noisy",
        "false_promoted_active_noisy", "lgg_contaminated_active_noisy", "lgg_merges_noisy",
        "memory_contamination_rate", "post_clean_contamination_rate",
        "noisy_phase_sr", "recovery_after_misleading_sr",
    ])
    s4.to_csv(proc_dir / "exp4_misleading_feedback_summary.csv", index=False)

    s5 = summarize_grouped(df_exp5, ["variant"], [
        "overall_stream_sr", "noisy_pre_shift_sr", "post_shift_recovered_sr",
        "heldout_unseen_sr", "pre_shift_contamination", "post_shift_contamination",
        "active_memory_items", "validated_strategies", "memory_kb",
    ])
    full_sr5 = float(s5.loc[s5["variant"] == "EVSA (Full)", "overall_stream_sr_mean"].iloc[0])
    s5["delta_vs_full_abs"] = s5["overall_stream_sr_mean"] - full_sr5
    s5.to_csv(proc_dir / "exp5_ablation_summary.csv", index=False)

    s6 = summarize_grouped(df_exp6, ["agent"], [
        "task_success_rate", "total_wall_time_ms", "eval_time_ms_per_task",
        "memory_kb", "active_memory_items", "total_rollouts",
        "total_strategy_evaluations", "prompt_tokens", "completion_tokens",
        "total_tokens", "estimated_api_cost_usd", "sr_per_10k_tokens",
    ])
    s6.to_csv(proc_dir / "exp6_efficiency_summary.csv", index=False)

    s_fair = summarize_grouped(df_fairness, ["agent"], [
        "overall_sr", "unseen_instance_sr", "unseen_compositional_sr",
    ])
    s_fair.to_csv(proc_dir / "sensitivity_execution_parity_summary.csv", index=False)

    proposed = "Proposed Agent E: EVSA"
    baselines = [
        "Baseline A: Static",
        "Baseline B: Conventional Memory",
        "Baseline C: Reflection",
        "Baseline D: Experience Replay",
    ]
    all_tests: List[Dict[str, Any]] = []

    def run_family(
        df_in: pd.DataFrame,
        col: str,
        fam_name: str,
        comps: Sequence[str],
        prop_lbl: str = proposed,
        agent_col: str = "agent",
    ) -> None:
        piv = df_in.pivot(index="seed", columns=agent_col, values=col).sort_index()
        fam_list = []
        for b in comps:
            res = paired_statistical_comparison(piv[prop_lbl].to_numpy(), piv[b].to_numpy(), prop_lbl, b)
            res["family"] = fam_name
            res["metric"] = col
            fam_list.append(res)
        all_tests.extend(apply_holm_bonferroni(fam_list, alpha=0.05))

    run_family(df_exp1_main, "task_success_rate", "RQ1_Common_Benchmark_SR", baselines)
    run_family(df_exp1_main, "unseen_instance_sr", "RQ1_Unseen_Instance_SR", baselines)
    run_family(df_exp1_main, "unseen_compositional_sr", "RQ1_Unseen_Compositional_SR", baselines)
    run_family(df_exp2_summary, "final_sr", "RQ2_Final_Episode_SR", baselines)
    run_family(df_exp2_summary, "block1_sr", "RQ2_Early_Block1_SR", baselines)
    run_family(df_exp3_summary, "recovered_post_shift_sr", "RQ4_Post_Shift_Recovered_SR", baselines)
    run_family(df_exp3_summary, "adaptation_gain", "RQ4_Adaptation_Gain", baselines)
    run_family(df_exp3_summary, "recovery_rate_pct", "RQ4_Recovery_Rate_Pct", baselines)
    run_family(df_exp3_summary, "unshifted_domain_sr", "RQ4_Unshifted_Domain_Retention", baselines)

    mem_baselines = baselines[1:]
    for eta in (0.10, 0.25, 0.35):
        sub4 = df_exp4[np.isclose(df_exp4["corruption_rate_eta"], eta)]
        run_family(sub4, "memory_contamination_rate", f"RQ6_Contamination_eta_{eta:.2f}", mem_baselines)
        run_family(sub4, "recovery_after_misleading_sr", f"RQ6_PostCleanRecovery_eta_{eta:.2f}", baselines)

    ab_comps = [
        "w/o Evidence Validation", "w/o Confidence Estimation",
        "w/o Applicability Conditions", "w/o Shift Detection", "w/o Memory Invalidation",
    ]
    run_family(df_exp5, "overall_stream_sr", "RQ3_Ablation_Overall_SR", ab_comps, prop_lbl="EVSA (Full)", agent_col="variant")
    run_family(df_exp5, "post_shift_recovered_sr", "RQ3_Ablation_PostShift_SR", ab_comps, prop_lbl="EVSA (Full)", agent_col="variant")
    run_family(df_exp5, "heldout_unseen_sr", "RQ3_Ablation_Unseen_SR", ab_comps, prop_lbl="EVSA (Full)", agent_col="variant")

    piv_fair = df_fairness.pivot(index="seed", columns="agent", values="overall_sr").sort_index()
    fair_tests = apply_holm_bonferroni([
        {
            **paired_statistical_comparison(
                piv_fair["EVSA (Stochastic Prompt Exec)"].to_numpy(),
                piv_fair["Replay (Stochastic Prompt Exec)"].to_numpy(),
                "EVSA (Stochastic Prompt Exec)",
                "Replay (Stochastic Prompt Exec)",
            ),
            "family": "Fairness_Execution_Parity",
            "metric": "overall_sr",
        },
        {
            **paired_statistical_comparison(
                piv_fair["EVSA (Direct Macro Exec)"].to_numpy(),
                piv_fair["Replay (Direct Macro Exec)"].to_numpy(),
                "EVSA (Direct Macro Exec)",
                "Replay (Direct Macro Exec)",
            ),
            "family": "Fairness_Execution_Parity",
            "metric": "overall_sr",
        },
    ])
    all_tests.extend(fair_tests)

    # Reproducible simultaneous global-74 Holm-Bonferroni adjustment across all pooled tests
    all_tests = apply_global_holm_bonferroni(all_tests, alpha=0.05)

    df_tests = pd.DataFrame(all_tests)
    stat_dir = exp_dir / "statistical_analysis"
    df_tests.to_csv(stat_dir / "paired_hypothesis_tests_holm.csv", index=False)

    tbl_dir = ROOT / "tables"
    s1.to_csv(tbl_dir / "table_exp1_generalization.csv", index=False)
    s2_sum.to_csv(tbl_dir / "table_exp2_continual_learning.csv", index=False)
    s3_sum.to_csv(tbl_dir / "table_exp3_distribution_shift.csv", index=False)
    s4.to_csv(tbl_dir / "table_exp4_feedback_corruption.csv", index=False)
    s5.to_csv(tbl_dir / "table_exp5_ablation.csv", index=False)
    s6.to_csv(tbl_dir / "table_exp6_efficiency.csv", index=False)
    s_fair.to_csv(tbl_dir / "table_sensitivity_fairness.csv", index=False)
    s1_30.to_csv(tbl_dir / "table_sensitivity_30seeds.csv", index=False)
    df_tests.to_csv(tbl_dir / "table_statistical_tests_all.csv", index=False)
    export_markdown_summary_tables(tbl_dir)

    elapsed = time.perf_counter() - t0
    cpu_model = detect_cpu_model()
    digest = {
        "manifest": {
            "version": evsa_pkg.__version__,
            "author": "Dev Parth",
            "role": "Independent Researcher",
            "affiliation": "Independent Research in Artificial Intelligence",
            "release_date": "2026-11-09",
            "execution_timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "cpu_model": cpu_model,
            "num_primary_seeds": 20,
            "num_stability_seeds": 30,
            "mdes_dz_80pct_power_n20": round(minimum_detectable_dz(20, 0.05, 0.80), 4),
            "mdes_dz_90pct_power_n20": round(minimum_detectable_dz(20, 0.05, 0.90), 4),
            "total_suite_runtime_sec": round(elapsed, 2),
        },
        "exp1": s1.to_dict(orient="records"),
        "exp1_30seeds": s1_30.to_dict(orient="records"),
        "exp2": s2_sum.to_dict(orient="records"),
        "exp3": s3_sum.to_dict(orient="records"),
        "exp4": s4.to_dict(orient="records"),
        "exp5": s5.to_dict(orient="records"),
        "exp6": s6.to_dict(orient="records"),
        "fairness_parity": s_fair.to_dict(orient="records"),
        "num_hypothesis_tests": len(df_tests),
        "num_significant_per_family_05": int(df_tests["significant_at_05"].sum()),
        "num_significant_primary_global74_05": int(df_tests["significant_global74_at_05"].sum()),
        "num_significant_sensitivity_t_holm_05": int(df_tests["sensitivity_t_significant_holm"].sum()),
        "num_significant_sensitivity_t_global74_05": int(df_tests["sensitivity_t_significant_holm_global74"].sum()),
        "num_significant_sensitivity_wilcoxon_global74_05": int(df_tests["sensitivity_wilcoxon_significant_holm_global74"].sum()),
    }
    (stat_dir / "statistical_digest.json").write_text(json.dumps(digest, indent=2), encoding="utf-8")
    (exp_dir / "experiment_manifest.json").write_text(json.dumps(digest["manifest"], indent=2), encoding="utf-8")
    logger.info(
        "Complete EVSA v%s suite finished in %.2f seconds (%d tests; %d per-family primary sig, %d global-74 primary sig, %d global-74 sensitivity-t sig).",
        evsa_pkg.__version__,
        elapsed,
        len(df_tests),
        int(df_tests["significant_at_05"].sum()),
        int(df_tests["significant_global74_at_05"].sum()),
        int(df_tests["sensitivity_t_significant_holm_global74"].sum()),
    )


if __name__ == "__main__":
    main()
