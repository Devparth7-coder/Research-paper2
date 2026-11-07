"""Comprehensive unit and scientific integrity test suite for EVSA v2.0.0."""

from __future__ import annotations

from dataclasses import fields, replace
import inspect
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import src.agents.baselines as baselines_mod
import src.agents.evsa_agent as evsa_agent_mod
import src.agents.experimentation as exp_mod
import src.benchmark.llm_backbone as backbone_mod
import src.drift_detection.cusum_detector as cusum_mod
import src.memory.hypothesis_generator as hyp_mod
import src.memory.memory_manager as mem_mgr_mod
import src.memory.repository as repo_mod
import src.retrieval.retriever as ret_mod
import src.validation.confidence as conf_mod
import src.validation.sprt_validator as sprt_mod

from src.agents import (
    ConventionalMemoryAgent,
    EVSAAgent,
    ExperienceReplayAgent,
    ExperimentationEngine,
    ReflectionAgent,
    StaticAgent,
)
from src.benchmark import (
    CalibratedSymbolicLLMBackbone,
    NonStationaryTaskEnvironment,
    RolloutResult,
    StrategyRecord,
    StrategyStatus,
    TaskInstance,
)
from src.drift_detection import DistributionShiftDetector
from src.evaluation import (
    apply_holm_bonferroni,
    audit_evsa_repository,
    compute_accuracy,
    compute_paired_power,
    minimum_detectable_dz,
    paired_statistical_comparison,
)
from src.memory import AdaptiveMemoryManager, HypothesisGenerator, StrategyRepository
from src.retrieval import StrategyRetriever
from src.validation import ConfidenceEstimator, EvidenceValidationEngine


def test_task_and_rollout_have_zero_oracle_fields() -> None:
    """Issue 1: TaskInstance and RolloutResult must expose zero oracle/ground-truth attributes."""
    task_field_names = {f.name for f in fields(TaskInstance)}
    rollout_field_names = {f.name for f in fields(RolloutResult)}
    forbidden = {
        "optimal_plan",
        "governing_archetype",
        "regime_id",
        "true_success",
        "true_reward",
        "step_matches",
        "alignment_score",
        "feedback_corrupted",
    }
    assert task_field_names.isdisjoint(forbidden)
    assert rollout_field_names.isdisjoint(forbidden)


def test_zero_oracle_leakage_across_all_agent_and_backbone_modules() -> None:
    """Issue 1: Verify zero oracle identifiers appear anywhere in agent/memory/backbone code."""
    audited_modules = (
        backbone_mod,
        conf_mod,
        sprt_mod,
        cusum_mod,
        ret_mod,
        repo_mod,
        hyp_mod,
        mem_mgr_mod,
        exp_mod,
        baselines_mod,
        evsa_agent_mod,
    )
    forbidden_tokens = (
        "optimal_plan",
        "governing_archetype",
        "regime_id",
        "true_success",
        "true_reward",
        "get_optimal_plan",
        "get_task_oracle",
    )
    for mod in audited_modules:
        # Check executable statements (excluding module docstrings)
        lines = [
            ln for ln in inspect.getsource(mod).splitlines()
            if not ln.strip().startswith("#")
        ]
        joined = "\n".join(lines)
        for tok in forbidden_tokens:
            assert tok not in joined, f"Forbidden token {tok!r} found in {mod.__name__}"


def test_backbone_does_not_know_latent_xor_parity() -> None:
    """Issue 1: Zero-shot backbone uses linear inclusive-OR heuristic and does not know XOR on c=7."""
    env = NonStationaryTaskEnvironment(seed=1001)
    t_arch7 = env.sample_task_stream(1, allowed_archetypes=[7], stream_seed=77)[0]
    assert t_arch7.structural_predicates[:3] == (1, 1, 1)
    _, _, opt_arch7 = env.get_task_oracle(t_arch7.task_id)

    bb = CalibratedSymbolicLLMBackbone(seed=42, base_step_competence=0.67)
    step3_probs = bb._base_step_probs(t_arch7, 3)
    # On c=7 (p0=1, p1=1), true XOR is 0 whereas linear inclusive-OR puts prior mass on 1
    assert int(np.argmax(step3_probs)) != opt_arch7[3]


def test_environment_determinism_and_regime_invariance() -> None:
    env_a = NonStationaryTaskEnvironment(seed=1001)
    env_b = NonStationaryTaskEnvironment(seed=1001)
    s_a = env_a.sample_task_stream(30, stream_seed=77)
    s_b = env_b.sample_task_stream(30, stream_seed=77)
    for ta, tb in zip(s_a, s_b):
        assert ta.task_id == tb.task_id
        assert env_a.get_task_oracle(ta.task_id) == env_b.get_task_oracle(tb.task_id)
        assert np.allclose(ta.query_embedding, tb.query_embedding)
    for arch in range(8):
        assert env_a.get_optimal_plan("resource_planning", arch, 0) == env_a.get_optimal_plan(
            "resource_planning", arch, 1
        )
        assert env_a.get_optimal_plan("api_orchestration", arch, 0) != env_a.get_optimal_plan(
            "api_orchestration", arch, 1
        )


def test_confidence_estimator_bayesian_vs_empirical_ablation() -> None:
    """Issue 2: Disabled ConfidenceEstimator must return raw empirical mean, NOT constant 0.75."""
    est_bayes = ConfidenceEstimator(credible_delta=0.10, gamma_base=1.0, enabled=True)
    est_emp = ConfidenceEstimator(credible_delta=0.10, gamma_base=1.0, enabled=False)

    rec = StrategyRecord(
        strategy_id="s1",
        domain="api_orchestration",
        preconditions=((0, 1),),
        embedding=np.ones(20) / np.sqrt(20),
        operator_plan=(0, 1, 2, 0),
    )
    # Untested record -> 0.50 empirical prior
    assert np.isclose(est_emp.posterior_mean(rec), 0.50)
    assert np.isclose(est_emp.lower_credible_bound(rec), 0.50)

    # 1 success, 3 failures -> raw empirical frequency = 0.25 (NOT 0.75!)
    est_emp.update_posterior(rec, 1, 1)
    for ep in (2, 3, 4):
        est_emp.update_posterior(rec, 0, ep)
    assert np.isclose(est_emp.posterior_mean(rec), 0.25)
    assert np.isclose(est_emp.lower_credible_bound(rec), 0.25)
    assert np.isclose(est_emp.upper_credible_bound(rec), 0.25)

    # Bayesian estimator produces strict credible interval lcb < mean < ucb
    rec_b = StrategyRecord(
        strategy_id="s1b",
        domain="api_orchestration",
        preconditions=((0, 1),),
        embedding=np.ones(20) / np.sqrt(20),
        operator_plan=(0, 1, 2, 0),
    )
    for ep in (1, 2, 3):
        est_bayes.update_posterior(rec_b, 1, ep)
    assert 0.50 < est_bayes.lower_credible_bound(rec_b) < est_bayes.posterior_mean(rec_b) < est_bayes.upper_credible_bound(rec_b)


def test_discounted_sprt_promotion_and_quarantine_gates() -> None:
    est = ConfidenceEstimator(credible_delta=0.10, gamma_base=0.985)
    val = EvidenceValidationEngine(min_trials=3, tau_lcb=0.48)
    rec = StrategyRecord(
        strategy_id="s2",
        domain="data_pipeline",
        preconditions=((0, 0),),
        embedding=np.ones(20) / np.sqrt(20),
        operator_plan=(1, 1, 1, 1),
    )
    est.update_posterior(rec, 1, 1)
    assert val.step_evidence(rec, 1, est, 1) == StrategyStatus.CANDIDATE
    est.update_posterior(rec, 1, 2)
    assert val.step_evidence(rec, 1, est, 2) == StrategyStatus.CANDIDATE
    est.update_posterior(rec, 1, 3)
    assert val.step_evidence(rec, 1, est, 3) == StrategyStatus.VALIDATED

    for ep in range(4, 9):
        est.update_posterior(rec, 0, ep)
        st = val.step_evidence(rec, 0, est, ep)
        if st == StrategyStatus.QUARANTINED:
            break
    assert rec.status == StrategyStatus.QUARANTINED


def test_safe_lgg_generalization_and_revalidation() -> None:
    """Issue 4: Broadening a VALIDATED rule via LGG requires 1 confirming trial before re-promotion."""
    env = NonStationaryTaskEnvironment(seed=42)
    tasks = env.sample_task_stream(2, allowed_archetypes=[0], stream_seed=11)
    t1 = tasks[0]
    _, _, opt1 = env.get_task_oracle(t1.task_id)
    t2 = replace(t1, structural_predicates=(0, 0, 0, 1 - t1.structural_predicates[3], t1.structural_predicates[4]))

    repo = StrategyRepository()
    est = ConfidenceEstimator()
    val = EvidenceValidationEngine()
    mgr = AdaptiveMemoryManager(est)

    p1 = tuple((i, b) for i, b in enumerate(t1.structural_predicates))
    p2 = tuple((i, b) for i, b in enumerate(t2.structural_predicates))

    # 3 consecutive successes on p1 promote record to VALIDATED
    for ep in range(3):
        rec = mgr.admit_or_merge_candidate(repo, t1, p1, opt1, 1, ep, val)
    assert rec.status == StrategyStatus.VALIDATED
    assert len(rec.preconditions) == 5

    # Merging with t2 broadens preconditions to 4 literals and steps back to CANDIDATE for re-verification
    merged = mgr.admit_or_merge_candidate(repo, t2, p2, opt1, 1, 3, val)
    assert len(merged.preconditions) == 4
    assert repo.lgg_merges == 1
    assert repo.lgg_revalidations_triggered == 1
    assert merged.status == StrategyStatus.CANDIDATE

    # One confirming trial on the broadened rule restores VALIDATED status
    est.update_posterior(merged, 1, 4)
    assert val.step_evidence(merged, 1, est, 4) == StrategyStatus.VALIDATED


def test_retriever_tau_rel_gate_and_frozen_eval_validated_only() -> None:
    """Issue 4: Retriever excludes VALIDATED with LCB < tau_rel, and learn=False skips CANDIDATEs."""
    est = ConfidenceEstimator()
    ret = StrategyRetriever(est, use_applicability=True, tau_rel=0.45)
    repo = StrategyRepository()
    emb = np.zeros(20)
    emb[0] = 1.0

    # Weak VALIDATED strategy with LCB < 0.45 (alpha=2, beta=3 -> LCB_0.10 = 0.14 < 0.45)
    rec_weak_val = StrategyRecord(
        strategy_id="weak_val",
        domain="system_diagnosis",
        preconditions=((0, 1), (1, 1), (2, 1)),
        embedding=emb.copy(),
        operator_plan=(0, 0, 0, 0),
        alpha=2.0,
        beta=3.0,
        status=StrategyStatus.VALIDATED,
    )
    # Candidate strategy matching same predicates
    rec_cand = StrategyRecord(
        strategy_id="cand_1",
        domain="system_diagnosis",
        preconditions=((0, 1), (1, 1), (2, 1)),
        embedding=emb.copy(),
        operator_plan=(1, 1, 1, 1),
        alpha=3.0,
        beta=1.0,
        status=StrategyStatus.CANDIDATE,
    )
    repo.add(rec_weak_val)
    repo.add(rec_cand)

    env = NonStationaryTaskEnvironment(seed=9)
    task = env.sample_task_stream(1, allowed_archetypes=[7], stream_seed=9)[0]
    task = replace(task, domain="system_diagnosis", query_embedding=emb.copy())

    vals, cands, _ = ret.retrieve(repo, task)
    assert len(vals) == 0  # Excluded by LCB < tau_rel!
    assert len(cands) == 1

    exp = ExperimentationEngine()
    chosen_train, is_val_tr = exp.select_strategy_for_trial(vals, cands, 1, learn=True)
    assert chosen_train is rec_cand and not is_val_tr

    chosen_eval, is_val_ev = exp.select_strategy_for_trial(vals, cands, 1, learn=False)
    assert chosen_eval is None and not is_val_ev


def test_cusum_cooldown_and_selective_shift_quarantine() -> None:
    """Issue 3: CUSUM enforces warmup + cooldown, and shift quarantine uses failure/LCB evidence (not age)."""
    det = DistributionShiftDetector(
        slack_nu=0.12, drift_threshold=1.45, min_obs_since_reset=3, cooldown_episodes=12
    )
    fired_ep = None
    for ep in range(6):
        if det.update("api_orchestration", 0.88, 0, ep):
            fired_ep = ep
            break
    assert fired_ep is not None and fired_ep >= 2
    assert det.cusum_stat.get("resource_planning", 0.0) == 0.0

    # Immediate subsequent failures inside cooldown window must NOT fire a duplicate alarm
    for ep in range(fired_ep + 1, fired_ep + 6):
        assert det.update("api_orchestration", 0.88, 0, ep) is False

    # Verify selective shift quarantine quarantines failing record regardless of young age (created_episode=99)
    est = ConfidenceEstimator()
    mgr = AdaptiveMemoryManager(est, tau_quarantine_shift=0.58)
    repo = StrategyRepository()
    failing_young = StrategyRecord(
        strategy_id="young_fail",
        domain="api_orchestration",
        preconditions=((0, 0), (1, 0), (2, 0)),
        embedding=np.ones(20) / np.sqrt(20),
        operator_plan=(0, 0, 0, 0),
        alpha=6.0,
        beta=2.0,
        consecutive_failures=1,
        status=StrategyStatus.VALIDATED,
        created_episode=99,
    )
    unfalsified_old = StrategyRecord(
        strategy_id="old_untested",
        domain="api_orchestration",
        preconditions=((0, 1), (1, 0), (2, 0)),
        embedding=np.ones(20) / np.sqrt(20),
        operator_plan=(2, 1, 0, 1),
        alpha=8.0,
        beta=1.0,
        consecutive_failures=0,
        status=StrategyStatus.VALIDATED,
        created_episode=5,
    )
    repo.add(failing_young)
    repo.add(unfalsified_old)
    mgr.apply_domain_shift_discount(repo, "api_orchestration", gamma_shift=0.35, tau_lcb=0.48, episode_idx=100)
    assert failing_young.status == StrategyStatus.QUARANTINED
    assert unfalsified_old.status == StrategyStatus.CANDIDATE  # Demoted for re-verification, NOT blindly wiped by age!


def test_budget_parity_and_contamination_audit() -> None:
    env = NonStationaryTaskEnvironment(seed=101)
    tasks = env.sample_task_stream(20, stream_seed=202)
    agents = [
        StaticAgent(seed=101),
        ConventionalMemoryAgent(seed=101),
        ReflectionAgent(seed=101),
        ExperienceReplayAgent(seed=101, direct_macro_execution=True),
        EVSAAgent(seed=101, stochastic_prompt_execution=True),
        EVSAAgent(seed=101),
    ]
    for ag in agents:
        run_env = NonStationaryTaskEnvironment(seed=303)
        run_env.copy_metadata_from(env)
        for t in tasks:
            tel_tr = ag.run_episode(t, run_env, "test_tr", max_trials=2, learn=True)
            assert 1 <= tel_tr.trials_used <= 2
            tel_ev = ag.run_episode(t, run_env, "test_ev", max_trials=1, learn=False)
            assert tel_ev.trials_used == 1
        if isinstance(ag, EVSAAgent):
            bd = audit_evsa_repository(ag.repo, env, current_regime=0)
            assert bd["n_validated"] == bd["clean_validated"] + bd["false_promoted_active"] + bd["lgg_contaminated_active"]


def test_statistical_power_and_holm_bonferroni() -> None:
    mdes = minimum_detectable_dz(n_seeds=20, alpha=0.05, target_power=0.80)
    assert 0.65 < mdes < 0.67
    assert compute_paired_power(4.5, n_seeds=20, alpha=0.05) > 0.9999

    x = np.linspace(0.55, 0.65, 20)
    y = np.linspace(0.28, 0.35, 20)
    comp = paired_statistical_comparison(x, y, "EVSA", "Baseline")
    adj = apply_holm_bonferroni([comp], alpha=0.05)
    assert adj[0]["significant_at_05"] is True
    assert compute_accuracy([1, 1, 0, 1]) == 0.75
