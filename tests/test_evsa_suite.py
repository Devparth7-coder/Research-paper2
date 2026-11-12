"""Comprehensive unit, hard-interface-boundary, and scientific integrity test suite for EVSA v2.0.2."""

from __future__ import annotations

from dataclasses import fields, replace
import inspect
import re
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import src.agents.baselines as baselines_mod
import src.agents.evsa_agent as evsa_agent_mod
import src.agents.experimentation as exp_mod
import src.benchmark.agent_interface as iface_mod
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
    AgentEpisodeOutcome,
    AgentEnvironmentInterface,
    CalibratedSymbolicLLMBackbone,
    NonStationaryTaskEnvironment,
    RolloutResult,
    StrategyRecord,
    StrategyStatus,
    TaskInstance,
)
from src.drift_detection import DistributionShiftDetector
from src.evaluation import (
    apply_global_holm_bonferroni,
    apply_holm_bonferroni,
    audit_evsa_repository,
    compute_accuracy,
    compute_paired_power,
    minimum_detectable_dz,
    paired_statistical_comparison,
    run_audited_episode,
    summarize_seed_distribution,
)
from src.memory import AdaptiveMemoryManager, HypothesisGenerator, StrategyRepository, make_strategy_id
from src.retrieval import StrategyRetriever
from src.validation import ConfidenceEstimator, EvidenceValidationEngine


def test_task_rollout_and_agent_outcome_have_zero_oracle_fields() -> None:
    """TaskInstance, RolloutResult, and AgentEpisodeOutcome must expose zero oracle fields."""
    task_field_names = {f.name for f in fields(TaskInstance)}
    rollout_field_names = {f.name for f in fields(RolloutResult)}
    outcome_field_names = {f.name for f in fields(AgentEpisodeOutcome)}
    forbidden = {
        "optimal_plan",
        "governing_archetype",
        "regime_id",
        "true_success",
        "true_reward",
        "step_matches",
        "alignment_score",
        "feedback_corrupted",
        "contaminated_active_ratio",
    }
    assert task_field_names.isdisjoint(forbidden)
    assert rollout_field_names.isdisjoint(forbidden)
    assert outcome_field_names.isdisjoint(forbidden)


def test_hard_interface_boundary_separates_agent_from_oracle() -> None:
    """Step 2: Agents must reject raw oracle environments and AgentEnvironmentInterface hides oracle state."""
    oracle_env = NonStationaryTaskEnvironment(seed=1001, feedback_corruption_rate=0.25)
    task = oracle_env.sample_task_stream(1, stream_seed=42)[0]
    agent_iface = oracle_env.make_agent_interface()

    # 1. AgentEnvironmentInterface has no __dict__ and exposes zero oracle attributes
    assert not hasattr(agent_iface, "__dict__")
    for hidden_attr in (
        "regime_tables",
        "_task_meta",
        "_rollout_audit_log",
        "get_optimal_plan",
        "get_task_oracle",
        "resolve_oracle_for_task",
        "feedback_corruption_rate",
        "finalize_episode_telemetry",
        "pop_episode_audits",
    ):
        assert not hasattr(agent_iface, hidden_attr), f"Interface leaked {hidden_attr}"

    # 2. Every agent class strictly rejects a raw NonStationaryTaskEnvironment with TypeError
    for ag in (
        StaticAgent(seed=1),
        ConventionalMemoryAgent(seed=1),
        ReflectionAgent(seed=1),
        ExperienceReplayAgent(seed=1),
        EVSAAgent(seed=1),
    ):
        with pytest.raises(TypeError, match="Hard oracle boundary violation"):
            ag.act_on_episode(task, oracle_env)  # type: ignore[arg-type]

        # Valid AgentEnvironmentInterface succeeds and returns AgentEpisodeOutcome
        out = ag.act_on_episode(task, agent_iface, max_trials=1, learn=False)
        assert isinstance(out, AgentEpisodeOutcome)


def test_zero_oracle_leakage_across_all_agent_and_backbone_modules() -> None:
    """Verify zero oracle classes or attributes appear in any agent, memory, or backbone code."""
    audited_modules = (
        iface_mod,
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
        "NonStationaryTaskEnvironment",
        "EpisodeTelemetry",
        "AuditOutcome",
        "optimal_plan",
        "governing_archetype",
        "regime_id",
        "true_success",
        "true_reward",
        "get_optimal_plan",
        "get_task_oracle",
    )
    for mod in audited_modules:
        src_txt = inspect.getsource(mod)
        # Strip module-level docstring before scanning executable code
        tree_doc = mod.__doc__ or ""
        if tree_doc:
            src_txt = src_txt.replace(tree_doc, "", 1)
        lines = [ln for ln in src_txt.splitlines() if not ln.strip().startswith("#")]
        joined = "\n".join(lines)
        for tok in forbidden_tokens:
            assert tok not in joined, f"Forbidden token {tok!r} found in {mod.__name__}"


def test_backbone_does_not_know_latent_xor_parity() -> None:
    """Zero-shot backbone uses linear inclusive-OR heuristic and does not know XOR on c=7."""
    env = NonStationaryTaskEnvironment(seed=1001)
    t_arch7 = env.sample_task_stream(1, allowed_archetypes=[7], stream_seed=77)[0]
    assert t_arch7.structural_predicates[:3] == (1, 1, 1)
    _, _, opt_arch7 = env.get_task_oracle(t_arch7.task_id)

    bb = CalibratedSymbolicLLMBackbone(seed=42, base_step_competence=0.67)
    step3_probs = bb._base_step_probs(t_arch7, 3)
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
    """Disabled ConfidenceEstimator must return raw empirical mean, NOT constant 0.75."""
    est_bayes = ConfidenceEstimator(credible_delta=0.10, gamma_base=1.0, enabled=True)
    est_emp = ConfidenceEstimator(credible_delta=0.10, gamma_base=1.0, enabled=False)

    rec = StrategyRecord(
        strategy_id="s1",
        domain="api_orchestration",
        preconditions=((0, 1),),
        embedding=np.ones(20) / np.sqrt(20),
        operator_plan=(0, 1, 2, 0),
    )
    assert np.isclose(est_emp.posterior_mean(rec), 0.50)
    assert np.isclose(est_emp.lower_credible_bound(rec), 0.50)

    est_emp.update_posterior(rec, 1, 1)
    for ep in (2, 3, 4):
        est_emp.update_posterior(rec, 0, ep)
    assert np.isclose(est_emp.posterior_mean(rec), 0.25)
    assert np.isclose(est_emp.lower_credible_bound(rec), 0.25)
    assert np.isclose(est_emp.upper_credible_bound(rec), 0.25)

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
    """Broadening a VALIDATED rule via LGG requires 1 confirming trial before re-promotion."""
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

    for ep in range(3):
        rec = mgr.admit_or_merge_candidate(repo, t1, p1, opt1, 1, ep, val)
    assert rec.status == StrategyStatus.VALIDATED
    assert len(rec.preconditions) == 5

    merged = mgr.admit_or_merge_candidate(repo, t2, p2, opt1, 1, 3, val)
    assert len(merged.preconditions) == 4
    assert repo.lgg_merges == 1
    assert repo.lgg_revalidations_triggered == 1
    assert merged.status == StrategyStatus.CANDIDATE

    est.update_posterior(merged, 1, 4)
    assert val.step_evidence(merged, 1, est, 4) == StrategyStatus.VALIDATED


def test_retriever_tau_rel_gate_and_frozen_eval_validated_only() -> None:
    """Retriever excludes VALIDATED with LCB < tau_rel, and learn=False skips CANDIDATEs."""
    est = ConfidenceEstimator()
    ret = StrategyRetriever(est, use_applicability=True, tau_rel=0.45)
    repo = StrategyRepository()
    emb = np.zeros(20)
    emb[0] = 1.0

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
    assert len(vals) == 0
    assert len(cands) == 1

    exp = ExperimentationEngine()
    chosen_train, is_val_tr = exp.select_strategy_for_trial(vals, cands, 1, learn=True)
    assert chosen_train is rec_cand and not is_val_tr

    chosen_eval, is_val_ev = exp.select_strategy_for_trial(vals, cands, 1, learn=False)
    assert chosen_eval is None and not is_val_ev


def test_cusum_cooldown_and_selective_shift_quarantine() -> None:
    """CUSUM enforces warmup + cooldown, and shift quarantine uses failure/LCB evidence."""
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

    for ep in range(fired_ep + 1, fired_ep + 6):
        assert det.update("api_orchestration", 0.88, 0, ep) is False

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
    assert unfalsified_old.status == StrategyStatus.CANDIDATE


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
            tel_tr = run_audited_episode(ag, t, run_env, "test_tr", max_trials=2, learn=True)
            assert 1 <= tel_tr.trials_used <= 2
            tel_ev = run_audited_episode(ag, t, run_env, "test_ev", max_trials=1, learn=False)
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


def test_reproducible_global74_holm_ledger() -> None:
    """Verify single-pass direct primary Holm (per-family & Global-74) plus labelled sensitivity families."""
    from src.evaluation.metrics_and_stats import _holm_adjust

    ledger_path = ROOT / "experiments" / "statistical_analysis" / "paired_hypothesis_tests_holm.csv"
    table_path = ROOT / "tables" / "table_statistical_tests_all.csv"
    assert ledger_path.exists() and table_path.exists()

    df = pd.read_csv(ledger_path)
    df_tbl = pd.read_csv(table_path)
    assert len(df) == 74
    assert df["family"].nunique() == 19
    required_cols = {
        "primary_pvalue_raw",
        "primary_pvalue_holm",
        "primary_pvalue_holm_global74",
        "significant_global74_at_05",
        "sensitivity_t_pvalue_holm_global74",
        "sensitivity_t_significant_holm_global74",
        "sensitivity_wilcoxon_pvalue_holm_global74",
        "sensitivity_wilcoxon_significant_holm_global74",
        "t_pvalue_holm_global74",
        "wilcoxon_pvalue_holm_global74",
    }
    assert required_cols.issubset(df.columns)
    assert required_cols.issubset(df_tbl.columns)

    # 1. Verify pre-adjustment primary selection rule
    expected_raw = np.where(df["shapiro_p"] < 0.05, df["wilcoxon_pvalue"], df["t_pvalue"])
    assert np.allclose(df["primary_pvalue_raw"].to_numpy(), expected_raw, atol=1e-15)

    # 2. Verify each of the 19 families applies _holm_adjust directly once to primary_pvalue_raw
    for _, grp in df.groupby("family", sort=False):
        _, direct_fam = _holm_adjust(grp["primary_pvalue_raw"].to_numpy(), alpha=0.05)
        assert np.allclose(grp["primary_pvalue_holm"].to_numpy(), direct_fam, atol=1e-15)

    # 3. Verify Global-74 primary applies _holm_adjust directly once to all 74 primary_pvalue_raw values
    _, direct_glob74 = _holm_adjust(df["primary_pvalue_raw"].to_numpy(), alpha=0.05)
    assert np.allclose(df["primary_pvalue_holm_global74"].to_numpy(), direct_glob74, atol=1e-15)

    # Confirm Global-74 primary is NOT a splice of separately adjusted t and Wilcoxon vectors
    spliced_glob74 = np.where(
        df["primary_test"] == "Wilcoxon signed-rank",
        df["sensitivity_wilcoxon_pvalue_holm_global74"],
        df["sensitivity_t_pvalue_holm_global74"],
    )
    assert not np.allclose(df["primary_pvalue_holm_global74"].to_numpy(), spliced_glob74, atol=1e-9)

    recomputed = pd.DataFrame(apply_global_holm_bonferroni(df.to_dict(orient="records"), alpha=0.05))
    assert np.allclose(df["primary_pvalue_holm_global74"], recomputed["primary_pvalue_holm_global74"], atol=1e-15)
    assert np.allclose(df["sensitivity_t_pvalue_holm_global74"], recomputed["sensitivity_t_pvalue_holm_global74"], atol=1e-15)
    assert np.allclose(df["sensitivity_wilcoxon_pvalue_holm_global74"], recomputed["sensitivity_wilcoxon_pvalue_holm_global74"], atol=1e-15)
    assert (df["significant_global74_at_05"] == recomputed["significant_global74_at_05"]).all()


def test_documentation_references_exact_source_symbols() -> None:
    """Verify consolidated 7 phase dossiers, exact source/test symbols, pinned deps, and CI workflow."""
    expected_phase_docs = [
        f"PHASE_{i}_{suffix}.md"
        for i, suffix in (
            (1, "PROBLEM_AND_THEORY"),
            (2, "ARCHITECTURE_AND_ALGORITHMS"),
            (3, "BENCHMARK_AND_BASELINES"),
            (4, "EXPERIMENTS_AND_ABLATIONS"),
            (5, "STATISTICAL_ANALYSIS_AND_POWER"),
            (6, "EFFICIENCY_AND_FAIRNESS_AUDIT"),
            (7, "MANUSCRIPT_AND_RELEASE_AUDIT"),
        )
    ]
    actual_docs = sorted(p.name for p in (ROOT / "docs").glob("*.md"))
    assert actual_docs == sorted(expected_phase_docs), (
        f"docs/ must contain strictly the 7 authoritative phase dossiers, got {actual_docs}"
    )

    forbidden_duplicate_outlines = {
        "PHASE_1_LITERATURE.md",
        "PHASE_2_PROBLEM_AND_RQS.md",
        "PHASE_3_MATH_AND_ARCHITECTURE.md",
        "PHASE_4_BENCHMARK_AND_BASELINES.md",
        "PHASE_5_EXPERIMENTS.md",
        "PHASE_6_STATISTICS_AND_ABLATIONS.md",
        "PHASE_7_AUDIT_AND_RELEASE.md",
    }
    all_md_names = {p.name for p in ROOT.rglob("*.md") if ".git" not in p.parts}
    assert forbidden_duplicate_outlines.isdisjoint(all_md_names), (
        f"Duplicate phase outline files remain: {sorted(forbidden_duplicate_outlines & all_md_names)}"
    )

    release_notes = sorted(p.name for p in ROOT.glob("RELEASE_NOTES_*.md"))
    assert release_notes == ["RELEASE_NOTES_v2.0.2.md"], (
        f"Expected strictly RELEASE_NOTES_v2.0.2.md at repo root, found {release_notes}"
    )

    doc_paths = [
        ROOT / "FINAL_AUDIT_REPORT.md",
        ROOT / "REPRODUCIBILITY.md",
        ROOT / "README.md",
        ROOT / "TIMELINE.md",
        ROOT / "RELEASE_NOTES_v2.0.2.md",
        *(ROOT / "docs" / name for name in expected_phase_docs),
    ]
    known_tests = {name for name in globals() if name.startswith("test_")} | {
        "test_independent_researcher_attribution_zero_college_mentions",
        "test_cross_platform_cpu_detection_and_utf8_safety",
        "test_wilcoxon_ties_zeros_and_percentile_bootstrap",
        "test_quarantine_tombstone_and_readmission_lifecycle",
    }
    for doc_path in doc_paths:
        assert doc_path.exists(), f"Missing documentation file {doc_path}"
        txt = doc_path.read_text(encoding="utf-8")
        assert "21-section" not in txt, f"Stale section count in {doc_path.name}"
        assert "EVSA-Research-Paper" not in txt, f"Stale repo slug in {doc_path.name}"
        for stale_16 in ("16-test", "16-Test", "16 / 16", "16/16", "16 automated tests", "# 16 unit"):
            assert stale_16 not in txt, f"Stale 16-test claim '{stale_16}' in {doc_path.name}"
        for rel_src in re.findall(r"`(src/[A-Za-z0-9_./]+\.py)`", txt):
            assert (ROOT / rel_src).exists(), f"{doc_path.name} references missing file {rel_src}"
        for tname in re.findall(r"`(test_[A-Za-z0-9_]+)`", txt):
            if tname != "test_":
                assert tname in known_tests, f"{doc_path.name} references missing test {tname}"

    gitignore_path = ROOT / ".gitignore"
    assert gitignore_path.exists()
    gitignore_lines = {
        ln.strip()
        for ln in gitignore_path.read_text(encoding="utf-8").splitlines()
        if ln.strip() and not ln.strip().startswith("#")
    }
    for must_track in ("experiments/", "figures/", "tables/", "paper/", "src/", "docs/"):
        assert must_track not in gitignore_lines, f"{must_track} must never be ignored in .gitignore"

    ci_txt = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    assert "ubuntu-latest" in ci_txt and "windows-latest" in ci_txt
    assert "python -m pytest -v tests/test_evsa_suite.py" in ci_txt

    cff_txt = (ROOT / "CITATION.cff").read_text(encoding="utf-8")
    verifier_txt = (ROOT / "experiments" / "verify_github_release.py").read_text(encoding="utf-8")
    tex_txt = (ROOT / "paper" / "research_paper.tex").read_text(encoding="utf-8")
    assert 'repository-code: "https://github.com/Devparth7-coder/EVSA"' in cff_txt
    assert 'default="Devparth7-coder/EVSA"' in verifier_txt
    assert "EVSA-Research-Paper" not in verifier_txt
    assert "https://github.com/Devparth7-coder/EVSA" in tex_txt
    assert "EVSA-Research-Paper" not in tex_txt
    assert "16-test" not in tex_txt and "18-test" in tex_txt
    req_lines = [
        ln.strip() for ln in (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()
        if ln.strip() and not ln.strip().startswith("#")
    ]
    assert len(req_lines) >= 6
    assert all("==" in ln for ln in req_lines), "All requirements.txt entries must be version-pinned with =="


def test_independent_researcher_attribution_zero_college_mentions() -> None:
    """Verify standardized independent researcher attribution and zero college/student mentions repo-wide."""
    forbidden_terms = (
        "mmmut",
        "madan mohan",
        "malaviya",
        "gorakhpur",
        "uttar pradesh",
        "department of computer science",
        "student / independent",
        "student researcher",
    )
    self_path = Path(__file__).resolve()
    scan_exts = {".md", ".cff", ".json", ".tex", ".bib", ".py", ".yml"}
    skip_dirs = {".git", "__pycache__", ".pytest_cache", ".venv"}

    for path in ROOT.rglob("*"):
        if not path.is_file() or path.resolve() == self_path:
            continue
        if any(part in skip_dirs for part in path.parts):
            continue
        if path.suffix.lower() not in scan_exts:
            continue
        lower_txt = path.read_text(encoding="utf-8").lower()
        for term in forbidden_terms:
            assert term not in lower_txt, f"Forbidden affiliation term '{term}' found in {path.relative_to(ROOT)}"

    for required_file in (
        ROOT / "README.md",
        ROOT / "CITATION.cff",
        ROOT / "paper" / "research_paper.tex",
        ROOT / "src" / "__init__.py",
    ):
        content = required_file.read_text(encoding="utf-8")
        if required_file.name == "CITATION.cff":
            assert 'family-names: "Parth"' in content and 'given-names: "Dev"' in content
        else:
            assert "Dev Parth" in content
        assert "Independent Researcher" in content
        assert "Independent Research in Artificial Intelligence" in content


def test_cross_platform_cpu_detection_and_utf8_safety(monkeypatch, tmp_path) -> None:
    """Verify run_complete_suite uses cross-platform CPU detection without lscpu or shell=True."""
    import contextlib
    import importlib.util
    import platform
    import subprocess
    import sys
    import types

    for checked_script in (
        ROOT / "experiments" / "run_complete_suite.py",
        ROOT / "paper" / "build_docx.py",
        ROOT / "experiments" / "verify_github_release.py",
    ):
        src_txt = checked_script.read_text(encoding="utf-8")
        assert "lscpu" not in src_txt
        assert "shell=True" not in src_txt

    runner_path = ROOT / "experiments" / "run_complete_suite.py"
    spec = importlib.util.spec_from_file_location("_evsa_runner_mod", runner_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    # 1. Actual host platform check: must return a non-empty string without assuming CPU vendor
    cpu_now = mod.detect_cpu_model()
    assert isinstance(cpu_now, str) and len(cpu_now.strip()) > 0

    # 2. Windows registry primary path (mocked winreg so test runs identically on Linux & Windows)
    fake_winreg = types.SimpleNamespace(
        HKEY_LOCAL_MACHINE=1,
        OpenKey=lambda root, subkey: contextlib.nullcontext("key_handle"),
        QueryValueEx=lambda handle, value_name: ("ControlledRegistryCPU-Model 3.40GHz", 1),
    )
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setitem(sys.modules, "winreg", fake_winreg)
    monkeypatch.setenv("PROCESSOR_IDENTIFIER", "ShouldNotBeUsedWhenRegistrySucceeds")
    assert mod.detect_cpu_model() == "ControlledRegistryCPU-Model 3.40GHz"

    # 3. Windows fallback path: block winreg (sys.modules["winreg"] = None) so real Windows
    # runners also exercise PROCESSOR_IDENTIFIER fallback deterministically
    monkeypatch.setitem(sys.modules, "winreg", None)
    monkeypatch.setenv("PROCESSOR_IDENTIFIER", "ControlledEnvFallbackCPU-Model")
    assert mod.detect_cpu_model() == "ControlledEnvFallbackCPU-Model"

    # 4. Linux /proc/cpuinfo path via controlled temporary UTF-8 file
    fake_cpuinfo = tmp_path / "cpuinfo"
    fake_cpuinfo.write_text(
        "processor\t: 0\nmodel name\t: ControlledLinuxCPU @ 2.90GHz\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(sys, "platform", "linux")
    assert mod.detect_cpu_model(cpuinfo_path=fake_cpuinfo) == "ControlledLinuxCPU @ 2.90GHz"

    # 5. macOS sysctl path without shell=True
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda cmd, **kwargs: types.SimpleNamespace(
            returncode=0, stdout="ControlledAppleSilicon-M3\n"
        ),
    )
    assert mod.detect_cpu_model() == "ControlledAppleSilicon-M3"

    # 6. Generic platform fallback check
    monkeypatch.setattr(sys, "platform", "freebsd14")
    monkeypatch.setattr(platform, "processor", lambda: "")
    monkeypatch.setattr(platform, "machine", lambda: "amd64")
    assert mod.detect_cpu_model() == "amd64"


def test_wilcoxon_ties_zeros_and_percentile_bootstrap() -> None:
    """Verify Wilcoxon exact/tie/zero method accounting and 5,000-resample percentile bootstrap CIs."""
    from scipy import stats

    # Case 1: Strictly untied non-zero differences -> exact Wilcoxon null distribution
    y_base = np.zeros(20)
    x_untied = np.array([0.01 * (i + 1) * (1 if i % 3 != 0 else -1) for i in range(20)])
    res_untied = paired_statistical_comparison(x_untied, y_base, "A", "B")
    exact_ref = stats.wilcoxon(x_untied, zero_method="wilcox", alternative="two-sided", method="exact")
    assert res_untied["n_zero_diffs"] == 0
    assert res_untied["wilcoxon_n_nonzero"] == 20
    assert res_untied["wilcoxon_has_ties"] is False
    assert res_untied["wilcoxon_method"] == "exact"
    assert np.isclose(res_untied["wilcoxon_pvalue"], exact_ref.pvalue, atol=1e-15)

    # Case 2: Zero differences present alongside untied non-zero differences
    x_with_zeros = x_untied.copy()
    x_with_zeros[:3] = 0.0
    res_zeros = paired_statistical_comparison(x_with_zeros, y_base, "A", "B")
    assert res_zeros["n_zero_diffs"] == 3
    assert res_zeros["wilcoxon_n_nonzero"] == 17
    assert res_zeros["wilcoxon_has_ties"] is False
    assert res_zeros["wilcoxon_method"] == "conditional_exact_wilcox_zeros_dropped"

    # Case 3: Discrete ties with n_nonzero > 13 -> tie-adjusted asymptotic normal in SciPy 1.17.1
    x_tied_large = np.array([0.05] * 10 + [0.10] * 8 + [-0.05] * 2)
    res_tied_large = paired_statistical_comparison(x_tied_large, y_base, "A", "B")
    asymp_ref = stats.wilcoxon(
        x_tied_large, zero_method="wilcox", alternative="two-sided", method="asymptotic"
    )
    assert res_tied_large["wilcoxon_has_ties"] is True
    assert res_tied_large["wilcoxon_method"] == "asymptotic_tie_adjusted"
    assert np.isclose(res_tied_large["wilcoxon_pvalue"], asymp_ref.pvalue, atol=1e-15)

    # Case 4: Discrete ties with n_nonzero <= 13 -> exact sign-permutation distribution
    x_tied_small = np.array([0.05] * 6 + [0.10] * 4 + [-0.05] * 2 + [0.0] * 8)
    res_tied_small = paired_statistical_comparison(x_tied_small, y_base, "A", "B")
    assert res_tied_small["n_zero_diffs"] == 8
    assert res_tied_small["wilcoxon_n_nonzero"] == 12
    assert res_tied_small["wilcoxon_has_ties"] is True
    assert res_tied_small["wilcoxon_method"] == "exact_sign_permutation"

    # Case 5: Fewer than 3 non-zero differences -> conservative insufficient_nonzero fallback
    x_degenerate = np.array([0.10, -0.05] + [0.0] * 18)
    res_deg = paired_statistical_comparison(x_degenerate, y_base, "A", "B")
    assert res_deg["wilcoxon_method"] == "insufficient_nonzero"
    assert res_deg["wilcoxon_pvalue"] == 1.0

    # Percentile bootstrap verification (n_boot=5000, boot_seed=2026)
    vals = np.linspace(0.40, 0.72, 20)
    summary = summarize_seed_distribution(vals, n_boot=5000, boot_seed=2026)
    rng = np.random.default_rng(2026)
    boot_means = rng.choice(vals, size=(5000, vals.size), replace=True).mean(axis=1)
    lo, hi = np.percentile(boot_means, [2.5, 97.5])
    assert np.isclose(summary["boot_ci95_low"], lo, atol=1e-15)
    assert np.isclose(summary["boot_ci95_high"], hi, atol=1e-15)


def test_quarantine_tombstone_and_readmission_lifecycle() -> None:
    """Verify QUARANTINED tombstones resist failed-rollout resurrection and preserve audit history on re-admission."""
    est = ConfidenceEstimator()
    val = EvidenceValidationEngine(min_trials=3)
    mgr = AdaptiveMemoryManager(est, use_applicability=False, tau_quarantine_shift=0.58)
    repo = StrategyRepository()

    preconds = ((0, 1), (1, 0), (2, 1))
    plan = (2, 1, 3, 0)
    sid = make_strategy_id("api_orchestration", preconds, plan)
    task = TaskInstance(
        task_id="t_q_1",
        episode_idx=10,
        domain="api_orchestration",
        structural_predicates=(1, 0, 1, 0, 1),
        surface_topic_id=0,
        context_features=np.zeros(12),
        query_embedding=np.ones(20) / np.sqrt(20),
    )

    # Admit and promote strategy across 3 clean successes
    rec = None
    for ep in (10, 11, 12):
        rec = mgr.admit_or_merge_candidate(repo, task, preconds, plan, 1, ep, val)
    assert rec is not None and rec.strategy_id == sid
    assert rec.status == StrategyStatus.VALIDATED
    assert repo.total_promoted == 1

    # Quarantine on shift discount after a consecutive failure
    rec.consecutive_failures = 1
    mgr.apply_domain_shift_discount(repo, "api_orchestration", gamma_shift=0.35, tau_lcb=0.48, episode_idx=42)
    assert rec.status == StrategyStatus.QUARANTINED
    assert rec.invalidated_episode == 42
    assert rec.quarantine_count == 1
    assert repo.total_quarantined == 1

    # Failed rollout on identical (domain, preconditions, plan) MUST NOT resurrect tombstone
    after_fail = mgr.admit_or_merge_candidate(repo, task, preconds, plan, 0, 43, val)
    assert after_fail is rec
    assert rec.status == StrategyStatus.QUARANTINED
    assert rec.invalidated_episode == 42
    assert rec.quarantine_count == 1

    # Positive evidence re-admits the SAME record in-place to probationary CANDIDATE, preserving audit history
    readmitted = mgr.admit_or_merge_candidate(repo, task, preconds, plan, 1, 50, val)
    assert readmitted is rec
    assert repo.get(sid) is rec
    assert rec.status == StrategyStatus.CANDIDATE
    assert rec.n_trials == 1
    assert rec.invalidated_episode == 42
    assert rec.quarantine_count == 1

    # Must complete full sequential validation (2 more positive trials -> n_trials == 3) before re-promotion
    mgr.admit_or_merge_candidate(repo, task, preconds, plan, 1, 51, val)
    assert rec.status == StrategyStatus.CANDIDATE
    mgr.admit_or_merge_candidate(repo, task, preconds, plan, 1, 52, val)
    assert rec.status == StrategyStatus.VALIDATED
    assert repo.total_promoted == 2

    # Second quarantine increments quarantine_count cleanly
    rec.consecutive_failures = 1
    mgr.apply_domain_shift_discount(repo, "api_orchestration", gamma_shift=0.35, tau_lcb=0.48, episode_idx=90)
    assert rec.status == StrategyStatus.QUARANTINED
    assert rec.invalidated_episode == 90
    assert rec.quarantine_count == 2

    # Direct StrategyRepository.add guard also preserves tombstone provenance if a fresh object shares sid
    replacement = StrategyRecord(
        strategy_id=sid,
        domain="api_orchestration",
        preconditions=preconds,
        embedding=np.ones(20) / np.sqrt(20),
        operator_plan=plan,
        status=StrategyStatus.CANDIDATE,
        created_episode=95,
    )
    repo.add(replacement)
    assert replacement.invalidated_episode == 90
    assert replacement.quarantine_count == 2
