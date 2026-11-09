# Phase 2: EVSA Ten-Module Architecture & Core Algorithms (`v2.0.1`)

**Project**: *Evidence-Validated Strategy Adaptation for Memory-Augmented Agents Under Distribution Shift*  
**Author**: Dev Parth (Department of Computer Science and Engineering, MMMUT, Gorakhpur, Uttar Pradesh, India)  
**Release Version**: `v2.0.1` | **Date**: 7 November 2026  

---

## 1. Closed-Loop Module Mapping to `src/`

EVSA represents each reusable procedural hypothesis as a `StrategyRecord` (`src/benchmark/types.py`) governed by three epistemic tiers (`StrategyStatus.CANDIDATE`, `StrategyStatus.VALIDATED`, `StrategyStatus.QUARANTINED`):

| Module | Source File | Primary Class & Key Methods | Role in Closed-Loop Cycle |
| :--- | :--- | :--- | :--- |
| **1. Capability Rollout Interface** | `src/benchmark/agent_interface.py` | `AgentEnvironmentInterface.evaluate_plan` | Slot-restricted (`__slots__ = ("__weakref__",)`) agent-facing rollout handle with zero oracle state |
| **2. Non-Oracle Backbone** | `src/benchmark/llm_backbone.py` | `CalibratedSymbolicLLMBackbone.propose_plan`, `execute_guided_plan` | Observable prior (`0.67` step competence, linear OR on Step 3) and guided execution |
| **3. Hypothesis Generator** | `src/memory/hypothesis_generator.py` | `HypothesisGenerator.extract_preconditions`, `propose_exploratory_plan` | Observed step-predicate outcome salience and tombstone-avoiding exploration |
| **4. Tiered Strategy Store** | `src/memory/repository.py` | `StrategyRepository.add`, `quarantine`, `merge_via_lgg` | Three-tier storage, tombstone signatures, and precondition intersection |
| **5. Bayesian Credible Estimator** | `src/validation/confidence.py` | `ConfidenceEstimator.posterior_mean`, `lower_credible_bound`, `upper_credible_bound`, `update_posterior` | Discounted $\mathrm{Beta}(\alpha,\beta)$ quantiles ($\gamma=0.985, \delta=0.10$) & exact empirical ablation |
| **6. Sequential Evidence Gate** | `src/validation/sprt_validator.py` | `EvidenceValidationEngine.step_evidence` | Discounted log-odds promotion ($n_{\min}=3, A \approx 2.045, \tau_{\mathrm{LCB}}=0.48$) & quarantine ($B_{\mathrm{low}} \approx -1.780$) |
| **7. Applicability Retriever** | `src/retrieval/retriever.py` | `StrategyRetriever.retrieve`, `retrieve_step_guidance` | Conjunctive $\phi_h(\psi_t)=1$ & $\underline{p}_h^{\mathrm{cred}} \ge 0.45$ gating + marginal step composition |
| **8. Experimentation Engine** | `src/agents/experimentation.py` | `ExperimentationEngine.select_strategy_for_trial` | `VALIDATED`-first exploitation; Bayes-UCB `CANDIDATE` testing only when `learn=True` |
| **9. Residual CUSUM Detector** | `src/drift_detection/cusum_detector.py` | `DistributionShiftDetector.update` | Per-domain out-of-sample residual monitoring ($\nu=0.12, h=1.45, n_{\min,\mathrm{obs}}=3, \mathrm{cooldown}=12$) |
| **10. Adaptive Memory Manager** | `src/memory/memory_manager.py` | `AdaptiveMemoryManager.admit_or_merge_candidate`, `apply_domain_shift_discount` | Safe LGG re-verification ($\gamma_{\mathrm{LGG}}=0.80$) and selective post-shift quarantine |

Orchestrated end-to-end by `EVSAAgent.act_on_episode` in `src/agents/evsa_agent.py`.

---

## 2. Verified Algorithmic Invariants & Unit Tests

1. **Discounted Beta Posterior vs. Empirical Ablation (`src/validation/confidence.py`)**:
   - With `enabled=True`, $\underline{p}_h^{\mathrm{cred}}(0.10) = I_{0.10}^{-1}(\alpha_h, \beta_h)$ regularizes 1-trial flukes ($\alpha=2, \beta=1 \Rightarrow 0.316 < 0.48$).
   - With `enabled=False` (`w/o Confidence Estimation`), `ConfidenceEstimator` returns exact discounted empirical frequency $(\alpha_h - \alpha_0)/(\alpha_h - \alpha_0 + \beta_h - \beta_0)$ (`0.50` at zero trials), verified by `test_confidence_estimator_bayesian_vs_empirical_ablation`.
2. **Sequential Promotion & Safe LGG Re-Verification (`src/validation/sprt_validator.py`, `src/memory/memory_manager.py`)**:
   - Under $(p_0=0.35, p_1=0.80, \alpha_{\mathrm{FA}}=0.11, \beta_{\mathrm{FR}}=0.15)$, 3 consecutive successes reach $\Lambda_h^{(3)} \approx 2.443 \ge 2.045$ (`test_discounted_sprt_promotion_and_quarantine_gates`).
   - Whenever LGG precondition intersection strictly broadens an already-`VALIDATED` strategy ($|\phi_h^{\mathrm{LGG}}| < |\phi_h|, |\phi_h^{\mathrm{LGG}}| \ge 3$), excess pseudo-counts are discounted by $0.80$, $\Lambda_h$ is clamped to $\le A - 0.15$, and status reverts to `CANDIDATE` until confirmed (`test_safe_lgg_generalization_and_revalidation`).
3. **`VALIDATED`-Only Frozen Evaluation (`src/retrieval/retriever.py`, `src/agents/experimentation.py`)**:
   - Frozen evaluation (`learn=False`) strictly blocks `CANDIDATE` strategies from execution and enforces $\underline{p}_h^{\mathrm{cred}} \ge 0.45$, verified by `test_retriever_tau_rel_gate_and_frozen_eval_validated_only`.
4. **Update Ordering & Selective Shift Quarantine (`src/agents/evsa_agent.py`, `src/drift_detection/cusum_detector.py`)**:
   - Pre-rollout predictive mean $\hat{p}_{\mathrm{pre}}$ is captured prior to execution; posterior and sequential updates execute first, followed by residual CUSUM monitoring $e_t = \hat{p}_{\mathrm{pre}} - \tilde{y}_t$. On alarm, strategies in that domain with `consecutive_failures >= 1` or `LCB < 0.58` are quarantined while unfalsified rules are discounted ($\gamma_{\mathrm{shift}}=0.35$) to `CANDIDATE`, leaving unshifted domains untouched (`test_cusum_cooldown_and_selective_shift_quarantine`).
