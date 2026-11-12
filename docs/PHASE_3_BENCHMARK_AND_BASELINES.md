# Phase 3: Synthetic Benchmark (`ShiftBench-Sim v1.0`), Hard Oracle Isolation & Baselines (`v2.0.2`)

**Project**: *Evidence-Validated Strategy Adaptation for Memory-Augmented Agents Under Distribution Shift*  
**Author**: Dev Parth — Independent Researcher (*Independent Research in Artificial Intelligence*)  
**Release Version**: `v2.0.2` | **Date**: 9 November 2026  

---

## 1. Synthetic Symbolic Benchmark Scope (`ShiftBench-Sim v1.0`)

`ShiftBench-Sim v1.0` (`src/benchmark/environment.py`) is an intentionally controlled, low-dimensional **synthetic symbolic-procedural simulation**—explicitly distinct from live neural LLM benchmarks (`SWE-bench`, `WebArena`)—designed to audit sequential validation, precondition collisions, concept changepoints, and symmetric feedback noise without pretraining contamination:
- **Domains & Workflow Space**: 4 domains (`api_orchestration`, `system_diagnosis`, `data_pipeline`, `resource_planning`), each requiring a 4-step discrete operator tuple $\pi \in \{0..3\} \times \{0..4\} \times \{0..3\} \times \{0..2\}$ (`240` distinct plans/domain).
- **Observable Binary Preconditions**: 5-bit vector $\psi_t \in \{0,1\}^5$ comprising 3 governing bits $(\psi_0, \psi_1, \psi_2)$ ($8$ archetypes $c \in \{0..7\}$) and 2 nuisance bits $(\psi_3, \psi_4)$. Archetype $c=7$ ($\psi_0=\psi_1=\psi_2=1$) is held out for zero-shot compositional testing, with additive single-bit dependencies on Steps 0–2 and a non-linear XOR parity rule ($\psi_0 \oplus \psi_1$) on Step 3 (`test_backbone_does_not_know_latent_xor_parity`).
- **Surface-Topic Collisions**: 20-dimensional unit-normalized synthetic embeddings $x_t \in \mathbb{R}^{20}$ weight shared surface topics (`0.82`) above noisy structural features (`0.36`), creating realistic high-cosine collisions across conflicting archetypes.
- **Regime Shifts & Outcome Channel**:
  - `Regime 0`: Stationary baseline $\pi^*_{d,c,0}$.
  - `Regime 1`: Abrupt concept shift (+2 modulo cardinality on Steps 1–2) in 3 domains while keeping `resource_planning` invariant (`test_environment_determinism_and_regime_invariance`).
  - `Regime 2`: Compound shift across all 4 domains (Steps 0 and 2) plus covariate shift onto $c \in \{4,5,6,7\}$.
  - True completion probabilities $(0.92, 0.28, 0.09, 0.02)$ for $(4, 3, 2, \le 1)$ matching steps, observed through symmetric corruption channel $\eta_t \in [0, 0.5)$.

---

## 2. Capability-Restricted API Boundary for Accidental-Oracle Isolation (`v2.0.2`)

To guarantee that no benchmark agent can inspect evaluator ground truth through public APIs or object attributes (note: because agents and evaluator execute inside the same Python process, this is an engineering encapsulation boundary against accidental oracle leakage rather than an OS/process-level security sandbox against hostile code):
1. **Purely Observable Data Structures (`src/benchmark/types.py`)**: `TaskInstance`, `RolloutResult`, and `AgentEpisodeOutcome` contain zero oracle attributes (`test_task_rollout_and_agent_outcome_have_zero_oracle_fields`).
2. **Slot-Restricted Capability Handle (`src/benchmark/agent_interface.py`)**: `NonStationaryTaskEnvironment.make_agent_interface` wraps rollout execution inside `AgentEnvironmentInterface` (`__slots__ = ("__weakref__",)`, private `WeakKeyDictionary` dispatch, no `__dict__`), exposing **only** `evaluate_plan(task, executed_plan) -> RolloutResult`.
3. **Runtime `TypeError` Guard & Evaluator-Side Telemetry (`src/evaluation/audit.py`)**: All five agents implement `act_on_episode(task, env: AgentEnvironmentInterface, ...)` and raise `TypeError` if passed a raw `NonStationaryTaskEnvironment` (`test_hard_interface_boundary_separates_agent_from_oracle`). Ground-truth `EpisodeTelemetry` is assembled exclusively post-episode inside `run_audited_episode`, and zero agent/memory/backbone modules import oracle types (`test_zero_oracle_leakage_across_all_agent_and_backbone_modules`).

---

## 3. Budget-Matched Symbolic Baseline Adaptations (`src/agents/baselines.py`)

All five agents share the identical `CalibratedSymbolicLLMBackbone`, identical task streams per seed, and identical rollout budgets ($B=2$ adaptation, $B=1$ frozen evaluation):
- **Baseline A (`StaticAgent`)**: Memoryless ReAct-style adaptation sampling strictly from `CalibratedSymbolicLLMBackbone.propose_plan`.
- **Baseline B (`ConventionalMemoryAgent`)**: Episodic buffer ($400$ max) ranked by $0.55\cos(x_t,x_i) + 0.20\cdot 0.993^{\Delta t} + 0.25\,\tilde{r}_i$, conditioning `execute_guided_plan` on top-3 positive matches.
- **Baseline C (`ReflectionAgent`)**: Per-domain sliding reflection buffers ($10$/domain) tracking step-failure avoidance constraints and unvalidated positive traces.
- **Baseline D (`ExperienceReplayAgent`)**: Distilled successful workflow buffer ($200$ max, deduplicated at $\cos > 0.80$), supporting both default `stochastic_prompt` and controlled `direct_macro` execution parity modes (`test_budget_parity_and_contamination_audit`).
