# Evidence-Validated Strategy Adaptation (EVSA) Under Distribution Shift

**Subtitle**: *A Statistical Framework for Reliable Experience Validation, Applicability-Gated Retrieval, and Selective Memory Invalidation*  
**Author**: Dev Parth  
**Role**: Independent Researcher  
**Research Area**: Independent Research in Artificial Intelligence  
**Release Version**: `v2.0.2` | **Release Date**: 9 November 2026 | **License**: MIT  

**Canonical Repository Links**:
- **Repository Root**: [https://github.com/Devparth7-coder/EVSA-Research-Paper](https://github.com/Devparth7-coder/EVSA-Research-Paper)
- **Seven Research Phase Dossiers (`docs/`)**: [docs](https://github.com/Devparth7-coder/EVSA-Research-Paper/tree/main/docs)
- **LaTeX & DOCX Manuscript (`paper/`)**: [paper](https://github.com/Devparth7-coder/EVSA-Research-Paper/tree/main/paper)
- **Capability-Isolated Python Source (`src/`)**: [src](https://github.com/Devparth7-coder/EVSA-Research-Paper/tree/main/src)
- **74-Test Statistical & Direct Global-74 Holm Ledger (`experiments/statistical_analysis/`)**: [experiments/statistical_analysis](https://github.com/Devparth7-coder/EVSA-Research-Paper/tree/main/experiments/statistical_analysis)
- **Raw Seed-Level Experimental Logs (`experiments/raw/`)**: [experiments/raw](https://github.com/Devparth7-coder/EVSA-Research-Paper/tree/main/experiments/raw)
- **16-Test Cross-Platform Scientific Integrity Suite (`tests/`)**: [tests](https://github.com/Devparth7-coder/EVSA-Research-Paper/tree/main/tests)

---

## Overview

Memory-augmented autonomous agents accumulate experiential trajectories, verbal reflections, or induced workflows across tasks, yet routinely suffer **negative transfer**, **stale-strategy persistence**, and **feedback-corruption amplification** when task distributions shift or environment evaluators emit noisy rewards.

**Evidence-Validated Strategy Adaptation (`EVSA`)** reformulates experiential memory as a sequential statistical inference problem over structured procedural hypotheses (`CANDIDATE` $\to$ `VALIDATED` $\to$ `QUARANTINED`). Evaluated in **`ShiftBench-Sim v1.0`**—a controlled, low-dimensional synthetic symbolic-procedural multi-domain simulation (`4`-step discrete operator workflows, observable `5`-bit binary state descriptors, `20`-dimensional synthetic embeddings, and a heuristic `CalibratedSymbolicLLMBackbone` isolated behind a slot-restricted `AgentEnvironmentInterface`; $>360,000$ simulated episodes across $N=20$ primary seeds and $N=30$ stability seeds; note that synthetic symbolic simulation does not yet establish live frontier-LLM deployment gains)—EVSA integrates:
1. **Sequential Evidence Gating**: A discounted log-odds sequential heuristic motivated by Wald's SPRT (`min_trials = 3`) paired with discounted Bayesian $\mathrm{Beta}(\alpha, \beta)$ lower credible bounds (`ConfidenceEstimator` + `EvidenceValidationEngine`).
2. **Applicability-Gated Retrieval & Re-Verified Safe LGG**: Multiplicative precondition and reliability gating $\phi_h(\psi_t) \cdot \mathbb{I}[\underline{p}_h^{\mathrm{cred}} \ge 0.45]$ with Plotkin-style Least General Generalization (`Safe LGG`, requiring confirming verification whenever a `VALIDATED` strategy broadens its preconditions), while restricting frozen evaluation (`learn=False`) strictly to `VALIDATED` strategies.
3. **Domain-Scoped Residual CUSUM Drift Detection**: Online monitoring of out-of-sample prediction residuals $e_t = \hat{p}_{h^*}^{(t-1)} - \tilde{y}_t$ (`slack_nu = 0.12`, `drift_threshold = 1.45`, `cooldown_episodes = 12`) to selectively quarantine empirically failed or low-LCB strategies in shifted domains without cross-domain degradation on stationary control domains (`82.21%` vs. `81.75%` pre-shift).

---

## Key Verified Empirical Findings (`v2.0.2`, $N = 20$ Paired Seeds, Direct Primary & Sensitivity Holm Ledgers)

| Experiment | Primary Metric | EVSA (Proposed) | Best Baseline | Effect Size & Primary Holm $p$-Value |
| :--- | :--- | :--- | :--- | :--- |
| **Exp 1: Unseen Benchmark** | Held-Out Success Rate (`240` tasks, `VALIDATED`-only) | **`54.50% ± 5.04%`** (`55.17%` at $N=30$) | `29.62% ± 2.77%` (`ConvMem`) | $d_z = 4.47, p_{\mathrm{Holm}} = 8.15 \times 10^{-14}$ |
| **Exp 1: Unseen Comp. ($c=7$)** | Zero-Shot Held-Out Archetype SR | **`31.13% ± 7.67%`** | `17.94% ± 4.61%` (`Static`) | $d_z = 1.57, p_{\mathrm{Holm}} = 1.07 \times 10^{-6}$ (`+19.00 pp` vs `Replay`) |
| **Fairness Parity Control** | `EVSA` vs `Replay` under Identical Stochastic Exec | **`49.62% ± 4.98%`** | `27.58% ± 1.74%` (`Replay`) | **`+22.04` pp** ($\sim 82\%$ of gain; $d_z = 4.09, p_{\mathrm{Holm}} = 1.57 \times 10^{-13}$) |
| **Exp 2: Continual Learning** | Terminal Block-8 SR (`320` eps) | **`65.21% ± 8.52%`** (`21.50` rules) | `35.68% ± 5.47%` (`ConvMem`) | $d_z = 3.00, p_{\mathrm{Holm}} = 3.85 \times 10^{-11}$ |
| **Exp 3: Distribution Shift** | Post-Shift Adaptation Gain | **`+20.80 ± 12.49` pp** (`56.80%` Rec.) | `+4.70 ± 12.93` pp (`ConvMem`) | $d_z = 0.96, p_{\mathrm{Holm}} = 4.08 \times 10^{-4}$ (*Rec. Ratio vs ConvMem ns: $p=0.099$*) |
| **Exp 3: Stationary Control** | Unshifted Control Domain SR | **`82.21% ± 11.21%`** | `57.49% ± 8.91%` (`ConvMem`) | $d_z = 1.66$, primary $p_{\mathrm{Holm}} = 4.77 \times 10^{-5}$ ($p_{t,\mathrm{Holm}} = 8.16 \times 10^{-6}$) |
| **Exp 4: Noisy Feedback ($\eta=0.35$)** | Rollout FAR / Cand. FPR / Clean Rec. | **`2.53%` / `11.32%` / `74.90%`** | `37.00%` FAR / `49.40%` Rec. | Clean Rec $d_z = 2.55, p_{\mathrm{Holm}} = 1.68 \times 10^{-9}$; FDP `-15.2 pp` ($p_{\mathrm{Holm}}=0.0063$; `65.44%` FRR) |
| **Exp 5: Component Ablations** | `w/o Applicability` / `w/o Shift Detection` | `-24.75` pp stream / `-8.92` pp post-shift | `EVSA Full` (`53.81%` / `58.58%`) | Primary $p_{\mathrm{Holm}} \le 3.70 \times 10^{-3}$ (`w/o Val` $2.1\times$ FDP `70.6%` vs `33.2%`, SR diff ns $p=0.684$) |
| **Exp 6: Efficiency & Projections** | Memory / Rollouts / Est. Tokens | `13.14 KB` / `590.5` / `46.8k tok` | `10.6–106.1 KB` / `645–651` | `0.117` SR per 10k est. tokens (`1.83x–4.68x` baselines; `< 1.0 ms/task` local CPU) |

Across all **74 paired hypothesis tests** (`19` pre-defined metric families; `66` paired-$t$ primary and `8` exact Wilcoxon primary selected via Shapiro-Wilk pre-testing before adjustment), **62 / 74** are significant at $\alpha=0.05$ under direct per-family primary Holm-Bonferroni (`apply_holm_bonferroni`) and **60 / 74** remain significant under direct single-pass Global-74 primary Holm-Bonferroni (`apply_global_holm_bonferroni`), alongside explicitly labelled pure-parametric (`60 / 74`) and pure-Wilcoxon (`57 / 74`) sensitivity families.

---

## Authoritative Seven-Phase Research Documentation (`docs/`)

1. [`docs/PHASE_1_PROBLEM_AND_THEORY.md`](docs/PHASE_1_PROBLEM_AND_THEORY.md) — Problem formulation, 30-source verified literature synthesis (`AdaMEM`, `Dynamic Cheatsheet`, `ReasoningBank`), and `RQ1`–`RQ5`.
2. [`docs/PHASE_2_ARCHITECTURE_AND_ALGORITHMS.md`](docs/PHASE_2_ARCHITECTURE_AND_ALGORITHMS.md) — Ten-module architecture mapping to `src/`, discounted SPRT heuristic, Beta credible bounds, and Safe LGG.
3. [`docs/PHASE_3_BENCHMARK_AND_BASELINES.md`](docs/PHASE_3_BENCHMARK_AND_BASELINES.md) — Synthetic symbolic scope of `ShiftBench-Sim v1.0`, `AgentEnvironmentInterface` hard oracle isolation, and 4 baselines.
4. [`docs/PHASE_4_EXPERIMENTS_AND_ABLATIONS.md`](docs/PHASE_4_EXPERIMENTS_AND_ABLATIONS.md) — Multi-seed experimental protocols and empirical outcomes across Experiments 1–5.
5. [`docs/PHASE_5_STATISTICAL_ANALYSIS_AND_POWER.md`](docs/PHASE_5_STATISTICAL_ANALYSIS_AND_POWER.md) — Pre-selected primary testing, exact non-central-$t$ power, and direct single-pass per-family vs. Global-74 Holm ledgers.
6. [`docs/PHASE_6_EFFICIENCY_AND_FAIRNESS_AUDIT.md`](docs/PHASE_6_EFFICIENCY_AND_FAIRNESS_AUDIT.md) — Controlled $2 \times 2$ execution-channel parity attribution and local CPU vs. projected token economics.
7. [`docs/PHASE_7_MANUSCRIPT_AND_RELEASE_AUDIT.md`](docs/PHASE_7_MANUSCRIPT_AND_RELEASE_AUDIT.md) — 17-section manuscript structure, cross-platform Linux + Windows CI pipeline, and `v2.0.2` release verification.

---

## Repository Structure

```text
.
├── .github/workflows/ci.yml           # Cross-platform GitHub Actions CI (Ubuntu Py 3.11–3.13 + Windows Py 3.13)
├── .gitignore                         # Tracked ignore rules for bytecode, caches & LaTeX intermediates
├── README.md                          # Landing page: overview, verified v2.0.2 metrics & quickstart
├── RELEASE_NOTES_v2.0.2.md            # Release changelog, attribution & post-push verification guide
├── requirements.txt                   # Exact pinned scientific, testing & document dependencies
├── LICENSE                            # MIT License (Copyright 2026 Dev Parth)
├── CITATION.cff                       # Standard Citation File Format metadata (v2.0.2, 2026-11-09)
├── REPRODUCIBILITY.md                 # Cross-platform specs, seed policy & step-by-step commands
├── TIMELINE.md                        # Conceptual 7-phase organization vs. verified git timestamps
├── FINAL_AUDIT_REPORT.md              # Forensic interface-isolation, statistical & release audit
├── UNRESOLVED_LIMITATIONS.md          # Candid register of theoretical & empirical limitations
├── docs/                              # Consolidated authoritative Phase 1–7 dossiers (7 files)
├── config/default_config.json         # Master benchmark & agent hyperparameters
├── src/
│   ├── benchmark/
│   │   ├── types.py                   # Observable TaskInstance, RolloutResult & AgentEpisodeOutcome
│   │   ├── agent_interface.py         # Slot-restricted AgentEnvironmentInterface (hard oracle boundary)
│   │   ├── environment.py             # ShiftBench-Sim v1.0 evaluator environment
│   │   └── llm_backbone.py            # Non-oracle CalibratedSymbolicLLMBackbone
│   ├── validation/
│   │   ├── sprt_validator.py          # EvidenceValidationEngine (discounted SPRT heuristic)
│   │   └── confidence.py              # ConfidenceEstimator (Beta credible bounds & empirical ablation)
│   ├── drift_detection/
│   │   └── cusum_detector.py          # DistributionShiftDetector (per-domain residual tabular CUSUM)
│   ├── retrieval/
│   │   └── retriever.py               # StrategyRetriever (precondition & LCB-gated retrieval)
│   ├── memory/
│   │   ├── repository.py              # StrategyRepository (3-tier store & tombstones)
│   │   ├── hypothesis_generator.py    # HypothesisGenerator (step-predicate salience)
│   │   └── memory_manager.py          # AdaptiveMemoryManager (Safe LGG & selective shift discount)
│   ├── agents/
│   │   ├── experimentation.py         # ExperimentationEngine (VALIDATED-only frozen eval & Bayes-UCB)
│   │   ├── baselines.py               # Static, ConventionalMemory, Reflection & ExperienceReplay agents
│   │   └── evsa_agent.py              # EVSAAgent closed-loop controller & parity modes
│   └── evaluation/
│       ├── audit.py                   # Evaluator-only run_audited_episode & audit_evsa_repository
│       └── metrics_and_stats.py       # Direct single-pass per-family/Global-74 Holm + sensitivity families
├── experiments/
│   ├── configs/                       # Per-experiment JSON protocol configurations
│   ├── raw/                           # Seed-level raw CSV logs (N=20 & N=30 seeds, CUSUM traces)
│   ├── processed/                     # Aggregated summary CSVs (Mean, SD, Median, IQR, 95% CI)
│   ├── statistical_analysis/          # 74 paired tests (direct primary + sensitivity Holm) + digest JSON
│   ├── run_complete_suite.py          # Cross-platform deterministic execution & statistical runner
│   ├── generate_all_figures.py        # Reproducible generator for all 10 PNG/PDF figures
│   └── verify_github_release.py       # Post-push GitHub tag, Actions CI & Release API verifier
├── tests/test_evsa_suite.py           # 16 unit, hard-boundary, attribution, Windows & ledger tests
├── figures/                           # 10 publication figures (.png @ 300 DPI + vector .pdf)
├── tables/                            # 9 CSV summary tables + summary_tables.md
└── paper/
    ├── research_paper.tex             # Complete 17-section LaTeX manuscript (v2.0.2)
    ├── research_paper.pdf             # Compiled publication PDF
    ├── research_paper.docx            # Synchronized Microsoft Word manuscript
    ├── build_docx.py                  # Programmatic UTF-8 DOCX generator matching LaTeX manuscript
    └── references.bib                 # 30 verified BibTeX scholarly references
```

---

## Quick Start (Linux, Windows, or macOS)

```bash
pip install -r requirements.txt
pytest -v tests/test_evsa_suite.py
python experiments/run_complete_suite.py
python experiments/generate_all_figures.py
```
See [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md) and [`RELEASE_NOTES_v2.0.2.md`](RELEASE_NOTES_v2.0.2.md) for manuscript compilation (`PDF` + `DOCX`), clean-clone verification, and post-push GitHub tag/CI verification (`python experiments/verify_github_release.py --tag v2.0.2`).
