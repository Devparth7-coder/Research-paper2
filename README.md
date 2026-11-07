# Evidence-Validated Strategy Adaptation (EVSA) Under Distribution Shift

**Subtitle**: *A Statistical Framework for Reliable Experience Validation, Applicability-Gated Retrieval, and Selective Memory Invalidation*  
**Author**: Dev Parth  
**Affiliation**: Department of Computer Science and Engineering, Madan Mohan Malaviya University of Technology (MMMUT), Gorakhpur, Uttar Pradesh, India  
**Role**: Student / Independent Researcher  
**Release Version**: `v2.0.0` | **Completion Date**: 7 November 2026 | **License**: MIT  

---

## Overview

Memory-augmented autonomous agents accumulate experiential trajectories, verbal reflections, or induced workflows across tasks, yet routinely suffer **negative transfer**, **stale-strategy persistence**, and **feedback-corruption amplification** when task distributions shift or environment evaluators emit noisy rewards.

**Evidence-Validated Strategy Adaptation (`EVSA`)** reformulates experiential memory as an explicit sequential statistical inference problem over structured procedural hypotheses (`CANDIDATE` $\to$ `VALIDATED` $\to$ `QUARANTINED`). Evaluated on **`ShiftBench-Sim v1.0`**—a deterministic, strictly zero-oracle-leakage symbolic-procedural multi-domain benchmark ($>360,000$ task episodes across $N=20$ primary seeds and $N=30$ stability seeds)—EVSA integrates:
1. **Sequential Evidence Gating**: A discounted log-odds sequential heuristic motivated by Wald's SPRT (`min_trials = 3`) paired with discounted Bayesian $\mathrm{Beta}(\alpha, \beta)$ lower credible bounds.
2. **Applicability-Gated Retrieval & Re-Verified Safe LGG**: Multiplicative precondition and reliability gating $\phi_h(\psi_t) \cdot \mathbb{I}[\underline{p}_h^{\mathrm{cred}} \ge 0.45]$ with Plotkin-style Least General Generalization (`Safe LGG`, requiring confirming verification whenever a `VALIDATED` strategy broadens its preconditions), while restricting frozen evaluation strictly to `VALIDATED` strategies.
3. **Domain-Scoped Residual CUSUM Drift Detection**: Online monitoring of out-of-sample prediction residuals $e_t = \hat{p}_{h^*}^{(t-1)} - \tilde{y}_t$ (`nu = 0.12`, `h = 1.45`, `cooldown = 12`) to selectively quarantine empirically failed or low-LCB strategies in shifted domains without collateral forgetting in stationary domains.

---

## Key Verified Empirical Findings (`v2.0.0`, $N = 20$ Paired Seeds, Holm-Bonferroni Corrected)

| Experiment | Primary Metric | EVSA (Proposed) | Best Baseline | Effect Size & Adjusted $p$-Value |
| :--- | :--- | :--- | :--- | :--- |
| **Exp 1: Unseen Benchmark** | Held-Out Success Rate (`240` tasks, `VALIDATED`-only) | **`54.50% ± 5.04%`** (`55.17%` at $N=30$) | `29.62% ± 2.77%` (`ConvMem`) | $d_z = 4.47, p_{\mathrm{Holm}} = 8.15 \times 10^{-14}$ |
| **Exp 1: Unseen Comp. ($c=7$)** | Zero-Shot Held-Out Archetype SR | **`31.13% ± 7.67%`** | `17.94% ± 4.61%` (`Static`) | $d_z = 1.57, p_{\mathrm{Holm}} = 1.07 \times 10^{-6}$ (`+19.00 pp` vs `Replay`) |
| **Fairness Parity Control** | `EVSA` vs `Replay` under Identical Stochastic Exec | **`49.62% ± 4.98%`** | `27.58% ± 1.74%` (`Replay`) | **`+22.04` pp** ($\sim 82\%$ of gain; $d_z = 4.09, p_{\mathrm{Holm}} = 1.57 \times 10^{-13}$) |
| **Exp 2: Continual Learning** | Terminal Block-8 SR (`320` eps) | **`65.21% ± 8.52%`** (`21.50` rules) | `35.68% ± 5.47%` (`ConvMem`) | $d_z = 3.00, p_{\mathrm{Holm}} = 3.85 \times 10^{-11}$ |
| **Exp 3: Distribution Shift** | Post-Shift Adaptation Gain | **`+20.80 ± 12.49` pp** (`56.80%` Rec.) | `+4.70 ± 12.93` pp (`ConvMem`) | $d_z = 0.96, p_{\mathrm{Holm}} = 4.08 \times 10^{-4}$ (*Rec. Ratio vs ConvMem ns: $p=0.099$*) |
| **Exp 3: Selective Retention** | Unshifted Control Domain SR | **`82.21% ± 11.21%`** | `57.49% ± 8.91%` (`ConvMem`) | $d_z = 1.66, p_{\mathrm{Holm}} = 4.08 \times 10^{-6}$ |
| **Exp 4: Noisy Feedback ($\eta=0.35$)** | Rollout FAR / Cand. FPR / Clean Rec. | **`2.53%` / `11.32%` / `74.90%`** | `37.00%` FAR / `49.40%` Rec. | Clean Rec $d_z = 2.55, p_{\mathrm{Holm}} = 1.26 \times 10^{-9}$; FDP `-15.2 pp` ($p=0.0063$; FRR `65.44%`) |
| **Exp 5: Component Ablations** | `w/o Applicability` / `w/o Shift Detection` | `-24.75` pp stream / `-8.92` pp post-shift | `EVSA Full` (`53.81%` / `58.58%`) | $p_{\mathrm{Holm}} \le 3.70 \times 10^{-3}$ (`w/o Val` $2.1\times$ FDP `70.6%` vs `33.2%`, SR diff ns $p=0.684$) |
| **Exp 6: Efficiency & Projections** | CPU Latency / Memory / Est. Tokens | `0.427 ms/task` / `13.14 KB` / `46.8k tok` | `0.077–0.251 ms` / `10.6–106.1 KB` | `0.117` SR per 10k est. tokens (`1.83x–4.68x` baselines; `590.5` rollouts) |

---

## Repository Structure

```text
.
├── README.md                          # Landing page: overview, verified v2.0.0 metrics & quickstart
├── requirements.txt                   # Python scientific, testing & document dependencies
├── LICENSE                            # MIT License (Copyright 2026 Dev Parth)
├── CITATION.cff                       # Standard Citation File Format metadata (v2.0.0)
├── REPRODUCIBILITY.md                 # Hardware specs, seed policy, and step-by-step commands
├── TIMELINE.md                        # Conceptual phase structure vs. verified repo timestamps
├── FINAL_AUDIT_REPORT.md              # Forensic algorithmic, mathematical & statistical audit report
├── UNRESOLVED_LIMITATIONS.md          # Candid register of theoretical & empirical limitations
├── config/
│   └── default_config.json            # Master benchmark & agent hyperparameters (v2.0.0)
├── src/
│   ├── benchmark/
│   │   ├── environment.py             # ShiftBench-Sim v1.0 (strict zero-oracle-leakage isolation)
│   │   └── llm_backbone.py            # Non-oracle CalibratedSymbolicLLMBackbone
│   ├── validation/
│   │   ├── sprt_validator.py          # Discounted SPRT sequential log-odds heuristic
│   │   └── confidence.py              # Discounted Beta credible bounds & raw empirical ablation
│   ├── drift_detection/
│   │   └── cusum_detector.py          # Per-domain out-of-sample residual tabular CUSUM monitor
│   ├── retrieval/
│   │   └── retriever.py               # Applicability & LCB-gated retrieval + Bayes-UCB exploration
│   ├── memory/
│   │   ├── repository.py              # Three-tier strategy store & tombstone signatures
│   │   └── memory_manager.py          # Safe LGG re-verification & selective drift quarantine
│   ├── agents/
│   │   ├── baselines.py               # Static, Conventional Memory, Reflection & Experience Replay
│   │   └── evsa_agent.py              # Closed-loop EVSA agent & fairness parity modes
│   └── evaluation/
│       └── metrics_and_stats.py       # Paired t/Wilcoxon, non-central-t power & Holm-Bonferroni
├── experiments/
│   ├── configs/                       # Per-experiment JSON protocol configurations
│   ├── raw/                           # Seed-level raw CSV logs (N=20 & N=30 seeds, CUSUM traces)
│   ├── processed/                     # Aggregated summary CSVs (Mean, SD, Median, IQR, 95% CI)
│   ├── statistical_analysis/          # 74 paired t/Wilcoxon tests (per-family & global-74 Holm)
│   ├── run_complete_suite.py          # Master deterministic execution & table export runner
│   └── generate_all_figures.py        # Reproducible generator for all 10 PNG/PDF figures
├── tests/
│   └── test_evsa_suite.py             # 11 unit & zero-oracle-leakage AST/source integrity tests
├── figures/                           # 10 publication figures (.png @ 300 DPI + vector .pdf)
├── tables/                            # 9 CSV summary tables + summary_tables.md
└── paper/
    ├── research_paper.tex             # Complete 21-section LaTeX manuscript (v2.0.0)
    ├── research_paper.pdf             # Compiled 21-page publication PDF
    ├── research_paper.docx            # Synchronized Microsoft Word manuscript
    ├── build_docx.py                  # Programmatic DOCX generator matching LaTeX manuscript
    └── references.bib                 # 30 verified BibTeX scholarly references
```

---

## Quick Start

```bash
pip install -r requirements.txt
pytest -v tests/test_evsa_suite.py
python3 experiments/run_complete_suite.py
python3 experiments/generate_all_figures.py
```
See [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md) for manuscript compilation (`PDF` + `DOCX`) and full verification details.
