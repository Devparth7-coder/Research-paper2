# Release Notes — EVSA `v2.0.0` (7 November 2026)

**Title**: *Evidence-Validated Strategy Adaptation for Memory-Augmented Agents Under Distribution Shift*  
**Author**: Dev Parth (Department of Computer Science and Engineering, MMMUT, Gorakhpur, Uttar Pradesh, India)  
**Tag**: `v2.0.0` | **License**: MIT  

---

## Summary of `v2.0.0`

`v2.0.0` is the clean-root, zero-oracle-verified research release of **Evidence-Validated Strategy Adaptation (`EVSA`)** and **`ShiftBench-Sim v1.0`**. It consolidates the repository at `/` (`README.md` landing page), resolves four deep code-audit findings across `src/`, re-executes all $>360,000$ multi-seed episodes from scratch, regenerates all 10 publication figures (`PNG` + `PDF`), and synchronizes the 21-section manuscript (`paper/research_paper.{tex,pdf,docx}`) and governance documentation.

### Key Scientific & Architectural Improvements in `v2.0.0`
1. **Strict Zero-Oracle Isolation**: Removed all ground-truth fields from `TaskInstance` and `RolloutResult` and replaced any latent XOR lookup in `CalibratedSymbolicLLMBackbone` with a strictly non-oracle linear inclusive-OR step prior (`0.67` marginal step competence), enforced by AST/source-level unit tests (`tests/test_evsa_suite.py`, `11/11` passing).
2. **True Unregularized Empirical Frequency Ablation**: Updated `w/o Confidence Estimation` (`src/validation/confidence.py`) to compute exact discounted empirical success rates $\hat{p}_h^{\mathrm{emp}}$ rather than a constant fallback.
3. **Aligned Residual CUSUM & Selective Evidence Quarantine**: Synchronized per-domain tabular CUSUM (`nu=0.12, h=1.45, warmup=3, cooldown=12`), enforced pre-CUSUM strategy parameter updates, and replaced creation-age shift pruning with selective quarantine of strategies exhibiting $f_h^{\mathrm{consec}} \ge 1$ or $\underline{p}_h^{\mathrm{cred}} < 0.58$ while discounting unfalsified peers (`0.35`) to `CANDIDATE`.
4. **Reliability-Gated Retrieval, `VALIDATED`-Only Frozen Evaluation & Safe LGG**: Enforced $\underline{p}_h^{\mathrm{cred}} \ge 0.45$ during retrieval, excluded `CANDIDATE` strategies when `learn=False`, and required one confirming trial whenever LGG broadens an already-`VALIDATED` strategy (`Safe LGG`), reducing high-noise ($\eta=0.35$) active contamination significantly below all memory baselines (`68.29%` vs. `83.12%–85.50%`, $p_{\mathrm{Holm}} = 0.0063$).

### Headline Verified Metrics (`N = 20` Primary Seeds; `N = 30` Stability Seeds)
- **Exp 1 (Held-Out Unseen Generalization)**: **`54.50% ± 5.04%`** (`55.17% ± 5.62%` at $N=30$) vs. `25.65%–29.62%` baselines ($p_{\mathrm{Holm}} \le 8.15 \times 10^{-14}$); Zero-Shot Compositional $c=7$: **`31.13% ± 7.67%`** vs. `12.12%–17.94%` ($p_{\mathrm{Holm}} \le 1.07 \times 10^{-6}$).
- **Execution-Channel Fairness Control**: **`+22.04 pp`** (`49.62%` vs. `27.58%`, $\sim 82\%$ of total gain, $p_{\mathrm{Holm}} = 1.57 \times 10^{-13}$) under identical stochastic prompt execution.
- **Exp 2 (Continual Learning, 320 eps)**: **`65.21% ± 8.52%`** (`+35.94 pp`, `21.50` active strategies, `13.14 KB`).
- **Exp 3 (Multi-Regime Shift)**: **`+20.80 ± 12.49 pp`** post-shock Adaptation Gain (`56.80%` recovered SR) with **`82.21% ± 11.21%`** retention on the unshifted control domain.
- **Exp 4 (Feedback Corruption $\eta \in \{0.10, 0.25, 0.35\}$)**: Rollout FAR **`1.34%–2.53%`**, Candidate FPR **`8.08%–11.32%`**, Clean Recovery SR **`74.90%–84.10%`** ($p_{\mathrm{Holm}} \le 1.95 \times 10^{-9}$).
- **Statistical Multiplicity**: `62 / 74` tests significant under per-family Holm-Bonferroni (`60 / 74` under simultaneous global-74 Holm-Bonferroni).

### Verification from Clean Clone
```bash
git clone --branch v2.0.0 <REPO_URL> evsa-research
cd evsa-research
pip install -r requirements.txt
pytest -v tests/test_evsa_suite.py
python3 experiments/run_complete_suite.py
python3 experiments/generate_all_figures.py
```
