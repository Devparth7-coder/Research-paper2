# Phase 4: Experimental Suite (`Experiments 1–5`) & Component Ablations (`v2.0.1`)

**Project**: *Evidence-Validated Strategy Adaptation for Memory-Augmented Agents Under Distribution Shift*  
**Author**: Dev Parth (Department of Computer Science and Engineering, MMMUT, Gorakhpur, Uttar Pradesh, India)  
**Release Version**: `v2.0.1` | **Date**: 7 November 2026  

---

## 1. Multi-Seed Experimental Protocol (`experiments/run_complete_suite.py`)

All core experiments execute deterministically across $N = 20$ primary seeds (`1001..1020`, plus $N = 30$ stability seeds `1001..1030` on Experiment 1), routing every episode through `run_audited_episode` (`src/evaluation/audit.py`) across $>360,000$ simulated task episodes.

---

## 2. Verified Empirical Outcomes (`N = 20` Seeds)

### Experiment 1: Held-Out Unseen Task Generalization (`240` Train $\to$ `240` Frozen Eval, `VALIDATED`-Only)
- **Overall SR**: EVSA achieves **`54.50% ± 5.04%`** (`95% CI [52.14%, 56.86%]`; **`55.17% ± 5.62%`** at $N=30$) vs. `Static` `26.48%`, `Conventional Memory` `29.62%`, `Reflection` `25.65%`, and `Experience Replay` `27.58%` ($d_z \in [4.31, 6.35], p_{\mathrm{Holm}} \le 8.15 \times 10^{-14}$).
- **Sub-Splits**: `66.19% ± 7.11%` on unseen surface instances of seen archetypes ($c \in \{0..6\}$) and **`31.13% ± 7.67%`** on zero-shot compositional archetype $c=7$ vs. `12.12%–17.94%` across baselines ($p_{\mathrm{Holm}} \le 1.07 \times 10^{-6}$), bounded below seen archetypes by the non-additive Step-3 XOR parity interaction.

### Experiment 2: Continual Learning Without Resets (`320` Episodes, 8 Blocks)
- **Avoidance of Early Negative Transfer**: At Block 1 (`t = 40`), memory baselines dip below zero-shot (`24.01%–26.51%` vs. `29.27%`), whereas EVSA rises immediately to **`32.86% ± 3.47%`** ($p_{\mathrm{Holm}} \le 9.21 \times 10^{-4}$).
- **Asymptotic Consolidation**: By Block 8 (`t = 320`), EVSA reaches **`65.21% ± 8.52%`** (`+35.94 pp` gain, $p_{\mathrm{Holm}} \le 3.85 \times 10^{-11}$) with just **`21.50`** active validated strategies (`0.95` false CUSUM alarms across 320 stationary episodes).

### Experiment 3: Multi-Regime Distribution Shift (`420` Episodes)
- **Post-Shift Recovery**: Following the `t = 160` 3-domain concept shift (`81.75%` pre-shift $\to$ `36.00%` shock), residual CUSUM fires within `38.40` mean global episodes and lifts EVSA to **`56.80% ± 8.76%`** recovered SR (**`+20.80 ± 12.49 pp`** Adaptation Gain vs. `-1.90` to `+4.70 pp` baselines, $p_{\mathrm{Holm}} \le 4.08 \times 10^{-4}$), and **`34.88%`** in Phase III compound shift (`9.00%–17.00%` baselines).
- **Cross-Domain Retention vs. Shifted-Domain Discounting**: On the unshifted control domain (`resource_planning`), EVSA maintains **`82.21% ± 11.21%`** (`0.25` false alarms, no measurable cross-domain degradation vs. `81.75%` pre-shift), whereas within alarmed shifted domains unfalsified `VALIDATED` strategies undergo conservative discounting ($\gamma_{\mathrm{shift}}=0.35$) back to `CANDIDATE`. Normalized Recovery Ratio (`70.22%`) is **non-significant** vs. `Conventional Memory` (`61.03%`, $p_{\mathrm{Holm}}=0.0988$) and `Experience Replay` (`58.55%`, $p_{\mathrm{Holm}}=0.0988$) due to denominator scaling.

### Experiment 4: Symmetric Feedback Corruption ($\eta \in \{0.10, 0.25, 0.35\}$)
- **Sequential Calibration**: Rollout FAR is restricted to **`1.34%`, `1.76%`, and `2.53%`** and Candidate FPR to **`8.08%`, `8.64%`, and `11.32%`** (closely tracking nominal $\alpha_{\mathrm{FA}} = 11\%$).
- **Contamination & Recovery**: Active noisy-phase FDP (`35.46%`, `46.36%`, `68.29%`) is lower than all memory baselines across all 9 pairs under per-family Holm ($p_{\mathrm{Holm}} \le 0.0063$; at $\eta=0.35$, global-74 $p_{\mathrm{Glob74}} = 0.0510$ borderline vs. `ConvMem` due to `65.44%` FRR shrinking clean promoted rules to `2.15`), before clean feedback slashes FDP to **`21.79%–30.60%`** and restores **`84.10%`, `77.30%`, and `74.90%`** recovery SR ($p_{\mathrm{Holm}} \le 1.95 \times 10^{-9}$).

### Experiment 5: Leave-One-Out Component Ablations (`360` Episodes, $\eta = 0.15$ + Shift)
- **`w/o Applicability Conditions`**: Catastrophic `-24.75 pp` stream SR collapse (`53.81% -> 29.06%`, $p_{\mathrm{Holm}} = 4.35 \times 10^{-14}$, `90.0%` pre-shift FDP).
- **`w/o Shift Detection`**: `-8.92 pp` post-shift recovered SR drop (`58.58% -> 49.67%`, $p_{\mathrm{Holm}} = 0.0037, p_{\mathrm{Glob74}} = 0.0175$) and `-2.25 pp` overall stream SR drop ($p_{\mathrm{Holm}} = 0.0464$ per-family; $p_{\mathrm{Glob74}} = 0.1508$ global-74).
- **Short-Horizon Purity vs. Coverage Trade-Offs**: `w/o Evidence Validation` (`55.51%` stream SR, $p_{\mathrm{Holm}} = 0.684$ ns) inflates active contamination $>2\times$ (`70.56%` vs. `33.18%`) and rule count $4.8\times$ (`42.55` vs. `8.95`); `w/o Memory Invalidation` spikes post-shift FDP to `63.29%`; `w/o Confidence Estimation` reduces noisy pre-shift SR by `-1.67 pp` (`6.25` vs. `8.95` rules).
