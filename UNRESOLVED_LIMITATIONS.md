# Register of Genuine Unresolved Limitations & Open Challenges (`v2.0.1`)

**Project**: *Evidence-Validated Strategy Adaptation for Memory-Augmented Agents Under Distribution Shift*  
**Author**: Dev Parth (MMMUT Gorakhpur, India)  
**Release Version**: `v2.0.1` | **Date**: 7 November 2026  

To ensure complete scientific candor and guide future extensions, this document catalogs the unresolved theoretical, algorithmic, and empirical limitations of `EVSA v2.0.1`.

---

## 1. Synthetic Low-Dimensional Simulation (`ShiftBench-Sim v1.0`) vs. Live Frontier LLM Evaluation
- **Current Scope**: All experiments are conducted in `ShiftBench-Sim v1.0`—a controlled, low-dimensional synthetic symbolic-procedural simulation (`4`-step discrete operator tuples $\pi \in \{0,1,2,3\}^4$, `20`-dimensional synthetic query embeddings, and a heuristic `CalibratedSymbolicLLMBackbone`) rather than live autoregressive LLM API calls over open-ended text or source code. Token counts (`46,758` for EVSA) and API dollar figures (`$0.213`) in Experiment 6 are **analytical cost projections** under a serialized prompt-accounting model (`$2.50 / $10.00` per $10^6$ input/output tokens; `$0.00` commercial API spend), whereas local CPU runtime (`~0.4–0.7 ms/task`) measures Python/NumPy statistical bookkeeping only.
- **Unresolved Gap**: Live LLM backbones exhibit prompt-position sensitivity, syntactic paraphrasing drift, long-horizon partial observability, and free-form code synthesis error modes not captured by discrete 4-step workflow tuples. Validating EVSA with live open-weights and frontier LLMs on `SWE-bench`, `WebArena`, and `OSWorld` remains essential future work.

## 2. Observability of Structured Binary Preconditions ($\psi_t$)
- **Current Scope**: `ShiftBench-Sim v1.0` provides an observable 5-bit binary state descriptor $\psi_t \in \{0,1\}^5$ (3 governing bits + 2 nuisance bits), enabling exact conjunctive precondition evaluation and Plotkin-style Safe Least General Generalization (`Safe LGG`).
- **Unresolved Gap**: In unstructured natural-language tasks, preconditions must be induced via learned classifiers, programmatic assertions, or LLM-generated predicates whose extraction noise could degrade $\phi_h(\psi_t)$ gating. Moreover, even with Safe LGG re-verification, two consecutive corrupted $\tilde{y}=1$ flips under severe noise ($\eta=0.35$) still leave `1.15` over-generalized active strategies per seed at $t=200$. Minimum-Description-Length (MDL) programmatic precondition synthesis is needed for open-world text environments.

## 3. Heuristic Nature of Discounted & Adaptively Stopped SPRT
- **Current Scope**: EVSA uses an exponentially discounted ($\gamma=0.985$), clipped log-likelihood ratio $\Lambda_h$ combined with a minimum-trial floor ($n_{\min}=3$), Safe-LGG clamping, and adaptive Bayes-UCB candidate selection.
- **Unresolved Gap**: Classical Wald (1945) Type-I/II error bounds do not hold analytically under discounting, clipping, or adaptive bandit stopping. Although empirical candidate false-promotion rates closely match calibration ($\mathrm{FPR}_{\mathrm{cand}} \in [8.08\%, 11.32\%]$ vs. nominal $\alpha_{\mathrm{FA}}=11\%$), replacing heuristic boundaries with non-asymptotic anytime-valid $e$-processes and test martingales (Howard et al., 2021; Ramdas et al., 2023) is required for formal finite-sample error guarantees.

## 4. High False Rejection Rate (FRR) Under Severe Noise, Multiplicity Borderlines & Short-Horizon Coverage Lag
- **Current Scope**: Under $\eta = 0.35$ symmetric feedback corruption, a truly optimal strategy ($p_{\mathrm{opt}} = 0.92$) has apparent trial success $p_{\mathrm{obs}} = 0.626$, driving EVSA's False Rejection Rate to **`65.44%`**, shrinking clean promoted strategies to `2.15` per seed at $t=200$ (inflating noisy-phase FDP to `68.29%` vs. `83.12%–85.50%` for baselines; $p_{\mathrm{Holm}} = 0.0063$ per-family, borderline $p_{\mathrm{Holm,global74}} = 0.0510$ under simultaneous 74-test pooling, before recovering to `30.60%` FDP and `74.90%` SR once clean feedback returns). Similarly, in sparse-visit horizons (`Exp 5`, `~6.4` visits/cell at $\eta=0.15$), requiring $n_{\min}=3$ trials plus Safe-LGG confirmation yields `-1.71 pp` lower raw stream coverage than 1-trial admission ($p_{\mathrm{Holm}}=0.684$, non-significant) as the price for cutting active contamination more than $2\times$ (`33.18%` vs. `70.56%`) and repository bloat $4.8\times$ (`8.95` vs. `42.55` rules).
- **Unresolved Gap**: Online estimation of channel noise $\hat{\eta}_t$ to adapt $(p_0, p_1, n_{\min})$ dynamically across noise and horizon regimes remains open.

## 5. Non-Additive Parity Ceiling on Zero-Shot Compositional Tasks ($c=7$) & Post-Shift Discount Cost
- **Current Scope**: On held-out archetype $c=7$ ($\psi_0=\psi_1=\psi_2=1$), Steps 0–2 follow additive single-bit dependencies while Step 3 follows a non-linear XOR parity interaction ($\psi_0 \oplus \psi_1$). Separately, upon a CUSUM shift alarm in a shifted domain, unfalsified `VALIDATED` rules within that alarmed domain undergo conservative posterior discounting ($\gamma_{\mathrm{shift}}=0.35$) back to `CANDIDATE` until re-verified (whereas unshifted stationary control domains experience no cross-domain degradation: `82.21%` post-shift vs. `81.75%` pre-shift).
- **Unresolved Gap**: Composing validated marginal step operators on Steps 0–2 lifts EVSA to **`31.13% ± 7.67%`** on $c=7$ (`+13.19 pp` over `Static` at `17.94%`, $p_{\mathrm{Holm}} = 1.07 \times 10^{-6}$; `+19.00 pp` over `Replay` at `12.12%`), yet accuracy remains bounded well below seen-archetype generalization (`66.19%`) because first-order marginal salience cannot infer higher-order parity interactions zero-shot.
