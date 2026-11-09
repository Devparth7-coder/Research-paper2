# Phase 1: Problem Formulation, Literature Synthesis & Mathematical Foundations (`v2.0.1`)

**Project**: *Evidence-Validated Strategy Adaptation for Memory-Augmented Agents Under Distribution Shift*  
**Author**: Dev Parth (Department of Computer Science and Engineering, MMMUT, Gorakhpur, Uttar Pradesh, India)  
**Release Version**: `v2.0.1` | **Date**: 7 November 2026  

---

## 1. Phase Scope & Research Motivation

Non-parametric experiential memory enables autonomous agents to reuse successful workflows without gradient updates, yet conventional memory designs break down when task streams violate stationarity or evaluator signals carry noise:
1. **Single-Episode Epistemic Trust**: One-shot admission stores lucky suboptimal trajectories ($p_{\mathrm{sub3}} = 0.28$) or noise-flipped failures ($\tilde{y}_t = 1$ under corruption rate $\eta_t > 0$) without testing task-operator alignment.
2. **Surface-Embedding Collisions**: High cosine similarity ($\cos(x_t, x_{t'}) > 0.85$) over shared surface topics masks conflicting latent structural preconditions $\psi_t \in \{0,1\}^K$.
3. **Temporal Non-Stationarity**: Partial-domain concept shifts invalidate stored plans in shifted domains $\mathcal{D}_{\mathrm{shift}} \subset \mathcal{D}$ while stationary domains $\mathcal{D} \setminus \mathcal{D}_{\mathrm{shift}}$ remain valid.

---

## 2. Verified 30-Source Literature Synthesis (`paper/references.bib`)

Phase 1 synthesizes 30 primary peer-reviewed and archival works across four threads:
- **Agentic Episodic, Reflective & Strategy Memory**:
  - *ReAct* (`yao2023react`, ICLR 2023), *Reflexion* (`shinn2023reflexion`, NeurIPS 2023), *CLIN* (`majumder2024clin`, COLM 2024), *Generative Agents* (`park2023generative`, UIST 2023), *MemGPT* (`packer2023memgpt`, 2023), *Voyager* (`wang2024voyager`, TMLR 2024), *ExpeL* (`zhao2024expel`, AAAI 2024), *Agent Workflow Memory (AWM)* (`wang2025awm`, ICML 2025), *Contextual Experience Replay (CER)* (`liu2025cer`, ACL 2025), and *A-MEM* (`xu2025amem`, NeurIPS 2025).
  - *Dynamic Cheatsheet* (`suzgun2026dynamic`, EACL 2026 Long Papers, pp. 7080–7106, `doi:10.18653/v1/2026.eacl-long.333`): test-time curated adaptive memory notes without multi-trial sequential hypothesis gating.
  - *ReasoningBank* (`ouyang2026reasoningbank`, ICLR 2026, 17 authors including Siru Ouyang, Jiawei Han, Chen-Yu Lee, and Tomas Pfister): distills reasoning strategies from self-judged successful and failed trajectories paired with Memory-aware Test-Time Scaling (`MaTTS`).
  - *AdaMEM* (`zhang2026adamem`, ICML 2026, *AdaMEM: Test-Time Adaptive Memory for Language Agents*): integrates offline long-term trajectory memory with dynamically synthesized short-term strategy memory and step-wise memory fine-tuning (`STEP-MFT`).
  - *ExpRL* (`jin2026exprl`, Findings of EMNLP 2026): exploration bounds for reasoning-driven experience selection.
- **Memory Reliability, Confidence Calibration & Poisoning**:
  - Overconfidence of verbalized LLM uncertainty (`xiong2024can`, ICLR 2024), experience-following error propagation (`xiong2026memory`, ACL 2026), and memory/RAG poisoning attacks (*AgentPoison* `chen2024agentpoison`, NeurIPS 2024; *PoisonedRAG* `zou2025poisonedrag`, USENIX Security 2025).
- **Sequential Testing, Bayesian Credible Bounds & Multiple Testing**:
  - Wald's Sequential Probability Ratio Test (`wald1945sequential`), Bayes-UCB (`kaufmann2012bayesian`, AISTATS 2012), non-asymptotic confidence sequences (`howard2021time`, AoS 2021), anytime-valid $e$-processes and test martingales (`ramdas2023game`, Stat. Sci. 2023), and Holm's sequentially rejective procedure (`holm1979simple`).
- **Concept Drift, Continual Adaptation & Symbolic Generalization**:
  - Page's tabular CUSUM (`page1954continuous`), ADWIN (`bifet2007learning`), concept drift taxonomy (`gama2014survey`), Plotkin's Least General Generalization (`plotkin1970inductive`), PER (`schaul2016prioritized`), EWC (`kirkpatrick2017overcoming`), and Tent (`wang2021tent`).

---

## 3. Formal Research Questions (`RQ1`–`RQ5`) & Theoretical Guardrails

1. **`RQ1` (Unseen Generalization & Execution Attribution)**: Held-out unseen task generalization (`VALIDATED`-only frozen execution) and controlled $2 \times 2$ separation of epistemic strategy selection from deterministic macro execution.
2. **`RQ2` (Continual Learning & Compactness)**: Monotonic competence accumulation without early negative transfer across 320 continual episodes.
3. **`RQ3` (Component Necessity & Horizon Trade-Offs)**: Leave-one-out ablations across sequential validation, Bayesian credible bounds, conjunctive preconditions, CUSUM drift detection, and selective invalidation.
4. **`RQ4` (Post-Shift Recovery & Cross-Domain Retention)**: Adaptation gain after abrupt 3-domain concept shifts and retention on unshifted stationary control domains.
5. **`RQ5` (Noisy-Feedback Calibration & Overhead)**: Rollout FAR, Candidate FPR, FRR, active store FDP decomposition, clean recovery across $\eta \in \{0.10, 0.25, 0.35\}$, and local CPU vs. projected token economics.

**Mathematical Honesty Guardrails**:
- Discounted $\mathrm{Beta}(\alpha_h, \beta_h)$ quantiles (`src/validation/confidence.py`) are framed strictly as **Bayesian credible bounds**, distinct from frequentist Clopper-Pearson intervals.
- Discounted ($\gamma=0.985$), clipped ($\pm 6.0$), adaptively selected sequential log-odds stopping (`src/validation/sprt_validator.py`) is framed strictly as an **empirically calibrated sequential heuristic** motivated by Wald's SPRT rather than claiming analytical i.i.d. Wald error guarantees.
