#!/usr/bin/env python3
"""
Generate all 10 required publication-grade figures (300-DPI .png and vector .pdf)
in figures/ directly from verified v2.0.1 datasets in experiments/.
"""

from __future__ import annotations

import math
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import pandas as pd
from scipy.special import betaincinv

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "experiments" / "raw"
PROC_DIR = ROOT / "experiments" / "processed"
FIG_DIR = ROOT / "figures"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9.5,
    "axes.titlesize": 10.8,
    "axes.titleweight": "bold",
    "axes.labelsize": 9.5,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "legend.fontsize": 8.3,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "--",
})

PALETTE = {
    "Baseline A: Static": "#64748B",
    "Baseline B: Conventional Memory": "#D97706",
    "Baseline C: Reflection": "#0284C7",
    "Baseline D: Experience Replay": "#BE185D",
    "Proposed Agent E: EVSA": "#0F766E",
}
SHORT_LABELS = {
    "Baseline A: Static": "Static (A)",
    "Baseline B: Conventional Memory": "Conv. Mem (B)",
    "Baseline C: Reflection": "Reflection (C)",
    "Baseline D: Experience Replay": "Exp. Replay (D)",
    "Proposed Agent E: EVSA": "EVSA (Proposed)",
}
MARKERS = {
    "Baseline A: Static": "s",
    "Baseline B: Conventional Memory": "^",
    "Baseline C: Reflection": "D",
    "Baseline D: Experience Replay": "v",
    "Proposed Agent E: EVSA": "o",
}
AGENT_ORDER = list(PALETTE.keys())


def save_both(fig: plt.Figure, stem: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG_DIR / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def draw_box(ax, x, y, w, h, title, subtitle, fc, ec, lw=1.6):
    patch = FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.35,rounding_size=0.8",
        facecolor=fc, edgecolor=ec, linewidth=lw,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h * 0.65, title, ha="center", va="center", fontsize=8.8, fontweight="bold", color="#0F172A")
    ax.text(x + w / 2, y + h * 0.28, subtitle, ha="center", va="center", fontsize=7.5, color="#334155")


def draw_arrow(ax, x1, y1, x2, y2, label="", color="#334155", lx=0.0, ly=0.0, rad=0.0):
    arr = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle="-|>,head_length=5.5,head_width=3.2",
        connectionstyle=f"arc3,rad={rad}",
        linewidth=1.35, color=color,
    )
    ax.add_patch(arr)
    if label:
        ax.text(
            0.5 * (x1 + x2) + lx, 0.5 * (y1 + y2) + ly, label,
            ha="center", va="center", fontsize=7.3, fontweight="bold", color=color,
            bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="none", alpha=0.88),
        )


# ---------------------------------------------------------------------------
# Figure 1: EVSA System Architecture
# ---------------------------------------------------------------------------
def make_fig01_architecture() -> None:
    fig, ax = plt.subplots(figsize=(12.2, 6.6))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 56)
    ax.axis("off")
    ax.text(50, 53.6, "Evidence-Validated Strategy Adaptation (EVSA) — Closed-Loop Architecture",
            ha="center", va="center", fontsize=12.5, fontweight="bold", color="#0F172A")

    draw_box(ax, 3, 41, 41, 9.5, "1. Non-Stationary Task Environment (ShiftBench-Sim)",
             "4 Domains x 240 Workflows | Covariate & Concept Shifts | Noise eta", "#EFF6FF", "#1D4ED8")
    draw_box(ax, 55, 41, 42, 9.5, "10. Evaluation & Statistical Audit Engine",
             "SR, Adaptation Gain, FAR/FPR/FDP, Latency & Token Estimates", "#F8FAFC", "#475569")

    draw_box(ax, 3, 27, 27, 9.5, "7. Applicability Retriever",
             "Precondition Gate phi_h(psi_t)\n+ Cosine Sim + Beta Credible Bound", "#ECFDF5", "#047857")
    draw_box(ax, 36, 27, 28, 9.5, "4. Experimentation Engine",
             "Budget-Matched Rollouts (B)\n& Candidate Verification", "#FFFBEB", "#B45309")
    draw_box(ax, 71, 27, 26, 9.5, "9. Residual Drift Detector",
             "Per-Domain Tabular CUSUM\ne_t = p_hat(h*) - y_tilde_t", "#FEF2F2", "#B91C1C")

    draw_box(ax, 3, 13.5, 27, 9.5, "2. Hypothesis Generator",
             "Marginal Predicate Salience\n& Compositional Synthesis", "#F5F3FF", "#6D28D9")
    draw_box(ax, 35.5, 13.5, 29, 9.5, "3. Tiered Strategy Repository",
             "CANDIDATE -> VALIDATED\n-> QUARANTINED | Domain Index", "#ECFEFF", "#0E7490", lw=2.0)
    draw_box(ax, 71, 13.5, 26, 9.5, "8. Adaptive Memory Manager",
             "LGG Precondition Intersection,\nConflict Demotion & Pruning", "#FDF2F8", "#9D174D")

    draw_box(ax, 12, 1.2, 34, 8.8, "5. Sequential Validation Gate",
             "Discounted SPRT Log-Odds (Lambda_h) + Min-Trials (n >= 3)", "#F0F9FF", "#0369A1")
    draw_box(ax, 54, 1.2, 35, 8.8, "6. Bayesian Reliability Estimator",
             "Discounted Beta(alpha, beta) Lower Credible Bound", "#F0F9FF", "#0369A1")

    draw_arrow(ax, 16.5, 41, 16.5, 36.5, "Task q_t", "#1D4ED8", lx=-4.5)
    draw_arrow(ax, 48, 36.5, 34, 41, "Plan pi", "#B45309", lx=-3.5, ly=-0.6)
    draw_arrow(ax, 42, 41, 78, 36.5, "Feedback y_tilde_t", "#B91C1C", lx=2.5, ly=1.0)
    draw_arrow(ax, 44, 45.8, 55, 45.8, "Telemetry", "#475569", ly=1.8)
    draw_arrow(ax, 30, 31.8, 36, 31.8, "Top Match", "#047857", ly=2.2)
    draw_arrow(ax, 42, 27, 18, 23, "Explore / Fail", "#6D28D9", lx=-4.0, ly=0.8)
    draw_arrow(ax, 30, 18.2, 35.5, 18.2, "Admit", "#6D28D9", ly=-2.3)
    draw_arrow(ax, 50, 23, 24, 27, "Active Pool", "#0E7490", lx=4.5, ly=-0.5)
    draw_arrow(ax, 84, 27, 84, 23, "Drift Alarm", "#B91C1C", lx=5.2)
    draw_arrow(ax, 71, 18.2, 64.5, 18.2, "Quarantine / LGG", "#9D174D", ly=-2.6)
    draw_arrow(ax, 40, 13.5, 28, 10.0, "Outcome", "#0369A1", lx=-4.5, ly=0.4)
    draw_arrow(ax, 41, 10.0, 50, 13.5, "Promote / Reject", "#0369A1", lx=5.5, ly=-0.6)
    draw_arrow(ax, 54, 5.6, 46, 5.6, "Credible Bound", "#0369A1", ly=2.1)

    save_both(fig, "fig01_system_architecture")


# ---------------------------------------------------------------------------
# Figure 2: Strategy Validation Lifecycle & Sequential Trajectories
# ---------------------------------------------------------------------------
def make_fig02_validation_workflow() -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.0, 4.5), gridspec_kw={"width_ratios": [1.05, 1.15]})

    # Panel (a): Lifecycle State Machine
    ax1.set_xlim(0, 100)
    ax1.set_ylim(0, 62)
    ax1.axis("off")
    ax1.set_title("(a) Three-Tier Epistemic Strategy Lifecycle", pad=8)

    draw_box(ax1, 6, 39, 38, 14, "CANDIDATE", "Unverified Hypothesis\nTested via Bayes-UCB", "#FEF3C7", "#D97706", lw=1.8)
    draw_box(ax1, 56, 39, 38, 14, "VALIDATED", "Trusted Exploitation\nMonitored by CUSUM", "#CCFBF1", "#0F766E", lw=2.0)
    draw_box(ax1, 31, 6, 38, 14, "QUARANTINED", "Tombstone Signature\nBlocks Re-Proposal", "#FEE2E2", "#B91C1C", lw=1.8)

    draw_arrow(ax1, 44, 48, 56, 48, "n>=3, Lambda>=2.05\nLCB>=0.48", "#0F766E", ly=4.5)
    draw_arrow(ax1, 56, 43, 44, 43, "Conflict / Safe-LGG", "#D97706", ly=-4.2)
    draw_arrow(ax1, 22, 39, 38, 20, "Lambda<=-1.78\nor 0/2 Fails", "#B91C1C", lx=-9.5, ly=-1.0)
    draw_arrow(ax1, 78, 39, 62, 20, "CUSUM Shift Fail or\n2 Fails & LCB<0.45", "#B91C1C", lx=9.5, ly=-1.0)

    # Panel (b): Exact Deterministic Trajectories of Discounted LLR & Beta Credible Bound
    p0, p1, gamma = 0.35, 0.80, 0.985
    pos_inc = math.log(p1 / p0)
    neg_inc = math.log((1 - p1) / (1 - p0))
    upper_b = math.log((1 - 0.15) / 0.11)
    lower_b = math.log(0.15 / (1 - 0.11))

    def sim_seq(outcomes):
        llr, a, b = 0.0, 1.0, 1.0
        llrs, lcbs = [0.0], [float(betaincinv(1.0, 1.0, 0.10))]
        for y in outcomes:
            llr = max(-6.0, min(6.0, gamma * llr + (pos_inc if y == 1 else neg_inc)))
            a = 1.0 + gamma * (a - 1.0) + y
            b = 1.0 + gamma * (b - 1.0) + (1 - y)
            llrs.append(llr)
            lcbs.append(float(betaincinv(a, b, 0.10)))
        return np.array(llrs), np.array(lcbs)

    seq_opt = [1, 1, 1, 1, 1, 0, 0, 0]       # Promotes at trial 3, shifts at trial 6 -> quarantined!
    seq_fluke = [1, 0, 0, 0]                 # 1-trial lucky fluke -> rejected at trial 3!
    t_opt = np.arange(len(seq_opt) + 1)
    t_flk = np.arange(len(seq_fluke) + 1)
    llr_opt, _ = sim_seq(seq_opt)
    llr_flk, _ = sim_seq(seq_fluke)

    ax2.plot(t_opt, llr_opt, "-o", color="#0F766E", lw=2.2, label="Optimal Strategy (Promoted t=3, Shifted t=6)")
    ax2.plot(t_flk, llr_flk, "--s", color="#BE185D", lw=2.0, label="Lucky 3-Step Fluke [1, 0, 0] (Rejected)")
    ax2.axhline(upper_b, color="#047857", ls=":", lw=1.4, label=f"Promotion Threshold A = +{upper_b:.2f}")
    ax2.axhline(lower_b, color="#B91C1C", ls=":", lw=1.4, label=f"Quarantine Threshold B = {lower_b:.2f}")
    ax2.axvline(5.0, color="#64748B", ls="--", alpha=0.6)
    ax2.text(5.12, 3.5, "Concept Shift\n(Post-Promotion)", fontsize=7.8, color="#475569")
    ax2.set_xlabel("Sequential Strategy Verification / Execution Trial (n)")
    ax2.set_ylabel("Discounted Log-Likelihood Ratio (Lambda_h)")
    ax2.set_title("(b) Sequential Evidence Gating & Post-Shift Quarantine", pad=8)
    ax2.set_ylim(-3.4, 4.6)
    ax2.legend(loc="lower left", frameon=True, fontsize=7.6)

    fig.tight_layout()
    save_both(fig, "fig02_validation_workflow")


# ---------------------------------------------------------------------------
# Figure 3: Applicability-Gated Retrieval & Execution-Channel Parity
# ---------------------------------------------------------------------------
def make_fig03_applicability_retrieval() -> None:
    df_fair = pd.read_csv(PROC_DIR / "sensitivity_execution_parity_summary.csv")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.2, 4.5), gridspec_kw={"width_ratios": [1.05, 1.15]})

    # Panel (a): Schematic of Precondition Gate + LGG Abstraction
    ax1.set_xlim(0, 100)
    ax1.set_ylim(0, 60)
    ax1.axis("off")
    ax1.set_title("(a) Applicability Gate & LGG Nuisance Abstraction", pad=8)

    draw_box(ax1, 4, 38, 42, 14, "Incoming Task q_t",
             "Topic='Billing' (0.82 wt)\npsi_t = (1, 0, 1 | nuis: 0, 1)", "#EFF6FF", "#1D4ED8")
    draw_box(ax1, 54, 42, 42, 13, "Surface-Similar Stale Record",
             "Topic='Billing' (cos=0.91)\nphi_h = {(0,0),(1,0),(2,1)} -> BLOCKED", "#FEE2E2", "#B91C1C")
    draw_box(ax1, 54, 22, 42, 14, "LGG-Generalized Valid Strategy",
             "phi_h = {(0,1),(1,0),(2,1)} (nuis dropped)\nLCB = 0.74 -> RETRIEVED", "#CCFBF1", "#0F766E", lw=1.9)
    draw_box(ax1, 3, 3, 94, 12.5, "U(h, q_t) = phi_h(psi_t) * I[LCB>=0.45] * [0.28 cos + 0.52 LCB + 0.20]",
             "Blocks surface collisions & weak rules while Safe-LGG abstracts nuisance bits", "#F8FAFC", "#475569")

    draw_arrow(ax1, 46, 47, 54, 48, "phi_h=0", "#B91C1C", ly=2.5)
    draw_arrow(ax1, 46, 42, 54, 30, "phi_h=1", "#0F766E", lx=-2.0, ly=-2.5)

    # Panel (b): Empirical Execution-Channel Fairness Parity Study (N=20 seeds)
    labels = [
        "EVSA\n(Direct Macro)",
        "EVSA\n(Stoch. Prompt)",
        "Exp. Replay\n(Direct Macro)",
        "Exp. Replay\n(Stoch. Prompt)",
    ]
    agent_keys = [
        "EVSA (Direct Macro Exec)",
        "EVSA (Stochastic Prompt Exec)",
        "Replay (Direct Macro Exec)",
        "Replay (Stochastic Prompt Exec)",
    ]
    sub = df_fair.set_index("agent").loc[agent_keys]
    x = np.arange(len(labels))
    w = 0.26

    ax2.bar(x - w, sub["overall_sr_mean"] * 100, w,
            yerr=(sub["overall_sr_ci95_high"] - sub["overall_sr_mean"]) * 100,
            capsize=3, color="#0F766E", edgecolor="#0F172A", lw=0.7, label="Overall Benchmark SR")
    ax2.bar(x, sub["unseen_instance_sr_mean"] * 100, w,
            yerr=(sub["unseen_instance_sr_ci95_high"] - sub["unseen_instance_sr_mean"]) * 100,
            capsize=3, color="#0284C7", edgecolor="#0F172A", lw=0.7, label="Unseen Instances SR")
    ax2.bar(x + w, sub["unseen_compositional_sr_mean"] * 100, w,
            yerr=(sub["unseen_compositional_sr_ci95_high"] - sub["unseen_compositional_sr_mean"]) * 100,
            capsize=3, color="#D97706", edgecolor="#0F172A", lw=0.7, label="Zero-Shot Comp. SR")

    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=8.0)
    ax2.set_ylabel("Held-Out Task Success Rate (%)")
    ax2.set_ylim(0, 86)
    ax2.set_title("(b) Execution-Channel Fairness Control (N = 20 Seeds, 95% CI)", pad=8)
    ax2.legend(loc="upper right", frameon=True, fontsize=7.8)

    fig.tight_layout()
    save_both(fig, "fig03_applicability_retrieval")


# ---------------------------------------------------------------------------
# Figure 4: Empirical Distribution-Shift CUSUM Detection Analysis
# ---------------------------------------------------------------------------
def make_fig04_drift_detection() -> None:
    df_cusum = pd.read_csv(PROC_DIR / "exp3_domain_cusum_summary.csv")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10.8, 5.8), sharex=True)

    dom_styles = {
        "api_orchestration": ("#B91C1C", "-", "api_orchestration (Shifts at t=160 & t=320)"),
        "system_diagnosis": ("#D97706", "--", "system_diagnosis (Shifts at t=160 & t=320)"),
        "data_pipeline": ("#6D28D9", "-.", "data_pipeline (Shifts at t=160 & t=320)"),
        "resource_planning": ("#0F766E", "-", "resource_planning (Stationary Control in Phase II; Shifts at t=320)"),
    }

    for dom, (col, ls, lbl) in dom_styles.items():
        sub = df_cusum[df_cusum["domain"] == dom].sort_values("episode_idx")
        # Smooth 5-episode rolling mean for visual clarity across 420 episodes
        y_stat = sub["cusum_stat_mean"].rolling(5, min_periods=1).mean()
        ax1.plot(sub["episode_idx"], y_stat, color=col, linestyle=ls, lw=1.8, label=lbl)

        # Cumulative drift alarms per seed
        cum_alarms = sub["alarm_fired_mean"].cumsum()
        ax2.plot(sub["episode_idx"], cum_alarms, color=col, linestyle=ls, lw=2.0, label=lbl)

    for ax in (ax1, ax2):
        ax.axvline(160, color="#B91C1C", ls="--", lw=1.4, alpha=0.8)
        ax.axvline(320, color="#B45309", ls="--", lw=1.4, alpha=0.8)

    ax1.set_ylim(0, 0.92)
    ax1.text(80, 0.80, "Phase I: Stationary (R0)", ha="center", fontsize=8.5, fontweight="bold")
    ax1.text(240, 0.80, "Phase II: Partial Concept Shift (R1)", ha="center", fontsize=8.5, fontweight="bold", color="#991B1B")
    ax1.text(370, 0.80, "Phase III: Compound (R2)", ha="center", fontsize=8.5, fontweight="bold", color="#92400E")

    ax1.set_ylabel("Mean CUSUM Stat S_d^(t)")
    ax1.set_title("Empirical Per-Domain Residual CUSUM Monitoring Across Multi-Regime Shifts (N = 20 Seeds)", pad=8)

    ax2.set_xlabel("Sequential Task Stream Episode Index (t)")
    ax2.set_ylabel("Cumulative Alarms / Seed")
    ax2.set_ylim(-0.04, 1.45)
    ax2.set_title("Domain-Scoped Drift Alarm Accumulation (Near-Zero False Alarms on Unshifted Control Domain in Phase II)", pad=8)
    ax2.legend(loc="upper left", fontsize=7.6, frameon=True)

    fig.tight_layout()
    save_both(fig, "fig04_drift_detection")


# ---------------------------------------------------------------------------
# Figure 5: Experiment 1 Unseen Task Generalization (with Seed Distributions)
# ---------------------------------------------------------------------------
def make_fig05_exp1_generalization() -> None:
    df_sum = pd.read_csv(PROC_DIR / "exp1_baseline_summary.csv").set_index("agent").loc[AGENT_ORDER]
    df_raw = pd.read_csv(RAW_DIR / "exp1_baseline_raw_seeds.csv")

    fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.4), sharey=True)
    panels = [
        ("task_success_rate", "(a) Overall Unseen Benchmark (240 Tasks)"),
        ("unseen_instance_sr", "(b) Unseen Instances of Seen Archetypes"),
        ("unseen_compositional_sr", "(c) Zero-Shot Compositional Archetype c=7"),
    ]
    rng = np.random.default_rng(2026)

    for ax, (col, title) in zip(axes, panels):
        x = np.arange(len(AGENT_ORDER))
        means = df_sum[f"{col}_mean"].to_numpy() * 100.0
        err_lo = (df_sum[f"{col}_mean"] - df_sum[f"{col}_ci95_low"]).to_numpy() * 100.0
        err_hi = (df_sum[f"{col}_ci95_high"] - df_sum[f"{col}_mean"]).to_numpy() * 100.0
        colors_l = [PALETTE[a] for a in AGENT_ORDER]

        bars = ax.bar(x, means, width=0.58, color=colors_l, alpha=0.82, edgecolor="#0F172A", lw=0.8,
                      yerr=[err_lo, err_hi], capsize=3.5, error_kw={"lw": 1.2, "zorder": 5})

        for idx, ag in enumerate(AGENT_ORDER):
            pts = df_raw.loc[df_raw["agent"] == ag, col].to_numpy() * 100.0
            jitter = rng.uniform(-0.15, 0.15, size=pts.size)
            ax.scatter(np.full(pts.size, idx) + jitter, pts, s=14, color="#0F172A", alpha=0.45, zorder=4)
            ax.text(idx, means[idx] + err_hi[idx] + 2.2, f"{means[idx]:.1f}%",
                    ha="center", va="bottom", fontsize=8.0, fontweight="bold")

        ax.set_xticks(x)
        ax.set_xticklabels([SHORT_LABELS[a] for a in AGENT_ORDER], rotation=22, ha="right")
        ax.set_title(title, fontsize=9.8)
        ax.set_ylim(0, 92)

    axes[0].set_ylabel("Held-Out Task Success Rate (%)")
    fig.suptitle("Experiment 1: Unseen Task Generalization Across N = 20 Seeds (Mean, 95% CI, and Seed Distribution)",
                 fontsize=11.2, fontweight="bold", y=1.02)
    fig.tight_layout()
    save_both(fig, "fig05_exp1_generalization")


# ---------------------------------------------------------------------------
# Figure 6: Experiment 2 Continual Learning Curves
# ---------------------------------------------------------------------------
def make_fig06_exp2_learning() -> None:
    df = pd.read_csv(PROC_DIR / "exp2_learning_curves_summary.csv")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.8, 4.4))

    for ag in AGENT_ORDER:
        sub = df[df["agent"] == ag].sort_values("tasks_seen")
        x = sub["tasks_seen"].to_numpy()
        y = sub["eval_success_rate_mean"].to_numpy() * 100.0
        lo = sub["eval_success_rate_ci95_low"].to_numpy() * 100.0
        hi = sub["eval_success_rate_ci95_high"].to_numpy() * 100.0

        lw = 2.4 if "EVSA" in ag else 1.6
        ax1.plot(x, y, marker=MARKERS[ag], color=PALETTE[ag], lw=lw, label=SHORT_LABELS[ag])
        ax1.fill_between(x, lo, hi, color=PALETTE[ag], alpha=0.14)

        ax2.plot(x, sub["active_memory_items_mean"].to_numpy(), marker=MARKERS[ag],
                 color=PALETTE[ag], lw=lw, label=SHORT_LABELS[ag])

    ax1.set_xlabel("Cumulative Training Episodes Seen")
    ax1.set_ylabel("Held-Out Probe Success Rate (%)")
    ax1.set_title("(a) Continual Learning Curve (N = 20 Seeds, 95% CI)")
    ax1.set_ylim(18, 78)
    ax1.legend(loc="upper left", frameon=True)

    ax2.set_xlabel("Cumulative Training Episodes Seen")
    ax2.set_ylabel("Active Stored Memory Records")
    ax2.set_title("(b) Active Memory Growth vs. LGG/SPRT Consolidation")
    ax2.legend(loc="upper left", frameon=True)

    fig.tight_layout()
    save_both(fig, "fig06_exp2_continual_learning")


# ---------------------------------------------------------------------------
# Figure 7: Experiment 3 Multi-Regime Shift Trajectory & Selective Retention
# ---------------------------------------------------------------------------
def make_fig07_exp3_shift() -> None:
    df_win = pd.read_csv(PROC_DIR / "exp3_shift_windows_summary.csv")
    df_sum = pd.read_csv(PROC_DIR / "exp3_shift_summary.csv").set_index("agent").loc[AGENT_ORDER]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.6, 4.6), gridspec_kw={"width_ratios": [1.35, 1.0]})

    ax1.axvspan(0, 160, color="#F8FAFC", alpha=0.9)
    ax1.axvspan(160, 320, color="#FEF2F2", alpha=0.55)
    ax1.axvspan(320, 420, color="#FFFBEB", alpha=0.65)
    ax1.axvline(160, color="#B91C1C", ls="--", lw=1.5)
    ax1.axvline(320, color="#B45309", ls="--", lw=1.5)

    for ag in AGENT_ORDER:
        sub = df_win[df_win["agent"] == ag].sort_values("window_mid")
        x = sub["window_mid"].to_numpy()
        y = sub["success_rate_mean"].to_numpy() * 100.0
        lo = sub["success_rate_ci95_low"].to_numpy() * 100.0
        hi = sub["success_rate_ci95_high"].to_numpy() * 100.0
        lw = 2.4 if "EVSA" in ag else 1.5
        ax1.plot(x, y, marker=MARKERS[ag], markersize=4.5, color=PALETTE[ag], lw=lw, label=SHORT_LABELS[ag])
        ax1.fill_between(x, lo, hi, color=PALETTE[ag], alpha=0.12)

    ax1.text(80, 91, "Phase I: Stationary", ha="center", fontsize=8.3, fontweight="bold")
    ax1.text(240, 91, "Phase II: Concept Shift", ha="center", fontsize=8.3, fontweight="bold", color="#991B1B")
    ax1.text(370, 91, "Phase III:\nCompound", ha="center", fontsize=8.0, fontweight="bold", color="#92400E")
    ax1.set_xlabel("Sequential Task Stream Episode Index (t)")
    ax1.set_ylabel("20-Episode Window Success Rate (%)")
    ax1.set_ylim(5, 98)
    ax1.set_title("(a) Multi-Regime Adaptation Trajectory (95% CI)")
    ax1.legend(loc="lower left", ncol=2, fontsize=7.4, frameon=True)

    x = np.arange(len(AGENT_ORDER))
    w = 0.26
    ax2.bar(x - w, df_sum["recovered_post_shift_sr_mean"] * 100, w, color="#0F766E",
            edgecolor="#0F172A", lw=0.7, label="Post-Shift Recovered SR")
    ax2.bar(x, df_sum["unshifted_domain_sr_mean"] * 100, w, color="#0284C7",
            edgecolor="#0F172A", lw=0.7, label="Unshifted Domain SR")
    ax2.bar(x + w, df_sum["adaptation_gain_mean"] * 100, w, color="#D97706",
            edgecolor="#0F172A", lw=0.7, label="Adaptation Gain (pp)")
    ax2.axhline(0, color="#334155", lw=0.8)
    ax2.set_xticks(x)
    ax2.set_xticklabels([SHORT_LABELS[a] for a in AGENT_ORDER], rotation=20, ha="right")
    ax2.set_ylabel("Percentage / Percentage Points")
    ax2.set_ylim(-6, 95)
    ax2.set_title("(b) Recovery, Selective Retention & Adaptation Gain")
    ax2.legend(loc="upper left", fontsize=7.6, frameon=True)

    fig.tight_layout()
    save_both(fig, "fig07_exp3_shift_adaptation")


# ---------------------------------------------------------------------------
# Figure 8: Experiment 4 Feedback Corruption & Contamination Decomposition
# ---------------------------------------------------------------------------
def make_fig08_exp4_corruption() -> None:
    df = pd.read_csv(PROC_DIR / "exp4_misleading_feedback_summary.csv")
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(13.2, 4.5))

    etas = [0.10, 0.25, 0.35]
    x = np.arange(len(etas))
    mem_agents = AGENT_ORDER[1:]
    w = 0.19

    # Panel (a): Rollout FAR across memory agents + EVSA Candidate FPR overlay
    for idx, ag in enumerate(mem_agents):
        sub = df[df["agent"] == ag].set_index("corruption_rate_eta").loc[etas]
        ax1.bar(x + (idx - 1.5) * w, sub["false_acceptance_rate_mean"] * 100, w,
                color=PALETTE[ag], edgecolor="#0F172A", lw=0.7, label=SHORT_LABELS[ag])

    evsa_sub = df[df["agent"] == "Proposed Agent E: EVSA"].set_index("corruption_rate_eta").loc[etas]
    ax1.plot(x + 1.5 * w, evsa_sub["candidate_false_promotion_rate_mean"] * 100, "o--",
             color="#0F172A", lw=1.4, markersize=5, label="EVSA Candidate FPR (%)")
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"eta = {int(e*100)}%" for e in etas])
    ax1.set_ylabel("False Acceptance / Promotion Rate (%)")
    ax1.set_ylim(0, 52)
    ax1.set_title("(a) Rollout FAR & Candidate FPR")
    ax1.legend(loc="upper left", ncol=2, fontsize=6.9, frameon=True)

    # Panel (b): EVSA Validated Store Decomposition (Why low FAR coexists with high FDP!)
    clean_v = evsa_sub["clean_validated_end_noisy_mean"].to_numpy()
    fp_v = evsa_sub["false_promoted_active_noisy_mean"].to_numpy()
    lgg_v = evsa_sub["lgg_contaminated_active_noisy_mean"].to_numpy()

    ax2.bar(x, clean_v, width=0.48, color="#10B981", edgecolor="#0F172A", lw=0.7, label="Clean Optimal Validated")
    ax2.bar(x, fp_v, width=0.48, bottom=clean_v, color="#EF4444", edgecolor="#0F172A", lw=0.7, label="Creation-Time False Promotions")
    ax2.bar(x, lgg_v, width=0.48, bottom=clean_v + fp_v, color="#F59E0B", edgecolor="#0F172A", lw=0.7, label="Corrupted LGG Over-Generalizations")
    for i in range(len(etas)):
        tot = clean_v[i] + fp_v[i] + lgg_v[i]
        cont_pct = evsa_sub["memory_contamination_rate_mean"].iloc[i] * 100
        frr_pct = evsa_sub["false_rejection_rate_mean"].iloc[i] * 100
        ax2.text(i, tot + 0.35, f"FDP={cont_pct:.1f}%\n(FRR={frr_pct:.0f}%)", ha="center", va="bottom", fontsize=7.5, fontweight="bold")

    ax2.set_xticks(x)
    ax2.set_xticklabels([f"eta = {int(e*100)}%" for e in etas])
    ax2.set_ylabel("Mean Active VALIDATED Strategies (t=200)")
    ax2.set_ylim(0, 17.5)
    ax2.set_title("(b) EVSA Validated Store Decomposition")
    ax2.legend(loc="upper right", fontsize=7.2, frameon=True)

    # Panel (c): Clean Recovery SR after corruption ceases
    w5 = 0.15
    for idx, ag in enumerate(AGENT_ORDER):
        sub = df[df["agent"] == ag].set_index("corruption_rate_eta").loc[etas]
        ax3.bar(x + (idx - 2) * w5, sub["recovery_after_misleading_sr_mean"] * 100, w5,
                yerr=(sub["recovery_after_misleading_sr_ci95_high"] - sub["recovery_after_misleading_sr_mean"]) * 100,
                capsize=2.5, color=PALETTE[ag], edgecolor="#0F172A", lw=0.7, label=SHORT_LABELS[ag])

    ax3.set_xticks(x)
    ax3.set_xticklabels([f"eta = {int(e*100)}%" for e in etas])
    ax3.set_ylabel("Post-Corruption Clean Recovery SR (%)")
    ax3.set_ylim(0, 96)
    ax3.set_title("(c) Clean Recovery Accuracy (95% CI)")
    ax3.legend(loc="upper right", ncol=2, fontsize=7.0, frameon=True)

    fig.tight_layout()
    save_both(fig, "fig08_exp4_feedback_corruption")


# ---------------------------------------------------------------------------
# Figure 9: Experiment 5 Leave-One-Out Ablation Study
# ---------------------------------------------------------------------------
def make_fig09_exp5_ablation() -> None:
    df = pd.read_csv(PROC_DIR / "exp5_ablation_summary.csv")
    order = [
        "EVSA (Full)",
        "w/o Evidence Validation",
        "w/o Confidence Estimation",
        "w/o Applicability Conditions",
        "w/o Shift Detection",
        "w/o Memory Invalidation",
    ]
    df = df.set_index("variant").loc[order][::-1]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.2, 4.7), sharey=True, gridspec_kw={"width_ratios": [1.35, 1.0]})
    y = np.arange(len(order))
    h = 0.25

    ax1.barh(y + h, df["overall_stream_sr_mean"] * 100, h,
             xerr=(df["overall_stream_sr_ci95_high"] - df["overall_stream_sr_mean"]) * 100,
             capsize=2.5, color="#0F766E", label="Overall Stream SR")
    ax1.barh(y, df["post_shift_recovered_sr_mean"] * 100, h,
             xerr=(df["post_shift_recovered_sr_ci95_high"] - df["post_shift_recovered_sr_mean"]) * 100,
             capsize=2.5, color="#0284C7", label="Post-Shift Recovered SR")
    ax1.barh(y - h, df["heldout_unseen_sr_mean"] * 100, h,
             xerr=(df["heldout_unseen_sr_ci95_high"] - df["heldout_unseen_sr_mean"]) * 100,
             capsize=2.5, color="#D97706", label="Held-Out Unseen SR")
    ax1.set_yticks(y)
    ax1.set_yticklabels(df.index, fontweight="bold")
    ax1.set_xlabel("Task Success Rate (%)")
    ax1.set_xlim(0, 72)
    ax1.set_title("(a) Ablation Accuracy Across Regimes (95% CI)", pad=24)
    ax1.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=3, frameon=True, fontsize=7.8)

    ax2.barh(y + 0.14, df["pre_shift_contamination_mean"] * 100, 0.28, color="#EC4899", label="Noisy Pre-Shift FDP")
    ax2.barh(y - 0.14, df["post_shift_contamination_mean"] * 100, 0.28, color="#EA580C", label="Post-Shift FDP")
    for idx_i, v in enumerate(df["post_shift_contamination_mean"] * 100):
        ax2.text(v + 1.2, idx_i - 0.14, f"{v:.1f}%", va="center", fontsize=7.8, fontweight="bold")
    ax2.set_xlabel("Active Validated Contamination / FDP (%)")
    ax2.set_xlim(0, 103)
    ax2.set_title("(b) Strategy Store Contamination Rate", pad=24)
    ax2.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=2, frameon=True, fontsize=7.8)

    fig.tight_layout()
    save_both(fig, "fig09_exp5_ablation")


# ---------------------------------------------------------------------------
# Figure 10: Experiment 6 Computational Efficiency Comparison
# ---------------------------------------------------------------------------
def make_fig10_exp6_efficiency() -> None:
    df = pd.read_csv(PROC_DIR / "exp6_efficiency_summary.csv").set_index("agent").loc[AGENT_ORDER]
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(11.8, 7.6))
    x = np.arange(len(AGENT_ORDER))
    cols = [PALETTE[a] for a in AGENT_ORDER]
    slabels = [SHORT_LABELS[a] for a in AGENT_ORDER]

    # (a) Accuracy vs Estimated Token Budget
    offsets = {
        "Baseline A: Static": (1.5, -2.5),
        "Baseline B: Conventional Memory": (1.5, 1.5),
        "Baseline C: Reflection": (-22.0, 2.5),
        "Baseline D: Experience Replay": (1.5, -2.0),
        "Proposed Agent E: EVSA": (2.0, -1.0),
    }
    for ag in AGENT_ORDER:
        r = df.loc[ag]
        tx = r["total_tokens_mean"] / 1000.0
        sy = r["task_success_rate_mean"] * 100.0
        ax1.scatter(tx, sy, s=125, marker=MARKERS[ag], color=PALETTE[ag], edgecolor="#0F172A", lw=1.1, zorder=4)
        dx, dy = offsets[ag]
        ax1.text(tx + dx, sy + dy, SHORT_LABELS[ag], fontsize=8.2, fontweight="bold")
    ax1.set_xlabel("Estimated Prompt + Completion Tokens (Thousands)")
    ax1.set_ylabel("Held-Out Benchmark Success Rate (%)")
    ax1.set_xlim(35, 112)
    ax1.set_ylim(20, 64)
    ax1.set_title("(a) Accuracy vs. Estimated Token Budget")

    # (b) Local CPU Latency per Task
    lat = df["eval_time_ms_per_task_mean"].to_numpy()
    lat_sd = df["eval_time_ms_per_task_sd"].to_numpy()
    ax2.bar(x, lat, width=0.55, color=cols, edgecolor="#0F172A", lw=0.8, yerr=lat_sd, capsize=3)
    for i, v in enumerate(lat):
        ax2.text(i, v + lat_sd[i] + 0.015, f"{v:.3f}", ha="center", fontsize=8.0, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(slabels, rotation=18, ha="right")
    ax2.set_ylabel("Measured Local CPU Time (ms / task)")
    ax2.set_ylim(0, max(lat + lat_sd) * 1.25)
    ax2.set_title("(b) Local CPU Bookkeeping Latency (Xeon 2.60 GHz)")

    # (c) Persistent Memory Footprint (KB)
    mem = df["memory_kb_mean"].to_numpy()
    ax3.bar(x, mem, width=0.55, color=cols, edgecolor="#0F172A", lw=0.8)
    for i, v in enumerate(mem):
        ax3.text(i, v + 2.2, f"{v:.1f} KB", ha="center", fontsize=8.0, fontweight="bold")
    ax3.set_xticks(x)
    ax3.set_xticklabels(slabels, rotation=18, ha="right")
    ax3.set_ylabel("Persistent Memory Footprint (KB)")
    ax3.set_ylim(0, 124)
    ax3.set_title("(c) Active Persistent Memory Consumption")

    # (d) Environment Rollouts Used (480 Total Tasks)
    rolls = df["total_rollouts_mean"].to_numpy()
    ax4.bar(x, rolls, width=0.55, color=cols, edgecolor="#0F172A", lw=0.8)
    for i, v in enumerate(rolls):
        ax4.text(i, v + 5, f"{v:.1f}", ha="center", fontsize=8.0, fontweight="bold")
    ax4.set_xticks(x)
    ax4.set_xticklabels(slabels, rotation=18, ha="right")
    ax4.set_ylabel("Mean Environment Rollouts (480 Episodes)")
    ax4.set_ylim(480, 685)
    ax4.set_title("(d) Cumulative Environment Interaction Rollouts")

    fig.tight_layout()
    save_both(fig, "fig10_exp6_efficiency")


def main() -> None:
    make_fig01_architecture()
    make_fig02_validation_workflow()
    make_fig03_applicability_retrieval()
    make_fig04_drift_detection()
    make_fig05_exp1_generalization()
    make_fig06_exp2_learning()
    make_fig07_exp3_shift()
    make_fig08_exp4_corruption()
    make_fig09_exp5_ablation()
    make_fig10_exp6_efficiency()
    print(f"Generated all 10 figures (.png + .pdf) in {FIG_DIR}")


if __name__ == "__main__":
    main()
