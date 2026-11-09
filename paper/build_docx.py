#!/usr/bin/env python3
"""
Programmatically build paper/research_paper.docx synchronized in structure,
tables, figures, equations, citations, and numbers with paper/research_paper.tex.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches, Pt, RGBColor

ROOT = Path(__file__).resolve().parents[1]
PAPER_DIR = ROOT / "paper"
FIG_DIR = ROOT / "figures"
DIGEST_PATH = ROOT / "experiments" / "statistical_analysis" / "statistical_digest.json"


def clean_latex_text(s: str) -> str:
    """Convert inline LaTeX markup from research_paper.tex into clean readable Unicode prose."""
    # Remove comments
    s = re.sub(r"(?<!\\)%.*", "", s)
    # Formatting macros
    for cmd in ("textbf", "emph", "textit", "texttt", "mathrm", "mathcal"):
        s = re.sub(rf"\\{cmd}\{{([^{{}}]*)\}}", r"\1", s)
    # Simple math/symbol replacements
    repls = {
        r"\%": "%",
        r"\_": "_",
        r"\&": "&",
        r"\$": "$",
        r"\pm": "±",
        r"\le": "≤",
        r"\ge": "≥",
        r"\neq": "≠",
        r"\approx": "≈",
        r"\sim": "~",
        r"\times": "×",
        r"\to": "→",
        r"\leftarrow": "←",
        r"\land": "∧",
        r"\lor": "∨",
        r"\oplus": "⊕",
        r"\in": "∈",
        r"\subset": "⊂",
        r"\subseteq": "⊆",
        r"\emptyset": "∅",
        r"\dots": "...",
        r"\quad": "  ",
        r"\qquad": "    ",
        r"\,": " ",
        r"\;": " ",
        r"\!": "",
        r"\-": "",
        r"\allowbreak": "",
        r"---": "—",
        r"--": "–",
        r"``": "“",
        r"''": "”",
        r"\alpha": "α",
        r"\beta": "β",
        r"\gamma": "γ",
        r"\delta": "δ",
        r"\eta": "η",
        r"\theta": "θ",
        r"\Lambda": "Λ",
        r"\mu": "μ",
        r"\pi": "π",
        r"\tau": "τ",
        r"\phi": "ϕ",
        r"\psi": "ψ",
    }
    for k, v in repls.items():
        s = s.replace(k, v)
    s = re.sub(r"~\\cite\{([^}]*)\}", r" [\1]", s)
    s = re.sub(r"\\cite\{([^}]*)\}", r"[\1]", s)
    s = re.sub(r"~?\\(?:eq)?ref\{([^}]*)\}", r"(\1)", s)
    s = s.replace("$", "")
    s = re.sub(r"\\[a-zA-Z]+\*?", "", s)
    s = s.replace("{", "").replace("}", "")
    s = re.sub(r"[ \t]+", " ", s)
    return s.strip()


def add_styled_table(doc: Document, caption: str, headers: list[str], rows: list[list[str]]) -> None:
    cap = doc.add_paragraph()
    cap.paragraph_format.space_before = Pt(8)
    cap.paragraph_format.space_after = Pt(3)
    rc = cap.add_run(caption)
    rc.bold = True
    rc.font.size = Pt(9.5)

    tbl = doc.add_table(rows=1 + len(rows), cols=len(headers))
    tbl.style = "Table Grid"
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER

    for j, h in enumerate(headers):
        cell = tbl.rows[0].cells[j]
        cell.text = ""
        run = cell.paragraphs[0].add_run(h)
        run.bold = True
        run.font.size = Pt(8.5)

    for i, row in enumerate(rows):
        is_evsa = any("EVSA" in str(x) and ("Proposed" in str(x) or "Full" in str(x) or "direct_macro" in str(x)) for x in row)
        for j, val in enumerate(row):
            cell = tbl.rows[i + 1].cells[j]
            cell.text = ""
            run = cell.paragraphs[0].add_run(str(val))
            run.font.size = Pt(8.5)
            if is_evsa:
                run.bold = True


def add_figure(doc: Document, stem: str, caption: str, width_in: float = 6.3) -> None:
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(8)
    p.add_run().add_picture(str(FIG_DIR / f"{stem}.png"), width=Inches(width_in))

    cp = doc.add_paragraph()
    cp.alignment = WD_ALIGN_PARAGRAPH.LEFT
    cp.paragraph_format.space_after = Pt(8)
    rc = cp.add_run(caption)
    rc.italic = True
    rc.font.size = Pt(9.0)


def main() -> None:
    with open(DIGEST_PATH, encoding="utf-8") as f:
        digest = json.load(f)

    doc = Document()
    for section in doc.sections:
        section.top_margin = Inches(0.85)
        section.bottom_margin = Inches(0.85)
        section.left_margin = Inches(0.85)
        section.right_margin = Inches(0.85)

    normal = doc.styles["Normal"]
    normal.font.name = "Calibri"
    normal.font.size = Pt(10.5)

    # 1. Title & Author Information
    tp = doc.add_paragraph()
    tp.alignment = WD_ALIGN_PARAGRAPH.CENTER
    tr = tp.add_run("Evidence-Validated Strategy Adaptation for Memory-Augmented Agents Under Distribution Shift\n")
    tr.bold = True
    tr.font.size = Pt(16)
    tr.font.color.rgb = RGBColor(15, 23, 42)
    sr = tp.add_run("A Statistical Framework for Reliable Experience Validation, Applicability-Gated Retrieval, and Selective Memory Invalidation")
    sr.font.size = Pt(11.5)
    sr.font.color.rgb = RGBColor(51, 65, 85)

    ap = doc.add_paragraph()
    ap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r_auth = ap.add_run("Dev Parth\n")
    r_auth.bold = True
    r_auth.font.size = Pt(12)
    r_aff = ap.add_run(
        "Independent Researcher\n"
        "Independent Research in Artificial Intelligence\n"
        "Release Date: 9 November 2026"
    )
    r_aff.font.size = Pt(10)

    # Parse sections and prose from research_paper.tex for strict prose parity
    tex_src = (PAPER_DIR / "research_paper.tex").read_text(encoding="utf-8")

    # 2. Abstract & 3. Keywords
    abs_match = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", tex_src, re.S)
    doc.add_heading("Abstract", level=1)
    if abs_match:
        for para in abs_match.group(1).strip().split("\n\n"):
            doc.add_paragraph(clean_latex_text(para))

    doc.add_heading("Keywords", level=2)
    doc.add_paragraph(
        "Memory-augmented agents; Sequential hypothesis testing; Bayesian credible bounds; "
        "Applicability-gated retrieval; Concept drift detection; Selective memory invalidation; Continual adaptation."
    )

    # Helper to extract section blocks from TeX
    sec_splits = re.split(r"\\section\{([^}]*)\}", tex_src)[1:]
    sections_dict = {}
    for i in range(0, len(sec_splits), 2):
        title = clean_latex_text(sec_splits[i])
        body = sec_splits[i + 1]
        if "\\end{document}" in body:
            body = body.split("\\end{document}")[0]
        if "\\bibliographystyle" in body:
            body = body.split("\\bibliographystyle")[0]
        sections_dict[title] = body

    def emit_clean_body(body_tex: str) -> None:
        # Remove float environments (we insert native tables and high-DPI PNG figures explicitly)
        body_tex = re.sub(r"\\begin\{table\*?\}.*?\\end\{table\*?\}", "", body_tex, flags=re.S)
        body_tex = re.sub(r"\\begin\{figure\*?\}.*?\\end\{figure\*?\}", "", body_tex, flags=re.S)
        body_tex = re.sub(r"\\begin\{algorithm\}.*?\\end\{algorithm\}", "", body_tex, flags=re.S)

        blocks = re.split(r"\n\s*\n", body_tex.strip())
        for blk in blocks:
            blk = blk.strip()
            if not blk:
                continue
            sub_m = re.match(r"\\subsection\{([^}]*)\}(.*)", blk, flags=re.S)
            if sub_m:
                doc.add_heading(clean_latex_text(sub_m.group(1)), level=2)
                blk = sub_m.group(2).strip()
                if not blk:
                    continue
            para_m = re.match(r"\\paragraph\{([^}]*)\}(.*)", blk, flags=re.S)
            if para_m:
                p = doc.add_paragraph()
                r1 = p.add_run(clean_latex_text(para_m.group(1)) + " ")
                r1.bold = True
                p.add_run(clean_latex_text(para_m.group(2)))
                continue
            if "\\begin{itemize}" in blk or "\\begin{enumerate}" in blk:
                items = re.split(r"\\item\s+", blk)[1:]
                for it in items:
                    it_clean = re.sub(r"\\end\{(?:itemize|enumerate)\}", "", it)
                    doc.add_paragraph("• " + clean_latex_text(it_clean))
                continue
            cleaned = clean_latex_text(blk)
            if cleaned:
                doc.add_paragraph(cleaned)

    for title, body in sections_dict.items():
        doc.add_heading(title, level=1)
        emit_clean_body(body)

        if "Related Work" in title:
            add_styled_table(
                doc,
                "Table 1: Architectural comparison of representative agentic memory and continual adaptation frameworks.",
                ["Framework", "Cross-Task Memory", "Multi-Trial Gate", "Reliability Est.", "Preconditions", "Drift Detector", "Quarantine"],
                [
                    ["ReAct (2023)", "No", "No", "No", "No", "No", "No"],
                    ["Reflexion / CLIN", "Verbal / Abstractions", "No (1-episode)", "Verbalized only", "Implicit text", "No", "Buffer cap"],
                    ["Voyager (2024)", "Code Skills", "1-trial env check", "Binary pass/fail", "Implicit text", "No", "No"],
                    ["ExpeL / AWM", "Insights / Workflows", "Heuristic / LLM", "Empirical freq.", "Prompt sim.", "No", "Rule edit"],
                    ["CER / ReasoningBank (ICLR 2026)", "Trajectories / Strategies", "No (self-judged)", "Self-judged tag", "Dense embed.", "No", "Top-k dilution"],
                    ["Dynamic Cheatsheet (EACL 2026) / A-MEM", "Curated Notes", "LLM-judged edit", "Uncalibrated LLM", "Linked keywords", "No", "Prompt rewrite"],
                    ["AdaMEM (ICML 2026)", "Hybrid LT + ST Bank", "LLM synthesis / MFT", "Uncalibrated LLM", "Query / state context", "Short-term refresh", "Summary overwrite"],
                    ["EVSA (Proposed)", "3-Tier Strategies", "Discounted SPRT", "Beta Credible Bound", "Conjunctive + Safe LGG", "Per-Domain CUSUM", "Evidence Quarantine"],
                ],
            )

        elif "Proposed EVSA Architecture" in title:
            add_figure(doc, "fig01_system_architecture", "Figure 1: EVSA Ten-Module Closed-Loop System Architecture.")
            add_figure(doc, "fig02_validation_workflow", "Figure 2: Strategy Validation Lifecycle and Sequential Evidence Trajectories.")

        elif "Mathematical Formulation" in title:
            add_figure(doc, "fig03_applicability_retrieval", "Figure 3: Applicability-Gated Retrieval and Execution-Channel Fairness Parity.")
            add_figure(doc, "fig04_drift_detection", "Figure 4: Empirical Per-Domain Residual CUSUM Monitoring Across Multi-Regime Shifts (N = 20 Seeds).")

        elif "Experimental Results" in title:
            add_styled_table(
                doc,
                "Table 2: Experiment 1 — Held-out unseen benchmark generalization (N = 20 seeds, 240 tasks/seed).",
                ["Agent", "Overall SR (%)", "Exact Plan (%)", "Unseen Inst. (%)", "Zero-Shot c=7 (%)", "Avg. Reward"],
                [
                    [r["agent"], f"{r['task_success_rate_mean']*100:.2f} ± {r['task_success_rate_sd']*100:.2f}",
                     f"{r['exact_plan_accuracy_mean']*100:.2f}", f"{r['unseen_instance_sr_mean']*100:.2f}",
                     f"{r['unseen_compositional_sr_mean']*100:.2f}", f"{r['average_reward_mean']:.3f}"]
                    for r in digest["exp1"]
                ],
            )
            add_figure(doc, "fig05_exp1_generalization", "Figure 5: Experiment 1 Unseen Generalization Across Splits (N = 20 Seeds).")

            add_styled_table(
                doc,
                "Table 3: Execution-Channel Fairness Parity Control on Experiment 1 (N = 20 seeds).",
                ["Controlled Variant", "Overall SR (%)", "Unseen Inst. (%)", "Zero-Shot Comp. (%)"],
                [
                    [r["agent"], f"{r['overall_sr_mean']*100:.2f} ± {r['overall_sr_sd']*100:.2f}",
                     f"{r['unseen_instance_sr_mean']*100:.2f}", f"{r['unseen_compositional_sr_mean']*100:.2f}"]
                    for r in digest["fairness_parity"]
                ],
            )

            add_styled_table(
                doc,
                "Table 4: Experiment 2 — Continual learning trajectory across 320 episodes (N = 20 seeds).",
                ["Agent", "Init SR (%)", "Block 1 SR (%)", "Final Block 8 (%)", "Abs. Gain (pp)", "Retained", "Rejected"],
                [
                    [r["agent"], f"{r['initial_sr_mean']*100:.2f}", f"{r['block1_sr_mean']*100:.2f}",
                     f"{r['final_sr_mean']*100:.2f} ± {r['final_sr_sd']*100:.2f}",
                     f"{r['total_improvement_abs_mean']*100:+.2f}",
                     f"{r['strategies_retained_mean']:.2f}", f"{r['strategies_rejected_mean']:.2f}"]
                    for r in digest["exp2"]
                ],
            )
            add_figure(doc, "fig06_exp2_continual_learning", "Figure 6: Experiment 2 Continual Learning and Strategy Consolidation (N = 20 Seeds).")

            add_styled_table(
                doc,
                "Table 5: Experiment 3 — Multi-regime distribution shift adaptation and selective retention (N = 20 seeds).",
                ["Agent", "Pre-Shift (%)", "Shock (%)", "Recovered (%)", "Adapt. Gain (pp)", "Unshifted Dom. (%)", "Phase III Unseen (%)"],
                [
                    [r["agent"], f"{r['pre_shift_sr_mean']*100:.2f}", f"{r['immediate_post_shift_sr_mean']*100:.2f}",
                     f"{r['recovered_post_shift_sr_mean']*100:.2f} ± {r['recovered_post_shift_sr_sd']*100:.2f}",
                     f"{r['adaptation_gain_mean']*100:+.2f}", f"{r['unshifted_domain_sr_mean']*100:.2f}",
                     f"{r['unseen_tasks_post_shift_sr_mean']*100:.2f}"]
                    for r in digest["exp3"]
                ],
            )
            add_figure(doc, "fig07_exp3_shift_adaptation", "Figure 7: Experiment 3 Multi-Regime Shift Trajectory and Selective Retention (N = 20 Seeds).")

            add_styled_table(
                doc,
                "Table 6: Experiment 4 — Misleading feedback calibration, store decomposition, and clean recovery (N = 20 seeds).",
                ["Noise η", "Agent", "Rollout FAR (%)", "Cand. FPR (%)", "FRR (%)", "Noisy FDP (%)", "Post-Clean FDP (%)", "Clean Rec. (%)"],
                [
                    [f"{r['corruption_rate_eta']:.2f}", r["agent"], f"{r['false_acceptance_rate_mean']*100:.2f}",
                     f"{r['candidate_false_promotion_rate_mean']*100:.2f}", f"{r['false_rejection_rate_mean']*100:.2f}",
                     f"{r['memory_contamination_rate_mean']*100:.2f}", f"{r['post_clean_contamination_rate_mean']*100:.2f}",
                     f"{r['recovery_after_misleading_sr_mean']*100:.2f} ± {r['recovery_after_misleading_sr_sd']*100:.2f}"]
                    for r in digest["exp4"] if "Static" not in r["agent"]
                ],
            )
            add_figure(doc, "fig08_exp4_feedback_corruption", "Figure 8: Experiment 4 Feedback Corruption Calibration and Store Decomposition (N = 20 Seeds).")

        elif "Statistical Analysis" in title:
            import pandas as pd
            df_t = pd.read_csv(ROOT / "experiments" / "statistical_analysis" / "paired_hypothesis_tests_holm.csv")
            pick_idx = [1, 11, 8, 72, 73, 13, 25, 29, 52, 50, 55, 59, 65, 60, 57, 58]
            sub_t = df_t.iloc[pick_idx]
            add_styled_table(
                doc,
                "Table 7: Selected paired statistical tests (N = 20 seeds) with direct per-family primary Holm, direct Global-74 primary Holm, and labelled pure-test Global-74 sensitivity p-values.",
                ["Family", "Comparator", "Primary Test", "Cohen's dz", "Per-Fam p_Holm", "Glob-74 p_Holm", "Sens. t_Glob74", "Sens. W_Glob74", "Decision (Fam / Glob74)"],
                [
                    [
                        str(r["family"]),
                        str(r["comparator"]),
                        str(r["primary_test"]),
                        f"{r['cohens_dz']:+.2f}",
                        f"{r['primary_pvalue_holm']:.3e}",
                        f"{r['primary_pvalue_holm_global74']:.3e}",
                        f"{r['sensitivity_t_pvalue_holm_global74']:.3e}",
                        f"{r['sensitivity_wilcoxon_pvalue_holm_global74']:.3e}",
                        ("Sig. / Sig." if r["significant_global74_at_05"] else "Sig. / Non-Sig.")
                        if r["significant_at_05"]
                        else "Non-Sig. / Non-Sig.",
                    ]
                    for _, r in sub_t.iterrows()
                ],
            )

        elif "Ablation Studies" in title:
            add_styled_table(
                doc,
                "Table 8: Experiment 5 — Leave-one-out component ablation study (N = 20 seeds).",
                ["Variant", "Overall Stream (%)", "Post-Shift Rec. (%)", "Held-Out Unseen (%)", "Pre-Shift FDP (%)", "Post-Shift FDP (%)"],
                [
                    [r["variant"], f"{r['overall_stream_sr_mean']*100:.2f} ± {r['overall_stream_sr_sd']*100:.2f}",
                     f"{r['post_shift_recovered_sr_mean']*100:.2f} ± {r['post_shift_recovered_sr_sd']*100:.2f}",
                     f"{r['heldout_unseen_sr_mean']*100:.2f} ± {r['heldout_unseen_sr_sd']*100:.2f}",
                     f"{r['pre_shift_contamination_mean']*100:.2f}", f"{r['post_shift_contamination_mean']*100:.2f}"]
                    for r in digest["exp5"]
                ],
            )
            add_figure(doc, "fig09_exp5_ablation", "Figure 9: Experiment 5 Leave-One-Out Component Ablation (N = 20 Seeds).")

        elif "Computational Efficiency" in title:
            add_styled_table(
                doc,
                "Table 9: Experiment 6 — Directly measured local CPU/memory efficiency vs. projected token economics (N = 20 seeds).",
                ["Agent", "Eval SR (%)", "CPU (ms/task)", "Rollouts", "Memory (KB)", "Est. Tokens*", "Est. USD*", "SR/10k Tok*"],
                [
                    [r["agent"], f"{r['task_success_rate_mean']*100:.2f}",
                     f"{r['eval_time_ms_per_task_mean']:.3f} ± {r['eval_time_ms_per_task_sd']:.3f}",
                     f"{r['total_rollouts_mean']:.1f}", f"{r['memory_kb_mean']:.2f}",
                     f"{r['total_tokens_mean']:.0f}", f"${r['estimated_api_cost_usd_mean']:.3f}",
                     f"{r['sr_per_10k_tokens_mean']:.3f}"]
                    for r in digest["exp6"]
                ],
            )
            add_figure(doc, "fig10_exp6_efficiency", "Figure 10: Experiment 6 Computational Efficiency and Resource Footprint (N = 20 Seeds).")

    # 20. References (parsed directly from verified references.bib)
    doc.add_heading("References", level=1)
    bib_raw = (PAPER_DIR / "references.bib").read_text(encoding="utf-8")
    entries = re.findall(r"@\w+\{([^,]+),(.*?)\n\}", bib_raw, flags=re.S)
    for idx, (key, fields_txt) in enumerate(entries, start=1):
        fdict = {m.group(1).lower(): clean_latex_text(m.group(2))
                 for m in re.finditer(r"(\w+)\s*=\s*\{(.*?)\}", fields_txt, flags=re.S)}
        venue = fdict.get("booktitle") or fdict.get("journal") or ""
        yr = fdict.get("year", "")
        doi_url = fdict.get("doi") or fdict.get("url") or ""
        cite_line = f"[{idx}] ({key}) {fdict.get('author', '')} ({yr}). “{fdict.get('title', '')}.” {venue}. {doi_url}".strip()
        p = doc.add_paragraph(cite_line)
        p.paragraph_format.space_after = Pt(2)

    out_docx = PAPER_DIR / "research_paper.docx"
    doc.save(out_docx)
    print(f"Saved synchronized DOCX manuscript to {out_docx} ({out_docx.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
