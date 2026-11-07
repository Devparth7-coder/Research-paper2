"""
Calibrated Symbolic-Procedural Policy Backbone and Estimated Prompt/Token Meter.

Strict Non-Oracle Prior Design:
`CalibratedSymbolicLLMBackbone` computes zero-shot step distributions strictly
from observable `task.domain` and `task.structural_predicates` using first-order
additive heuristics (e.g., inclusive-OR `p0 | p1` on Step 3 rather than the
environment's true latent parity `b0 ^ b1`). It has zero access to latent
archetype IDs, ground-truth optimal plans, or hidden regime shifts.
"""

from __future__ import annotations

import json
import urllib.request
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np

from .environment import ACTION_DIMS, DOMAINS
from .types import TaskInstance

# Projected reference API pricing tier (USD per 1M tokens; strictly an ESTIMATE)
ESTIMATED_INPUT_COST_PER_1M_USD: float = 2.50
ESTIMATED_OUTPUT_COST_PER_1M_USD: float = 10.00


@dataclass
class TokenUsageLedger:
    """Tracks estimated serialized prompt and completion token counts across policy calls."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    llm_calls: int = 0

    def record_call(self, prompt_text: str, completion_text: str) -> Tuple[int, int]:
        p_tok = max(12, int(round(len(prompt_text) / 3.8)) + 8)
        c_tok = max(6, int(round(len(completion_text) / 3.8)) + 4)
        self.prompt_tokens += p_tok
        self.completion_tokens += c_tok
        self.llm_calls += 1
        return p_tok, c_tok

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    @property
    def estimated_cost_usd(self) -> float:
        return (
            (self.prompt_tokens / 1_000_000.0) * ESTIMATED_INPUT_COST_PER_1M_USD
            + (self.completion_tokens / 1_000_000.0) * ESTIMATED_OUTPUT_COST_PER_1M_USD
        )


class CalibratedSymbolicLLMBackbone:
    """
    Symbolic-procedural policy backbone with observable first-order additive
    step priors (`base_step_competence = 0.67`) and weighted experience-following
    adherence. Does NOT know the latent XOR parity rule on `(p0=1, p1=1)`.
    """

    def __init__(self, seed: int = 42, base_step_competence: float = 0.67) -> None:
        self.rng = np.random.default_rng(seed)
        self.base_step_competence = base_step_competence
        self.ledger = TokenUsageLedger()

    def _base_step_probs(self, task: TaskInstance, step_idx: int) -> np.ndarray:
        """Compute zero-shot operator prior strictly from observable task features."""
        n_act = ACTION_DIMS[step_idx]
        d_idx = DOMAINS.index(task.domain)
        preds = task.structural_predicates
        p0 = int(preds[0])
        p1 = int(preds[1])
        p2 = int(preds[2])

        if step_idx == 0:
            prior_op = int((d_idx + p0 * 2) % n_act)
        elif step_idx == 1:
            prior_op = int((d_idx * 2 + p1 * 3 + 1) % n_act)
        elif step_idx == 2:
            prior_op = int((d_idx + p2 * 2 + 1) % n_act)
        else:
            # First-order inclusive-OR heuristic (fails on non-linear XOR when p0=1, p1=1)
            p_linear = 1 if (p0 == 1 or p1 == 1) else 0
            prior_op = int((d_idx + p_linear) % n_act)

        probs = np.full(n_act, (1.0 - self.base_step_competence) / max(1, n_act - 1), dtype=np.float64)
        probs[prior_op] = self.base_step_competence
        return probs

    def generate_plan(
        self,
        task: TaskInstance,
        system_role: str = "react_static",
        recommended_plans: Optional[Sequence[Tuple[Tuple[int, ...], float]]] = None,
        avoid_operators: Optional[Dict[int, Sequence[int]]] = None,
        extra_context_text: str = "",
    ) -> Tuple[Tuple[int, ...], int, int]:
        prompt_parts = [
            f"[ROLE:{system_role}] Domain={task.domain} Predicates={task.structural_predicates} "
            f"Topic={task.surface_topic_id} Features={np.round(task.context_features[:6], 2).tolist()}"
        ]
        if recommended_plans:
            prompt_parts.append(
                "RetrievedGuidance="
                + json.dumps([(list(p), round(w, 3)) for p, w in recommended_plans])
            )
        if avoid_operators:
            prompt_parts.append("AvoidConstraints=" + json.dumps({k: list(v) for k, v in avoid_operators.items()}))
        if extra_context_text:
            prompt_parts.append(extra_context_text)
        prompt_str = "\n".join(prompt_parts)

        chosen: List[int] = []
        for s_idx in range(len(ACTION_DIMS)):
            probs = self._base_step_probs(task, s_idx).copy()

            if recommended_plans:
                rec_dist = np.zeros(ACTION_DIMS[s_idx], dtype=np.float64)
                total_w = 0.0
                for plan, weight in recommended_plans:
                    w = max(0.01, float(weight))
                    rec_dist[int(plan[s_idx])] += w
                    total_w += w
                if total_w > 0:
                    rec_dist /= total_w
                    adherence = float(np.clip(0.55 + 0.39 * min(1.0, total_w), 0.55, 0.96))
                    probs = (1.0 - adherence) * probs + adherence * rec_dist

            if avoid_operators and s_idx in avoid_operators:
                for bad_op in avoid_operators[s_idx]:
                    if 0 <= int(bad_op) < ACTION_DIMS[s_idx]:
                        probs[int(bad_op)] *= 0.03
                if probs.sum() <= 1e-12:
                    probs = np.ones(ACTION_DIMS[s_idx], dtype=np.float64)

            probs /= probs.sum()
            op = int(self.rng.choice(ACTION_DIMS[s_idx], p=probs))
            chosen.append(op)

        plan_tuple = tuple(chosen)
        completion_str = f"Reasoning: analyzing {task.domain} constraints -> ActionPlan={list(plan_tuple)}"
        p_tok, c_tok = self.ledger.record_call(prompt_str, completion_str)
        return plan_tuple, p_tok, c_tok

    def account_auxiliary_call(self, purpose: str, input_payload: str, output_payload: str) -> Tuple[int, int]:
        return self.ledger.record_call(f"[{purpose}] {input_payload}", output_payload)


class LiveOpenAICompatibleBackbone:
    """Optional HTTP adapter for future live LLM API experiments (not used in local benchmark runs)."""

    def __init__(self, api_key: str, model: str = "gpt-4o-mini", base_url: str = "https://api.openai.com/v1") -> None:
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.ledger = TokenUsageLedger()

    def query_json_plan(self, prompt: str) -> Tuple[List[int], int, int]:
        payload = json.dumps({
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2,
        }).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=payload,
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"},
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        content = body["choices"][0]["message"]["content"]
        usage = body.get("usage", {})
        p_tok = int(usage.get("prompt_tokens", len(prompt) // 4))
        c_tok = int(usage.get("completion_tokens", len(content) // 4))
        self.ledger.prompt_tokens += p_tok
        self.ledger.completion_tokens += c_tok
        self.ledger.llm_calls += 1
        parsed = json.loads(content)
        return [int(x) for x in parsed["plan"]], p_tok, c_tok
