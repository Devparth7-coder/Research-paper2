"""
Symbolic-Procedural Baseline Agent Implementations (Baselines A--D).

Important methodological qualification: These implementations are controlled
benchmark-specific adaptations of published language-agent paradigms to the
symbolic-procedural action space of ShiftBench-Sim—not exact reproductions of
free-form natural-language systems:
- Baseline A (StaticAgent): Stateless prompting adaptation of ReAct (Yao et al., 2023).
- Baseline B (ConventionalMemoryAgent): Similarity + recency + reward episodic retrieval
  adapted from Generative Agents (Park et al., 2023).
- Baseline C (ReflectionAgent): Sliding-window failure-avoidance & critique adaptation
  of Reflexion (Shinn et al., 2023) generalized across cross-instance domain streams.
- Baseline D (ExperienceReplayAgent): Distilled y_tilde=1 workflow replay buffer adapted
  from CER (Liu et al., 2025) and AWM (Wang et al., 2025).

Strict Zero-Oracle Guarantee:
No baseline class inspects ground-truth plans, latent archetypes, hidden regimes,
or uncorrupted rollout outcomes.
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np

from ..benchmark.agent_interface import AgentEnvironmentInterface
from ..benchmark.llm_backbone import CalibratedSymbolicLLMBackbone
from ..benchmark.types import AgentEpisodeOutcome, MemoryEntry, RolloutResult, TaskInstance


def _require_agent_interface(env: AgentEnvironmentInterface) -> None:
    if not isinstance(env, AgentEnvironmentInterface):
        raise TypeError(
            "Hard oracle boundary violation: agents accept strictly `AgentEnvironmentInterface`, "
            f"never raw evaluator instances (got {type(env).__name__})."
        )


def _estimate_entries_bytes(entries: Sequence[MemoryEntry]) -> int:
    if not entries:
        return 0
    per_item = 20 * 8 + 64 + 48
    return len(entries) * per_item


class BaseExperimentalAgent:
    """Common interface shared across all baseline and EVSA agents."""

    name: str = "BaseAgent"

    def __init__(self, seed: int = 42) -> None:
        self.seed = seed
        self.backbone = CalibratedSymbolicLLMBackbone(seed=seed)

    def act_on_episode(
        self,
        task: TaskInstance,
        env: AgentEnvironmentInterface,
        max_trials: int = 2,
        learn: bool = True,
    ) -> AgentEpisodeOutcome:
        raise NotImplementedError

    def active_memory_count(self) -> int:
        return 0

    def memory_footprint_bytes(self) -> int:
        return 0


class StaticAgent(BaseExperimentalAgent):
    """Baseline A: Memoryless ReAct-inspired static policy baseline."""

    name = "Baseline A: Static"

    def act_on_episode(
        self,
        task: TaskInstance,
        env: AgentEnvironmentInterface,
        max_trials: int = 2,
        learn: bool = True,
    ) -> AgentEpisodeOutcome:
        _require_agent_interface(env)
        _ = learn
        t0 = time.perf_counter()
        p_tok_ep, c_tok_ep = 0, 0
        trials_used = 0
        best_res: Optional[RolloutResult] = None
        best_trial_idx = 0

        while trials_used < max_trials:
            trial_zero_idx = trials_used
            trials_used += 1
            plan, p_tok, c_tok = self.backbone.generate_plan(task, system_role="react_static")
            p_tok_ep += p_tok
            c_tok_ep += c_tok
            res = env.evaluate_plan(task, plan)
            if best_res is None or res.observed_reward > best_res.observed_reward:
                best_res = res
                best_trial_idx = trial_zero_idx
            if res.observed_success == 1:
                best_res = res
                best_trial_idx = trial_zero_idx
                break

        dt_ms = (time.perf_counter() - t0) * 1000.0
        assert best_res is not None
        return AgentEpisodeOutcome(
            best_trial_idx=best_trial_idx,
            trials_used=trials_used,
            strategy_evaluations=0,
            execution_time_ms=dt_ms,
            active_memory_items=0,
            memory_bytes=0,
            prompt_tokens=p_tok_ep,
            completion_tokens=c_tok_ep,
            shift_detected=False,
        )


class ConventionalMemoryAgent(BaseExperimentalAgent):
    """Baseline B: Episodic buffer scored by similarity, recency decay, and reward."""

    name = "Baseline B: Conventional Memory"

    def __init__(
        self,
        seed: int = 42,
        max_entries: int = 400,
        top_k: int = 3,
        recency_gamma: float = 0.993,
    ) -> None:
        super().__init__(seed=seed)
        self.max_entries = max_entries
        self.top_k = top_k
        self.recency_gamma = recency_gamma
        self.memory: List[MemoryEntry] = []
        self._counter = 0

    def _retrieve(self, task: TaskInstance) -> Tuple[List[MemoryEntry], int]:
        candidates = [m for m in self.memory if m.domain == task.domain]
        if not candidates:
            return [], 0
        scored: List[Tuple[float, MemoryEntry]] = []
        for m in candidates:
            sim = float(np.dot(m.query_embedding, task.query_embedding))
            age = max(0, task.episode_idx - m.episode_idx)
            rec = float(self.recency_gamma ** min(age, 500))
            score = 0.55 * sim + 0.20 * rec + 0.25 * m.observed_reward
            scored.append((score, m))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [m for _, m in scored[: self.top_k]], len(candidates)

    def act_on_episode(
        self,
        task: TaskInstance,
        env: AgentEnvironmentInterface,
        max_trials: int = 2,
        learn: bool = True,
    ) -> AgentEpisodeOutcome:
        _require_agent_interface(env)
        t0 = time.perf_counter()
        p_tok_ep, c_tok_ep = 0, 0
        retrieved, eval_count = self._retrieve(task)

        rec_plans: List[Tuple[Tuple[int, ...], float]] = []
        for m in retrieved:
            if m.observed_success == 1 or m.observed_reward >= 0.65:
                sim = max(0.05, float(np.dot(m.query_embedding, task.query_embedding)))
                rec_plans.append((m.executed_plan, sim * m.observed_reward))

        trials_used = 0
        best_res: Optional[RolloutResult] = None
        best_trial_idx = 0
        local_avoid: Dict[int, List[int]] = {}

        while trials_used < max_trials:
            trial_zero_idx = trials_used
            trials_used += 1
            plan, p_tok, c_tok = self.backbone.generate_plan(
                task,
                system_role="conventional_memory",
                recommended_plans=rec_plans if trials_used == 1 else None,
                avoid_operators=local_avoid if local_avoid else None,
            )
            p_tok_ep += p_tok
            c_tok_ep += c_tok
            res = env.evaluate_plan(task, plan)

            if learn:
                self._counter += 1
                self.memory.append(
                    MemoryEntry(
                        entry_id=f"mem_{self._counter}",
                        task_id=task.task_id,
                        domain=task.domain,
                        episode_idx=task.episode_idx,
                        query_embedding=task.query_embedding.copy(),
                        predicates=task.structural_predicates,
                        executed_plan=res.executed_plan,
                        observed_success=res.observed_success,
                        observed_reward=res.observed_reward,
                        first_failed_step=res.first_failed_step,
                        failed_operator=res.failed_operator,
                    )
                )
                if len(self.memory) > self.max_entries:
                    self.memory = self.memory[-self.max_entries :]

            if best_res is None or res.observed_reward > best_res.observed_reward:
                best_res = res
                best_trial_idx = trial_zero_idx
            if res.observed_success == 1:
                best_res = res
                best_trial_idx = trial_zero_idx
                break
            if res.first_failed_step is not None and res.failed_operator is not None:
                local_avoid.setdefault(res.first_failed_step, []).append(res.failed_operator)

        dt_ms = (time.perf_counter() - t0) * 1000.0
        assert best_res is not None
        return AgentEpisodeOutcome(
            best_trial_idx=best_trial_idx,
            trials_used=trials_used,
            strategy_evaluations=eval_count,
            execution_time_ms=dt_ms,
            active_memory_items=len(self.memory),
            memory_bytes=_estimate_entries_bytes(self.memory),
            prompt_tokens=p_tok_ep,
            completion_tokens=c_tok_ep,
            shift_detected=False,
        )

    def active_memory_count(self) -> int:
        return len(self.memory)

    def memory_footprint_bytes(self) -> int:
        return _estimate_entries_bytes(self.memory)


class ReflectionAgent(BaseExperimentalAgent):
    """Baseline C: Sliding-window failure critique & avoidance adaptation of Reflexion."""

    name = "Baseline C: Reflection"

    def __init__(self, seed: int = 42, window_per_domain: int = 10, top_k: int = 3) -> None:
        super().__init__(seed=seed)
        self.window_per_domain = window_per_domain
        self.top_k = top_k
        self.domain_buffers: Dict[str, List[MemoryEntry]] = {}
        self._counter = 0
        self.accepted_count = 0
        self.accepted_history: List[Tuple[str, Tuple[int, ...]]] = []

    def _all_entries(self) -> List[MemoryEntry]:
        out: List[MemoryEntry] = []
        for buf in self.domain_buffers.values():
            out.extend(buf)
        return out

    def act_on_episode(
        self,
        task: TaskInstance,
        env: AgentEnvironmentInterface,
        max_trials: int = 2,
        learn: bool = True,
    ) -> AgentEpisodeOutcome:
        _require_agent_interface(env)
        t0 = time.perf_counter()
        p_tok_ep, c_tok_ep = 0, 0
        buf = self.domain_buffers.get(task.domain, [])
        eval_count = len(buf)

        scored = sorted(
            [(float(np.dot(m.query_embedding, task.query_embedding)), m) for m in buf],
            key=lambda x: x[0],
            reverse=True,
        )
        top_refs = [m for sim, m in scored[: self.top_k] if sim > 0.35]

        avoid_map: Dict[int, List[int]] = {}
        rec_plans: List[Tuple[Tuple[int, ...], float]] = []
        reflection_notes: List[str] = []

        for m in top_refs:
            for step_i, ops in m.reflection_avoid_map.items():
                for op in ops:
                    if op not in avoid_map.setdefault(step_i, []):
                        avoid_map[step_i].append(op)
            if m.observed_success == 1:
                sim = max(0.1, float(np.dot(m.query_embedding, task.query_embedding)))
                rec_plans.append((m.executed_plan, sim * 0.75))
            reflection_notes.append(f"Ref({m.entry_id}):avoid={m.reflection_avoid_map}")

        trials_used = 0
        best_res: Optional[RolloutResult] = None
        best_trial_idx = 0

        while trials_used < max_trials:
            trial_zero_idx = trials_used
            trials_used += 1
            plan, p_tok, c_tok = self.backbone.generate_plan(
                task,
                system_role="reflexion_actor",
                recommended_plans=rec_plans if rec_plans else None,
                avoid_operators=avoid_map if avoid_map else None,
                extra_context_text="; ".join(reflection_notes),
            )
            p_tok_ep += p_tok
            c_tok_ep += c_tok
            res = env.evaluate_plan(task, plan)

            if best_res is None or res.observed_reward > best_res.observed_reward:
                best_res = res
                best_trial_idx = trial_zero_idx

            if res.observed_success == 1:
                best_res = res
                best_trial_idx = trial_zero_idx
                if learn:
                    self._counter += 1
                    entry = MemoryEntry(
                        entry_id=f"ref_ok_{self._counter}",
                        task_id=task.task_id,
                        domain=task.domain,
                        episode_idx=task.episode_idx,
                        query_embedding=task.query_embedding.copy(),
                        predicates=task.structural_predicates,
                        executed_plan=res.executed_plan,
                        observed_success=1,
                        observed_reward=res.observed_reward,
                        reflection_avoid_map={k: list(v) for k, v in avoid_map.items()},
                    )
                    dbuf = self.domain_buffers.setdefault(task.domain, [])
                    dbuf.append(entry)
                    if len(dbuf) > self.window_per_domain:
                        self.domain_buffers[task.domain] = dbuf[-self.window_per_domain :]
                    self.accepted_count += 1
                    self.accepted_history.append((task.task_id, res.executed_plan))
                break
            else:
                critique_in = (
                    f"Task={task.domain} Pred={task.structural_predicates} "
                    f"Plan={res.executed_plan} FailStep={res.first_failed_step} Op={res.failed_operator}"
                )
                critique_out = (
                    f"Reflection: operator {res.failed_operator} at step {res.first_failed_step} "
                    f"caused failure; avoid {res.failed_operator} and test alternative."
                )
                rp, rc = self.backbone.account_auxiliary_call("self_reflection", critique_in, critique_out)
                p_tok_ep += rp
                c_tok_ep += rc

                if res.first_failed_step is not None and res.failed_operator is not None:
                    avoid_map.setdefault(res.first_failed_step, []).append(res.failed_operator)
                    reflection_notes.append(critique_out)

                if learn:
                    self._counter += 1
                    ref_entry = MemoryEntry(
                        entry_id=f"ref_fail_{self._counter}",
                        task_id=task.task_id,
                        domain=task.domain,
                        episode_idx=task.episode_idx,
                        query_embedding=task.query_embedding.copy(),
                        predicates=task.structural_predicates,
                        executed_plan=res.executed_plan,
                        observed_success=0,
                        observed_reward=res.observed_reward,
                        first_failed_step=res.first_failed_step,
                        failed_operator=res.failed_operator,
                        reflection_avoid_map={k: list(v) for k, v in avoid_map.items()},
                    )
                    dbuf = self.domain_buffers.setdefault(task.domain, [])
                    dbuf.append(ref_entry)
                    if len(dbuf) > self.window_per_domain:
                        self.domain_buffers[task.domain] = dbuf[-self.window_per_domain :]

        all_mem = self._all_entries()
        dt_ms = (time.perf_counter() - t0) * 1000.0
        assert best_res is not None
        return AgentEpisodeOutcome(
            best_trial_idx=best_trial_idx,
            trials_used=trials_used,
            strategy_evaluations=eval_count,
            execution_time_ms=dt_ms,
            active_memory_items=len(all_mem),
            memory_bytes=_estimate_entries_bytes(all_mem),
            prompt_tokens=p_tok_ep,
            completion_tokens=c_tok_ep,
            shift_detected=False,
        )

    def active_memory_count(self) -> int:
        return len(self._all_entries())

    def memory_footprint_bytes(self) -> int:
        return _estimate_entries_bytes(self._all_entries())


class ExperienceReplayAgent(BaseExperimentalAgent):
    """Baseline D: Distilled successful-workflow replay buffer adapted from CER / AWM."""

    name = "Baseline D: Experience Replay"

    def __init__(
        self,
        seed: int = 42,
        max_buffer: int = 200,
        top_k: int = 3,
        direct_macro_execution: bool = False,
        custom_name: Optional[str] = None,
    ) -> None:
        super().__init__(seed=seed)
        if custom_name is not None:
            self.name = custom_name
        self.max_buffer = max_buffer
        self.top_k = top_k
        self.direct_macro_execution = direct_macro_execution
        self.replay_buffer: List[MemoryEntry] = []
        self._counter = 0
        self.retained_strategies = 0
        self.rejected_strategies = 0
        self.admitted_history: List[Tuple[str, Tuple[int, ...]]] = []

    def _retrieve_replay(self, task: TaskInstance) -> Tuple[List[Tuple[Tuple[int, ...], float]], int]:
        cands = [m for m in self.replay_buffer if m.domain == task.domain]
        if not cands:
            return [], 0
        scored: List[Tuple[float, MemoryEntry]] = []
        for m in cands:
            sim = float(np.dot(m.query_embedding, task.query_embedding))
            scored.append((sim, m))
        scored.sort(key=lambda x: x[0], reverse=True)
        recs: List[Tuple[Tuple[int, ...], float]] = []
        for sim, m in scored[: self.top_k]:
            if sim > 0.25:
                recs.append((m.executed_plan, sim * m.observed_reward))
        return recs, len(cands)

    def _admit_successful_experience(self, task: TaskInstance, res: RolloutResult) -> None:
        for existing in self.replay_buffer:
            if existing.domain == task.domain and existing.executed_plan == res.executed_plan:
                sim = float(np.dot(existing.query_embedding, task.query_embedding))
                if sim > 0.80:
                    new_emb = 0.7 * existing.query_embedding + 0.3 * task.query_embedding
                    existing.query_embedding = new_emb / (np.linalg.norm(new_emb) + 1e-12)
                    existing.observed_reward = 0.7 * existing.observed_reward + 0.3 * res.observed_reward
                    existing.episode_idx = task.episode_idx
                    return

        self._counter += 1
        self.retained_strategies += 1
        self.admitted_history.append((task.task_id, res.executed_plan))

        self.replay_buffer.append(
            MemoryEntry(
                entry_id=f"rep_{self._counter}",
                task_id=task.task_id,
                domain=task.domain,
                episode_idx=task.episode_idx,
                query_embedding=task.query_embedding.copy(),
                predicates=task.structural_predicates,
                executed_plan=res.executed_plan,
                observed_success=1,
                observed_reward=res.observed_reward,
            )
        )
        if len(self.replay_buffer) > self.max_buffer:
            self.replay_buffer = self.replay_buffer[-self.max_buffer :]

    def act_on_episode(
        self,
        task: TaskInstance,
        env: AgentEnvironmentInterface,
        max_trials: int = 2,
        learn: bool = True,
    ) -> AgentEpisodeOutcome:
        _require_agent_interface(env)
        t0 = time.perf_counter()
        p_tok_ep, c_tok_ep = 0, 0
        recs, eval_count = self._retrieve_replay(task)

        trials_used = 0
        best_res: Optional[RolloutResult] = None
        best_trial_idx = 0
        local_avoid: Dict[int, List[int]] = {}

        while trials_used < max_trials:
            trial_zero_idx = trials_used
            trials_used += 1
            if self.direct_macro_execution and trials_used == 1 and recs:
                plan = recs[0][0]
                p_tok, c_tok = self.backbone.account_auxiliary_call(
                    "replay_macro_exec",
                    f"Task={task.domain} Plan={list(plan)}",
                    f"ActionPlan={list(plan)}",
                )
            else:
                plan, p_tok, c_tok = self.backbone.generate_plan(
                    task,
                    system_role="experience_replay",
                    recommended_plans=recs if trials_used == 1 else None,
                    avoid_operators=local_avoid if local_avoid else None,
                )
            p_tok_ep += p_tok
            c_tok_ep += c_tok
            res = env.evaluate_plan(task, plan)

            if best_res is None or res.observed_reward > best_res.observed_reward:
                best_res = res
                best_trial_idx = trial_zero_idx

            if res.observed_success == 1:
                best_res = res
                best_trial_idx = trial_zero_idx
                if learn:
                    dp, dc = self.backbone.account_auxiliary_call(
                        "replay_distill",
                        f"Task={task.domain} Plan={res.executed_plan}",
                        f"DistilledWorkflow={list(res.executed_plan)}",
                    )
                    p_tok_ep += dp
                    c_tok_ep += dc
                    self._admit_successful_experience(task, res)
                break
            else:
                if learn:
                    self.rejected_strategies += 1
                if res.first_failed_step is not None and res.failed_operator is not None:
                    local_avoid.setdefault(res.first_failed_step, []).append(res.failed_operator)

        dt_ms = (time.perf_counter() - t0) * 1000.0
        assert best_res is not None
        return AgentEpisodeOutcome(
            best_trial_idx=best_trial_idx,
            trials_used=trials_used,
            strategy_evaluations=eval_count,
            execution_time_ms=dt_ms,
            active_memory_items=len(self.replay_buffer),
            memory_bytes=_estimate_entries_bytes(self.replay_buffer),
            prompt_tokens=p_tok_ep,
            completion_tokens=c_tok_ep,
            shift_detected=False,
        )

    def active_memory_count(self) -> int:
        return len(self.replay_buffer)

    def memory_footprint_bytes(self) -> int:
        return _estimate_entries_bytes(self.replay_buffer)
