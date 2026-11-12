"""Core typed data structures for tasks, rollouts, strategy hypotheses, and telemetry.

Strict zero-leakage design:
- TaskInstance contains ONLY observable task inputs (zero ground-truth plans,
  zero latent archetype IDs, and zero hidden regime indices).
- RolloutResult contains ONLY observable environment feedback returned to the
  agent during interaction (zero true_success or oracle step_matches).
- AuditOutcome stores ground-truth evaluation quantities strictly inside the
  benchmark harness for post-hoc scoring.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple
import numpy as np


class StrategyStatus(str, Enum):
    """Lifecycle tier of a strategy hypothesis in the Strategy Repository."""

    CANDIDATE = "CANDIDATE"
    VALIDATED = "VALIDATED"
    QUARANTINED = "QUARANTINED"


@dataclass(frozen=True)
class TaskInstance:
    """Observable multi-step procedural task presented to an agent in ShiftBench-Sim."""

    task_id: str
    episode_idx: int
    domain: str
    structural_predicates: Tuple[int, ...]       # 5 observable binary state predicates
    surface_topic_id: int                        # Observable surface scenario index in {0..7}
    context_features: np.ndarray                 # Continuous observable 12-D context vector
    query_embedding: np.ndarray                  # L2-normalized 20-D task embedding
    is_unseen_variant: bool = False


@dataclass(frozen=True)
class AuditOutcome:
    """Environment-private ground-truth audit record (never exposed to agents)."""

    task_id: str
    domain: str
    regime_id: int
    governing_archetype: int
    optimal_plan: Tuple[int, ...]
    executed_plan: Tuple[int, ...]
    alignment_score: float
    true_success: int
    true_reward: float
    observed_success: int
    observed_reward: float
    feedback_corrupted: bool


@dataclass
class RolloutResult:
    """Observable outcome of executing a 4-step procedural plan in ShiftBench-Sim."""

    executed_plan: Tuple[int, ...]
    observed_success: int
    observed_reward: float
    first_failed_step: Optional[int]
    failed_operator: Optional[int]


@dataclass
class StrategyRecord:
    """Structured falsifiable strategy hypothesis h = (id, d, phi, c, pi, tau, alpha, beta, Lambda)."""

    strategy_id: str
    domain: str
    preconditions: Tuple[Tuple[int, int], ...]
    embedding: np.ndarray
    operator_plan: Tuple[int, ...]
    alpha: float = 1.0
    beta: float = 1.0
    n_trials: int = 0
    n_successes: int = 0
    sprt_llr: float = 0.0
    consecutive_failures: int = 0
    status: StrategyStatus = StrategyStatus.CANDIDATE
    created_episode: int = 0
    last_updated_episode: int = 0
    promoted_episode: Optional[int] = None
    invalidated_episode: Optional[int] = None
    created_task_id: str = ""
    lgg_merge_count: int = 0
    quarantine_count: int = 0

    def matches_preconditions(self, predicates: Tuple[int, ...]) -> bool:
        """Evaluate symbolic applicability conjunction phi_h(psi_t)."""
        for idx, req_val in self.preconditions:
            if idx >= len(predicates) or predicates[idx] != req_val:
                return False
        return True


@dataclass
class MemoryEntry:
    """Episodic or reflective record used by Conventional Memory, Reflection, and Replay baselines."""

    entry_id: str
    task_id: str
    domain: str
    episode_idx: int
    query_embedding: np.ndarray
    predicates: Tuple[int, ...]
    executed_plan: Tuple[int, ...]
    observed_success: int
    observed_reward: float
    first_failed_step: Optional[int] = None
    failed_operator: Optional[int] = None
    reflection_avoid_map: Dict[int, List[int]] = field(default_factory=dict)


@dataclass(frozen=True)
class AgentEpisodeOutcome:
    """Agent-side episode execution summary containing strictly non-oracle telemetry."""

    best_trial_idx: int
    trials_used: int
    strategy_evaluations: int
    execution_time_ms: float
    active_memory_items: int
    memory_bytes: int
    prompt_tokens: int
    completion_tokens: int
    shift_detected: bool = False


@dataclass
class EpisodeTelemetry:
    """Granular per-episode telemetry assembled post-episode strictly by the evaluator harness."""

    experiment_name: str
    agent_name: str
    seed: int
    episode_idx: int
    domain: str
    regime_id: int
    true_success: int
    observed_success: int
    true_reward: float
    observed_reward: float
    alignment_score: float
    trials_used: int
    strategy_evaluations: int
    execution_time_ms: float
    active_memory_items: int
    memory_bytes: int
    prompt_tokens: int
    completion_tokens: int
    shift_detected: bool = False
    contaminated_active_ratio: float = 0.0
