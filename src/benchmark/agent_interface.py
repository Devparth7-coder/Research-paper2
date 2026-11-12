"""
Capability-Restricted Agent Environment Interface (`src/benchmark/agent_interface.py`).

Separates evaluator-only ground truth (`NonStationaryTaskEnvironment`) from the
agent-facing interaction handle (`AgentEnvironmentInterface`).

Architectural Guarantee & Scope:
- `AgentEnvironmentInterface` defines `__slots__ = ("__weakref__",)` and stores its
  rollout callback inside a module-private `WeakKeyDictionary` rather than on `self`.
- Consequently, an `AgentEnvironmentInterface` instance holds ZERO attributes, ZERO
  `__dict__`, and ZERO references inspectable via `getattr`/`vars`/`dir` to the
  evaluator's `regime_tables`, `_task_meta`, `get_optimal_plan`, `get_task_oracle`,
  `_rollout_audit_log`, or `feedback_corruption_rate`.
- Scope Note: This provides architectural API encapsulation against accidental oracle
  access by benchmark agents; because agents and evaluator run inside the same Python
  interpreter process, it is an engineering boundary rather than an OS/process-level
  security sandbox against adversarial code introspection (e.g., `gc` or module globals).
"""

from __future__ import annotations

from typing import Callable, Tuple
from weakref import WeakKeyDictionary

from .types import RolloutResult, TaskInstance

_RolloutFn = Callable[[TaskInstance, Tuple[int, ...]], RolloutResult]
_CAPABILITY_DISPATCH: "WeakKeyDictionary[AgentEnvironmentInterface, _RolloutFn]" = WeakKeyDictionary()


class AgentEnvironmentInterface:
    """Capability-restricted environment handle passed to agents during episodes."""

    __slots__ = ("__weakref__",)

    def __init__(self, rollout_fn: _RolloutFn) -> None:
        if not callable(rollout_fn):
            raise TypeError("rollout_fn must be callable.")
        _CAPABILITY_DISPATCH[self] = rollout_fn

    def evaluate_plan(self, task: TaskInstance, executed_plan: Tuple[int, ...]) -> RolloutResult:
        """Execute `executed_plan` on `task` and return strictly observable `RolloutResult`."""
        fn = _CAPABILITY_DISPATCH.get(self)
        if fn is None:
            raise RuntimeError("AgentEnvironmentInterface capability handle is no longer active.")
        return fn(task, tuple(int(x) for x in executed_plan))
