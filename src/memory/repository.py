"""
Tiered Strategy Repository (Module 3).

Maintains CANDIDATE, VALIDATED, and QUARANTINED strategy hypotheses indexed by
domain, along with strictly observable transition counters (zero oracle fields).
"""

from __future__ import annotations

import hashlib
from typing import Dict, List, Optional, Sequence, Tuple

from ..benchmark.types import StrategyRecord, StrategyStatus


def make_strategy_id(
    domain: str,
    preconditions: Tuple[Tuple[int, int], ...],
    operator_plan: Tuple[int, ...],
) -> str:
    raw = f"{domain}|{preconditions}|{operator_plan}".encode("utf-8")
    return "strat_" + hashlib.sha1(raw).hexdigest()[:12]


class StrategyRepository:
    """Three-tier strategy store (CANDIDATE, VALIDATED, QUARANTINED)."""

    def __init__(self, max_active_per_domain: int = 48) -> None:
        self.max_active_per_domain = max_active_per_domain
        self.records: Dict[str, StrategyRecord] = {}
        self.total_promoted: int = 0
        self.total_quarantined: int = 0
        self.candidates_created: int = 0
        self.lgg_merges: int = 0
        self.lgg_revalidations_triggered: int = 0

    def get(self, strategy_id: str) -> Optional[StrategyRecord]:
        return self.records.get(strategy_id)

    def add(self, record: StrategyRecord) -> StrategyRecord:
        self.records[record.strategy_id] = record
        return record

    def domain_strategies(
        self,
        domain: str,
        statuses: Optional[Sequence[StrategyStatus]] = None,
    ) -> List[StrategyRecord]:
        allowed = set(statuses) if statuses is not None else {
            StrategyStatus.CANDIDATE,
            StrategyStatus.VALIDATED,
        }
        return [
            r for r in self.records.values()
            if r.domain == domain and r.status in allowed
        ]

    def active_strategies(self) -> List[StrategyRecord]:
        return [
            r for r in self.records.values()
            if r.status in (StrategyStatus.CANDIDATE, StrategyStatus.VALIDATED)
        ]

    def validated_strategies(self) -> List[StrategyRecord]:
        return [r for r in self.records.values() if r.status == StrategyStatus.VALIDATED]

    def memory_bytes(self) -> int:
        active = self.active_strategies()
        quarantined = [r for r in self.records.values() if r.status == StrategyStatus.QUARANTINED]
        return len(active) * 264 + min(len(quarantined), 128) * 48
