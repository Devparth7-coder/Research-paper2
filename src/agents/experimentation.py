"""
Controlled Experimentation Engine for Candidate Validation vs Validated Exploitation (Module 4).

Allocates rollouts strictly according to repository validation state and learning mode:
1. If an applicable VALIDATED strategy passing LCB >= tau_rel exists, exploit the
   top-ranked validated hypothesis.
2. Otherwise, if `learn=True` and an applicable CANDIDATE strategy exists, run a
   targeted validation experiment on the highest Bayes-UCB candidate.
3. During frozen evaluation (`learn=False`), unvalidated CANDIDATE hypotheses are
   NEVER executed; the agent falls back to policy synthesis when no VALIDATED
   strategy applies.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

from ..benchmark.types import StrategyRecord


class ExperimentationEngine:
    """Balances exploitation of validated strategies and sequential testing of candidates."""

    def select_strategy_for_trial(
        self,
        val_matches: List[Tuple[float, StrategyRecord]],
        cand_matches: List[Tuple[float, StrategyRecord]],
        trial_idx: int = 1,
        learn: bool = True,
    ) -> Tuple[Optional[StrategyRecord], bool]:
        """Return `(selected_strategy_or_None, is_validated_exploitation)`."""
        _ = trial_idx
        if val_matches:
            return val_matches[0][1], True
        if learn and cand_matches:
            return cand_matches[0][1], False
        return None, False
