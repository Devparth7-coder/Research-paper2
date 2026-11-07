"""
Per-Domain Page-Hinkley / One-Sided Residual CUSUM Drift Detector (Module 9).

Monitors one-step out-of-sample predictive residuals e_t = p_hat_{h*}^{(t-1)} - y_tilde_t
on validated strategy executions within each domain d. Incorporates a post-reset
observation warmup (`min_obs_since_reset = 3`) and inter-alarm cooldown
(`cooldown_episodes = 12`) to prevent chattering alarms during recovery.
"""

from __future__ import annotations

from typing import Dict, List, Tuple


class DistributionShiftDetector:
    """Detects abrupt or gradual regime shifts per domain from validated strategy residuals."""

    def __init__(
        self,
        slack_nu: float = 0.12,
        drift_threshold: float = 1.45,
        min_obs_since_reset: int = 3,
        cooldown_episodes: int = 12,
        shift_discount_gamma: float = 0.35,
        enabled: bool = True,
    ) -> None:
        self.slack_nu = slack_nu
        self.drift_threshold = drift_threshold
        self.min_obs_since_reset = min_obs_since_reset
        self.cooldown_episodes = cooldown_episodes
        self.shift_discount_gamma = shift_discount_gamma
        self.enabled = enabled

        self.cusum_stat: Dict[str, float] = {}
        self.obs_since_reset: Dict[str, int] = {}
        self.last_alarm_episode: Dict[str, int] = {}
        self.alarm_episodes: List[Tuple[int, str]] = []

    def update(
        self,
        domain: str,
        predicted_success_prob: float,
        observed_success: int,
        episode_idx: int,
    ) -> bool:
        """Update domain CUSUM accumulator; return True if a shift alarm triggers."""
        if not self.enabled:
            return False

        prev_s = self.cusum_stat.get(domain, 0.0)
        n_obs = self.obs_since_reset.get(domain, 0) + 1
        self.obs_since_reset[domain] = n_obs

        residual = float(predicted_success_prob) - float(observed_success)
        new_s = max(0.0, prev_s + residual - self.slack_nu)
        self.cusum_stat[domain] = new_s

        last_ep = self.last_alarm_episode.get(domain, -10_000)
        in_cooldown = (episode_idx - last_ep) < self.cooldown_episodes

        if (
            n_obs >= self.min_obs_since_reset
            and not in_cooldown
            and new_s >= self.drift_threshold
        ):
            self.alarm_episodes.append((episode_idx, domain))
            self.last_alarm_episode[domain] = episode_idx
            self.cusum_stat[domain] = 0.0
            self.obs_since_reset[domain] = 0
            return True
        return False

    def get_cusum_value(self, domain: str) -> float:
        return self.cusum_stat.get(domain, 0.0)
