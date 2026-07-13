"""Small deterministic progress and ETA reporting helpers for v6 pipeline work."""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from time import monotonic


def format_duration(seconds: float) -> str:
    """Format a non-negative duration without hiding the coarse ETA precision."""
    rounded = max(0, round(seconds))
    minutes, seconds = divmod(rounded, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h{minutes:02d}m"
    if minutes:
        return f"{minutes}m{seconds:02d}s"
    return f"{seconds}s"


@dataclass
class EtaProgress:
    """Format completed/total, rate, elapsed time, ETA, and total estimate."""

    label: str
    total: int
    clock: Callable[[], float] = monotonic
    started_at: float = field(init=False)

    def __post_init__(self) -> None:
        if self.total < 0:
            raise ValueError("total must be non-negative")
        self.started_at = self.clock()

    def update(self, completed: int, *, force: bool = False) -> str:
        del force  # Call sites decide report cadence; keeping this makes them explicit.
        completed = min(max(completed, 0), self.total)
        elapsed = max(0.0, self.clock() - self.started_at)
        if completed <= 0 or elapsed <= 0:
            return (
                f"[{self.label} {completed}/{self.total} rate=unknown "
                f"elapsed={format_duration(elapsed)} eta=unknown total_est=unknown]"
            )

        rate = completed / elapsed
        total_estimate = elapsed * self.total / completed
        eta = max(0.0, total_estimate - elapsed)
        return (
            f"[{self.label} {completed}/{self.total} rate={rate:.2f} it/s "
            f"elapsed={format_duration(elapsed)} eta={format_duration(eta)} "
            f"total_est={format_duration(total_estimate)}]"
        )
