"""The interface every method implements.

A method sees the serving stream through two hooks and nothing else:

* `context(task, phase, window)` returns text appended to the agent's system
  prompt for one episode (retrieved cases, notes, a policy bulletin, ...).
* `update(results)` is called after each block of serving episodes with
  their records and scores. It is never called for test episodes.

Methods with `learns = False` skip the serving stream and are evaluated on
the test sets only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class EpisodeResult:
    """What a method may learn from after a serving episode."""
    task_id: str
    instruction: str      # as the agent received it, timestamp line included
    timestamp: int        # stream position
    reward: float         # the score the agent was told, as a fraction
    record: str           # protocol.episode_record(...): the full episode text


class Method:
    name = "method"
    learns = False
    block_size = 12       # serving episodes per update

    def setup(self, bundle: dict[str, Any]) -> None:
        """Called once with the scenario bundle before the stream starts."""
        self.bundle = bundle

    def context(self, task, phase: str, window: int) -> str:
        return ""

    def update(self, results: list[EpisodeResult]) -> None:
        pass

    def row_extra(self, task, phase: str, window: int) -> dict:
        """Extra fields recorded on the episode's result row."""
        return {}

    def config(self) -> dict:
        return {"method": self.name}
