"""Reference settings that do not learn."""

from __future__ import annotations

from .base import Method


class Blind(Method):
    """The domain system prompt and the task only; no serving experience."""
    name = "blind"


class Oracle(Method):
    """Additionally told the hidden policy that is active in the task's
    window: an upper reference for a learner that has fully recovered it."""
    name = "oracle"

    def context(self, task, phase, window):
        rules = self.bundle["rules"]
        return rules.oracle_note(rules.WINDOW_STARTS[window])
