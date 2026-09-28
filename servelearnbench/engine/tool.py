"""Dual-faced Tool abstraction.

Every tool exposes two faces:
  - invoke(data, **kwargs) -> str : backend. Business validation is inline; it
    mutates `data` in place and returns a string observation. Errors are
    RETURNED as strings, never raised, so the reward layer only sees DB effects.
  - get_info() -> dict : the OpenAI function-call schema the agent sees.
"""

from __future__ import annotations

import abc
from typing import Any, Dict


class Tool(abc.ABC):
    @staticmethod
    def invoke(data: Dict[str, Any], **kwargs: Any) -> str:
        raise NotImplementedError

    @staticmethod
    def get_info() -> Dict[str, Any]:
        raise NotImplementedError

    @classmethod
    def name(cls) -> str:
        return cls.get_info()["function"]["name"]
