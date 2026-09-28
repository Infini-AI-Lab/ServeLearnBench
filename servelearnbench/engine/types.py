"""Engine data contracts. Domain-agnostic; verifier and replay build on these."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel

RESPOND_ACTION_NAME = "respond"
RESPOND_ACTION_FIELD = "content"
FINISH_ACTION_NAME = "finish"  # terminal action carrying the structured final answer


class Action(BaseModel):
    name: str  # a tool name, or RESPOND_ACTION_NAME
    kwargs: Dict[str, Any] = {}


class Task(BaseModel):
    task_id: str
    user_id: str
    instruction: str  # natural-language task text
    actions: List[Action]  # GT recipe; replay skips non-tool actions (respond/finish)
    # Graded final answer. None -> state-only task (cancel/return). Otherwise one of:
    #   {"kind": "value", "expect": <any>, "compare": "int"|"str_norm"|"set"}
    #   {"kind": "refuse"}   -> model must submit decision=refuse (invalid/against-policy request)
    answer: Optional[Dict[str, Any]] = None
    outputs: List[str] = []  # unused; graded answers go in `answer`
    gt: Optional[Any] = None  # reserved; unused
    z: Optional[Dict[str, Any]] = None  # drift variables
    stage: Optional[int] = None


class RewardResult(BaseModel):
    reward: float  # {0.0, 1.0} for EvalVerifier; continuous in [0, 1] for rubric scoring
    r_state: bool  # DB terminal-state hash match
    r_answer: bool  # structured final-answer match (typed/canonical, or refuse decision)
    agent_data_hash: str
    gt_data_hash: str
    answer_detail: Dict[str, Any] = {}


class StepResult(BaseModel):
    observation: str
    source: str  # tool name, "user", or "system"
