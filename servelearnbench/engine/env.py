"""Environment layer: holds data + tools + policy + verifier + trajectory.

Reward logic is NOT here -- Env only calls the verifier on demand. This keeps
the eval/serving split entirely inside the Verifier.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Type

from .tool import Tool
from .types import FINISH_ACTION_NAME, RESPOND_ACTION_NAME, Action, RewardResult, StepResult, Task
from .verifier import EvalVerifier


def apply_episode_view(data, task) -> None:
    """Split hygiene (opt-in via the loader's '_episode_view' marker): the
    episode sees the base world plus ITS OWN order only — other serving/test
    orders are not queryable, so the frozen test suite cannot be pre-studied
    from inside the stream. MUST be applied identically to the agent episode
    AND the verifier's GT-replay reference, or every state hash diverges."""
    view = data.pop("_episode_view", None)
    if view is None or task is None:
        return
    if view.get("kind") == "banking":
        # an episode sees the shared world (accounts/merchants/payees) plus
        # ITS OWN case only — other cases are not queryable, in any table
        cid = (task.z or {}).get("case_id")
        for tbl in ("transactions", "limit_requests", "transfers"):
            data[tbl] = {k: v for k, v in data.get(tbl, {}).items() if k == cid}
        return
    allow = set(view["base_orders"])
    oid = (task.z or {}).get("order_id")
    if oid:
        allow.add(oid)
    data["orders"] = {k: v for k, v in data["orders"].items() if k in allow}
    for u in data["users"].values():
        if isinstance(u.get("orders"), list):
            u["orders"] = [o for o in u["orders"] if o in allow]
    # serving-only catalog variants are invisible to TEST episodes: their
    # tool observations stay byte-identical to the pre-split catalog
    sov = view.get("serving_only_variants")
    if sov and (task.z or {}).get("split") == "test":
        drop = set(sov)
        for p in data["products"].values():
            for iid in list(p["variants"]):
                if iid in drop:
                    del p["variants"][iid]



class Env:
    def __init__(
        self,
        load_data: Callable[[], Dict[str, Any]],
        tools: List[Type[Tool]],
        policy: str = "",
        workflow: str = "",
        docs_toc: str = "",
        verifier: Optional[EvalVerifier] = None,
        user: Optional[Any] = None,
    ) -> None:
        self.load_data = load_data
        self.tools = tools
        self.tools_map: Dict[str, Type[Tool]] = {t.name(): t for t in tools}
        self.tools_info = [t.get_info() for t in tools]
        self.policy = policy
        self.workflow = workflow
        self.docs_toc = docs_toc  # retrievable-section index, shown in the system prompt
        self.verifier = verifier or EvalVerifier()
        self.user = user  # optional LLM user simulator
        self.data: Dict[str, Any] = {}
        self.task: Optional[Task] = None
        self.trajectory: List[Action] = []
        self.reset()

    # Budget: 50 TOOL calls per question (finish/respond are exempt — the cap
    # is on world interactions, not on answering). Enforced HERE so every
    # runner and the GT replay share one authority.
    from ..protocol import ENGINE_TOOL_BUDGET as TOOL_BUDGET

    def reset(self, task: Optional[Task] = None) -> None:
        self.data = self.load_data()  # fresh clean DB
        apply_episode_view(self.data, task)
        self.task = task
        self.trajectory = []
        self._tool_calls = 0

    def step(self, action: Action) -> StepResult:
        self.trajectory.append(action)
        if action.name == FINISH_ACTION_NAME:
            # terminal structured answer; never touches the DB
            return StepResult(observation="", source="finish")
        if action.name == RESPOND_ACTION_NAME:
            # answered by the user simulator when one is attached
            if self.user is not None:
                obs = self.user.step(action.kwargs.get("content", ""))
            else:
                obs = ""
            return StepResult(observation=obs, source="user")
        if action.name in self.tools_map:
            if self._tool_calls >= self.TOOL_BUDGET:
                return StepResult(observation=(
                    f"Error: tool-call budget ({self.TOOL_BUDGET}) exhausted — "
                    "no further tool calls are possible; submit your final "
                    "answer with `finish`."), source="tool")
            self._tool_calls += 1
            try:
                obs = self.tools_map[action.name].invoke(data=self.data, **action.kwargs)
            except Exception as e:  # errors become observations
                obs = f"Error: {e}"
            return StepResult(observation=obs, source=action.name)
        return StepResult(observation=f"Unknown action {action.name}", source="system")

    def compute_reward(self) -> RewardResult:
        assert self.task is not None, "compute_reward called before reset(task)"
        return self.verifier.compute(
            agent_data=self.data,
            task=self.task,
            load_data=self.load_data,
            tools_map=self.tools_map,
            trajectory=self.trajectory,
        )
