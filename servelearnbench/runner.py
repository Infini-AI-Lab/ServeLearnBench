"""Run a method on a scenario: the serving stream and the window test sets.

For each environment window, in order:

1. if the method learns, the window's serving tasks run in blocks of
   `method.block_size`; episodes within a block run in parallel against the
   method's frozen state, and the method is updated with the block's records
   after it completes;
2. the window's held-out test tasks run in parallel against the frozen state;
   their outcomes are recorded but never shown to the method.

Every episode is one JSON line in the output file; the first line records the
configuration. With `resume=True` a run continues from an existing file:
finished test episodes are kept, the method is rebuilt from the serving blocks
that completed, and a partially finished block is run again as a whole.
"""

from __future__ import annotations

import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from . import agent, llm, protocol
from .methods import EpisodeResult, Method
from .scenarios import get_bundle, n_windows
from .scoring import category

EPISODE_ATTEMPTS = 4   # reruns of an episode that failed for infrastructure reasons


def _config(scenario: str, method: Method, model_cfg: llm.ModelConfig) -> dict:
    return {"scenario": scenario, **method.config(), "model": model_cfg.model,
            "reasoning_effort": model_cfg.reasoning_effort,
            "max_tokens": model_cfg.max_tokens, "max_steps": protocol.MAX_STEPS,
            "tool_budget": protocol.ENGINE_TOOL_BUDGET}


class Runner:
    def __init__(self, scenario: str, method: Method, model_cfg: llm.ModelConfig,
                 out: str, workers: int = 8, resume: bool = False,
                 windows: list[int] | None = None, log=print, model=None):
        self.scenario = scenario
        self.method = method
        self.model_cfg = model_cfg
        self.model = model or llm.LLM(model_cfg)
        self.out = out
        self.workers = workers
        self.resume = resume
        self.log = log
        self.bundle = get_bundle(scenario, fresh=True)
        self.domain = self.bundle["meta"]["domain"]
        self.windows = windows if windows is not None else list(range(n_windows(self.bundle)))
        self._lock = threading.Lock()
        self.config = _config(scenario, method, model_cfg)

    # -- one episode -----------------------------------------------------------
    def _episode(self, task, phase: str, window: int, ts: int) -> dict:
        extra = self.method.context(task, phase, window)
        last = None
        for attempt in range(EPISODE_ATTEMPTS):
            try:
                reward, trace, usage, judge_usage = agent.run_episode(
                    self.model, self.bundle, task, extra_system=extra)
                break
            except Exception as e:  # noqa: BLE001  (infrastructure: rerun)
                last = e
                self.log(f"  [retry] {task.task_id}: {type(e).__name__}: {str(e)[:160]}")
                time.sleep(min(30 * (attempt + 1), 120))
        else:
            raise RuntimeError(f"{task.task_id}: {EPISODE_ATTEMPTS} attempts failed") from last
        z = task.z or {}
        row = {"scenario": self.scenario, "method": self.method.name,
               "model": self.model_cfg.model, "phase": phase, "window": window,
               "timestamp": ts, "task_id": task.task_id, "tier": z.get("tier"),
               "family": z.get("family"), "slice": z.get("slice"),
               "category": category(self.domain, z.get("slice")),
               "reward": reward.reward, "answer_detail": reward.answer_detail,
               "instruction": task.instruction, "tokens": usage,
               "judge_tokens": judge_usage, "trace": trace,
               **self.method.row_extra(task, phase, window)}
        if phase == "serving" and self.method.learns:
            row["record"] = protocol.episode_record(task.instruction, trace, reward.reward)
        with self._lock:
            self._fh.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
            self._fh.flush()
        return row

    def _parallel(self, jobs) -> list[dict]:
        if not jobs:
            return []
        with ThreadPoolExecutor(max_workers=self.workers) as ex:
            return list(ex.map(lambda j: self._episode(*j), jobs))

    @staticmethod
    def _learning_result(row: dict) -> EpisodeResult:
        return EpisodeResult(task_id=row["task_id"], instruction=row["instruction"],
                             timestamp=row["timestamp"], reward=row["reward"],
                             record=row["record"])

    # -- resume ---------------------------------------------------------------
    def _load_previous(self) -> dict[tuple[str, str], dict]:
        if not (self.resume and os.path.exists(self.out)):
            if os.path.exists(self.out) and os.path.getsize(self.out):
                raise FileExistsError(f"{self.out} exists; pass --resume or choose another path")
            with open(self.out, "w") as f:
                f.write(json.dumps({"config": self.config}) + "\n")
            return {}
        rows = [json.loads(line) for line in open(self.out) if line.strip()]
        if not rows or "config" not in rows[0]:
            raise ValueError(f"{self.out} has no config line")
        if rows[0]["config"] != self.config:
            raise ValueError(f"resume refused: configuration differs from {self.out}\n"
                             f"  file: {rows[0]['config']}\n  now:  {self.config}")
        return {(r["phase"], r["task_id"]): r for r in rows[1:]}

    def _rewrite(self, keep: dict[tuple[str, str], dict]) -> None:
        with open(self.out, "w") as f:
            f.write(json.dumps({"config": self.config}) + "\n")
            for r in keep.values():
                f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")

    # -- the stream -------------------------------------------------------------
    def run(self) -> str:
        b = self.bundle
        self.method.setup(b)
        serving = list(b["serving"])
        protocol.stamp_serving(serving)
        done = self._load_previous()

        # Blocks are fixed by the stream, so a resumed run can tell complete
        # blocks (replayed into the method) from an interrupted one (rerun).
        blocks = {}
        for w in self.windows:
            wt = [t for t in serving if t.z["window"] == w]
            blocks[w] = [wt[i:i + self.method.block_size]
                         for i in range(0, len(wt), self.method.block_size)]
        if done and self.method.learns:
            partial = []
            for w in self.windows:
                for blk in blocks[w]:
                    have = [done.get(("serving", t.task_id)) for t in blk]
                    if all(have):
                        continue
                    partial += [("serving", t.task_id) for t, h in zip(blk, have) if h]
            for k in partial:
                done.pop(k)
            if partial:
                self.log(f"resume: rerunning {len(partial)} episodes of interrupted blocks")
                self._rewrite(done)
        self._fh = open(self.out, "a")
        try:
            for w in self.windows:
                if self.method.learns:
                    for blk in blocks[w]:
                        rows = [done[("serving", t.task_id)] for t in blk
                                if ("serving", t.task_id) in done]
                        if not rows:
                            rows = self._parallel([(t, "serving", w, t.z["stream_pos"])
                                                   for t in blk])
                        self.method.update([self._learning_result(r) for r in rows])
                    n = len([t for t in serving if t.z["window"] == w])
                    self.log(f"W{w} serving: {n} episodes")
                tests = b["test_set"](w)
                ts = protocol.stamp_test(tests, b["serving"], w)
                todo = [(t, "test", w, ts) for t in tests if ("test", t.task_id) not in done]
                rows = self._parallel(todo) + [done[("test", t.task_id)] for t in tests
                                               if ("test", t.task_id) in done]
                mean = 100 * sum(r["reward"] for r in rows) / max(1, len(rows))
                self.log(f"W{w} test: {len(rows)} tasks, mean reward {mean:.1f}")
        finally:
            self._fh.close()
        return self.out


def run(scenario: str, method: Method, model_cfg: llm.ModelConfig, out: str,
        **kw: Any) -> str:
    return Runner(scenario, method, model_cfg, out, **kw).run()
