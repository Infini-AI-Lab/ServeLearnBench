"""RAG: retrieval of the agent's own past episodes.

The serving stream runs in blocks of 12. Within a block the case store is
frozen, so the block's episodes are independent; at the end of the block every
episode's record (instruction, full trajectory, score) is added to the store.
For each new task the 8 most similar past cases are retrieved with Okapi BM25
(k1=1.5, b=0.75) over the question text without its timestamp line, and are
shown in the system prompt with their scores and stream positions. Test tasks
retrieve from the frozen store and are never added to it.
"""

from __future__ import annotations

import math
import re

from .. import protocol
from .base import EpisodeResult, Method

DRIFT_NOTE = ("The environment may change over time; each example's "
              "'Current timestamp' line gives the stream position at which "
              "it was observed.")
HEADER = ("# Past cases\n"
          "These are complete records of YOUR OWN earlier episodes in this "
          "environment, retrieved because their questions are similar to the "
          "current one. Each carries the score it received. " + DRIFT_NOTE + "\n")
_TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


def bm25_top_k(query: str, docs: list[list[str]], k: int,
               k1: float = 1.5, b: float = 0.75) -> list[int]:
    """Indices of the k best-scoring documents with a positive score."""
    if not docs:
        return []
    n = len(docs)
    avgdl = sum(len(d) for d in docs) / n
    df: dict[str, int] = {}
    for d in docs:
        for t in set(d):
            df[t] = df.get(t, 0) + 1
    q = tokenize(query)
    scored = []
    for i, d in enumerate(docs):
        tf: dict[str, int] = {}
        for t in d:
            tf[t] = tf.get(t, 0) + 1
        s = 0.0
        for t in q:
            if t in tf:
                idf = math.log(1 + (n - df[t] + 0.5) / (df[t] + 0.5))
                s += idf * tf[t] * (k1 + 1) / (tf[t] + k1 * (1 - b + b * len(d) / avgdl))
        scored.append((s, i))
    scored.sort(key=lambda p: p[0], reverse=True)
    return [i for s, i in scored[:k] if s > 0]


class RAG(Method):
    name = "rag"
    learns = True

    def __init__(self, top_k: int = 8, block_size: int = 12, char_cap: int = 100_000):
        self.top_k = top_k
        self.block_size = block_size
        self.char_cap = char_cap
        self.cases: list[dict] = []
        self._docs: list[list[str]] = []
        self._last: dict[str, list[int]] = {}

    def retrieve(self, instruction: str) -> list[dict]:
        idx = bm25_top_k(protocol.strip_timestamp(instruction), self._docs, self.top_k)
        return [self.cases[i] for i in idx]

    def context(self, task, phase, window):
        cases = self.retrieve(task.instruction)
        self._last[task.task_id] = [c["timestamp"] for c in cases]
        if not cases:
            return ""
        parts = [HEADER]
        for c in cases:
            rec = c["record"]
            if len(rec) > self.char_cap:
                rec = rec[:self.char_cap] + "\n  (record truncated)"
            parts.append(f"## Case observed at stream position {c['timestamp']} — "
                         f"{protocol.score_line(c['reward'])}\n{rec}\n")
        return "\n".join(parts)

    def update(self, results: list[EpisodeResult]) -> None:
        for r in results:
            self.cases.append({"timestamp": r.timestamp, "reward": r.reward,
                               "record": r.record})
            self._docs.append(tokenize(protocol.strip_timestamp(r.instruction)))

    def row_extra(self, task, phase, window):
        ts = self._last.pop(task.task_id, [])
        return {"n_retrieved": len(ts), "retrieved_timestamps": ts}

    def config(self):
        return {"method": self.name, "retriever": "bm25", "top_k": self.top_k,
                "block_size": self.block_size, "char_cap": self.char_cap}
