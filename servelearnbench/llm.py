"""Chat-completion client for any OpenAI-compatible endpoint.

One streaming call path serves both the acting agent and the Pitch judge. It
reassembles streamed tool calls, retries transient failures with backoff, and
enforces a hard per-call wall clock with a watchdog (keep-alive bytes on a
stalled stream would otherwise reset the read timeout forever). Token usage is
accounted per thread, separately for the agent and for the judge, so a result
row can report the agent's own cost.
"""

from __future__ import annotations

import os
import random
import socket
import threading
import time
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import httpx
from openai import NOT_GIVEN, APIConnectionError, APIError, OpenAI, RateLimitError

from . import protocol

# ---------------------------------------------------------------------------
# token accounting
# ---------------------------------------------------------------------------
USAGE_KEYS = ("input", "cache_input", "output", "reasoning", "calls")
_TL = threading.local()


def _zero() -> dict:
    return dict.fromkeys(USAGE_KEYS, 0)


def usage_reset() -> None:
    """Start a fresh per-episode account on this thread."""
    _TL.agent = _zero()
    _TL.judge = _zero()


def agent_usage() -> dict:
    return dict(getattr(_TL, "agent", None) or _zero())


def judge_usage() -> dict:
    return dict(getattr(_TL, "judge", None) or _zero())


def _record_usage(u, judge: bool) -> None:
    if u is None:
        return
    det = getattr(u, "prompt_tokens_details", None)
    cache = (getattr(det, "cached_tokens", 0) or 0) if det else 0
    cache = cache or (getattr(u, "prompt_cache_hit_tokens", 0) or 0)
    odet = (getattr(u, "completion_tokens_details", None)
            or getattr(u, "output_tokens_details", None))
    acct_name = "judge" if judge else "agent"
    acct = getattr(_TL, acct_name, None)
    if acct is None:
        acct = _zero()
        setattr(_TL, acct_name, acct)
    acct["input"] += getattr(u, "prompt_tokens", 0) or 0
    acct["cache_input"] += cache
    acct["output"] += getattr(u, "completion_tokens", 0) or 0
    acct["reasoning"] += (getattr(odet, "reasoning_tokens", 0) or 0) if odet else 0
    acct["calls"] += 1


# ---------------------------------------------------------------------------
# errors
# ---------------------------------------------------------------------------
class ProviderParseError(RuntimeError):
    """The provider returned a tool call that cannot be reassembled."""


class InfraError(RuntimeError):
    """The request failed for infrastructure reasons after every retry. An
    episode that hits this is never scored; the runner reruns it."""


_OVERFLOW_WORDS = ("context length", "context_length", "maximum context",
                   "context window", "too long", "token limit",
                   "maximum length", "max_tokens", "exceeds the maximum")


def is_context_overflow(e: Exception) -> bool:
    """A 4xx that names the context limit is the model's own doing (a
    runaway generation filled the window); the episode is scored as it
    stands. Everything else is infrastructure."""
    if isinstance(e, (RateLimitError, APIConnectionError, ProviderParseError)):
        return False
    if isinstance(e, APIError):
        status = getattr(e, "status_code", 500) or 500
        msg = str(e).lower()
        return 400 <= status < 500 and any(w in msg for w in _OVERFLOW_WORDS)
    return False


# ---------------------------------------------------------------------------
# client
# ---------------------------------------------------------------------------
@dataclass
class ModelConfig:
    """How to reach a model. `reasoning_effort` is sent as-is in the request
    body; set it to None to use the provider's default."""
    model: str
    base_url: str = "https://api.fireworks.ai/inference/v1"
    api_key: str | None = None
    reasoning_effort: str | None = "high"
    max_tokens: int | None = protocol.MAX_TOKENS
    call_timeout_s: float = 600.0
    retries: int = 8
    stall_retries: int = 2
    extra_body: dict = field(default_factory=dict)

    def resolved_key(self) -> str:
        key = self.api_key or os.environ.get("SLB_API_KEY") or os.environ.get("OPENAI_API_KEY")
        return key or "EMPTY"   # local OpenAI-compatible servers do not check keys


class LLM:
    def __init__(self, cfg: ModelConfig):
        self.cfg = cfg
        self.client = OpenAI(base_url=cfg.base_url, api_key=cfg.resolved_key(),
                             timeout=httpx.Timeout(connect=15.0, read=240.0,
                                                   write=60.0, pool=60.0),
                             max_retries=0)

    def _body(self) -> dict | None:
        body = dict(self.cfg.extra_body)
        if self.cfg.reasoning_effort:
            body["reasoning_effort"] = self.cfg.reasoning_effort
        return body or None

    def complete(self, messages: list, *, tools: list | None = None,
                 temperature: float | None = None, top_p: float | None = None,
                 judge: bool = False) -> SimpleNamespace:
        """One streamed chat completion. Returns an object with `content`,
        `reasoning`, `tool_calls` ([{id, name, arguments}]) and
        `finish_reason`. Raises the last error once retries are exhausted."""
        cfg = self.cfg
        stalls = 0
        for attempt in range(cfg.retries):
            stalled = {"v": False}
            watchdog = None
            try:
                stream = self.client.chat.completions.create(
                    model=cfg.model, messages=messages, stream=True,
                    temperature=temperature if temperature is not None else NOT_GIVEN,
                    top_p=top_p if top_p is not None else NOT_GIVEN,
                    max_tokens=cfg.max_tokens or NOT_GIVEN,
                    stream_options={"include_usage": True},
                    tools=tools or NOT_GIVEN,
                    parallel_tool_calls=False if tools else NOT_GIVEN,
                    extra_body=self._body())

                def _kill(s=stream, f=stalled):
                    f["v"] = True
                    try:   # wake a thread blocked in recv() on this socket
                        ns = s.response.extensions.get("network_stream")
                        sock = ns.get_extra_info("socket") if ns is not None else None
                        if sock is not None:
                            sock.shutdown(socket.SHUT_RDWR)
                    except Exception:
                        pass
                    try:
                        s.close()
                    except Exception:
                        pass

                watchdog = threading.Timer(cfg.call_timeout_s, _kill)
                watchdog.daemon = True
                watchdog.start()
                t0 = time.monotonic()
                content, reasoning, calls = [], [], {}
                finish_reason, last_usage = None, None
                for chunk in stream:
                    if time.monotonic() - t0 > cfg.call_timeout_s:
                        stream.close()
                        stalled["v"] = True
                        raise httpx.ReadTimeout("stream exceeded the call wall clock")
                    if getattr(chunk, "usage", None):
                        last_usage = chunk.usage   # some endpoints send cumulative usage
                    if not getattr(chunk, "choices", None):
                        continue
                    ch = chunk.choices[0]
                    if getattr(ch, "finish_reason", None):
                        finish_reason = ch.finish_reason
                    d = ch.delta
                    if d is None:
                        continue
                    if getattr(d, "content", None):
                        content.append(d.content)
                    rc = getattr(d, "reasoning_content", None) or getattr(d, "reasoning", None)
                    if rc:
                        reasoning.append(rc)
                    for tc in getattr(d, "tool_calls", None) or []:
                        idx = tc.index if getattr(tc, "index", None) is not None else len(calls)
                        slot = calls.setdefault(idx, {"id": None, "name": None, "args": []})
                        if getattr(tc, "id", None) and not slot["id"]:
                            slot["id"] = tc.id
                        fn = getattr(tc, "function", None)
                        if fn is not None:
                            if getattr(fn, "name", None) and not slot["name"]:
                                slot["name"] = fn.name
                            if getattr(fn, "arguments", None):
                                slot["args"].append(fn.arguments)
                if stalled["v"]:
                    raise httpx.ReadTimeout("watchdog closed a stalled stream")
                _record_usage(last_usage, judge)
                tool_calls = []
                for idx in sorted(calls):
                    slot = calls[idx]
                    if not slot["name"]:
                        raise ProviderParseError(f"tool call #{idx} arrived without a name")
                    tool_calls.append({"id": slot["id"] or f"call_{idx}",
                                       "name": slot["name"],
                                       "arguments": "".join(slot["args"])})
                return SimpleNamespace(content="".join(content),
                                       reasoning="".join(reasoning) or None,
                                       tool_calls=tool_calls,
                                       finish_reason=finish_reason)
            except (ProviderParseError, RateLimitError, APIConnectionError, httpx.HTTPError):
                if stalled["v"]:
                    stalls += 1
                if attempt == cfg.retries - 1 or stalls >= cfg.stall_retries:
                    raise
                _backoff(attempt)
            except APIError as e:
                transient = (getattr(e, "status_code", 500) or 500) >= 500 or "NaN" in str(e)
                if not transient or attempt == cfg.retries - 1:
                    raise
                _backoff(attempt)
            except Exception:
                if not stalled["v"]:
                    raise
                stalls += 1
                if attempt == cfg.retries - 1 or stalls >= cfg.stall_retries:
                    raise httpx.ReadTimeout("watchdog closed a stalled stream")
                _backoff(attempt)
            finally:
                if watchdog is not None:
                    watchdog.cancel()
        raise InfraError("retries exhausted")


def _backoff(attempt: int) -> None:
    time.sleep(min(60.0, (2 ** attempt) + random.random()))


# ---------------------------------------------------------------------------
# the Pitch judge
# ---------------------------------------------------------------------------
DEFAULT_JUDGE_MODEL = "accounts/fireworks/models/glm-5p3-flash"
_JUDGE: dict[str, Any] = {"llm": None}


def set_judge(cfg: ModelConfig) -> None:
    """Configure the model that scores Pitch. The paper's scorer is
    glm-5p3-flash at reasoning_effort=high, temperature 0."""
    _JUDGE["llm"] = LLM(cfg)


def judge() -> LLM:
    if _JUDGE["llm"] is None:
        raise RuntimeError("the Pitch judge is not configured; call "
                           "servelearnbench.llm.set_judge(...) or pass --judge-model")
    return _JUDGE["llm"]
