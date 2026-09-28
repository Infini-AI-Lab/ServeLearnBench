# Methods

## Included methods

**Blind** sends the domain system prompt and the task, and keeps no serving experience. It runs on the test sets
only.

**Oracle** additionally appends the hidden policy of the task's window to the system prompt (for example the store's
current rules with their refusal codes, the bank's current thresholds, or the current customer's likes and
dislikes). It runs on the test sets only.

**RAG** runs the serving stream in blocks of 12 episodes. Within a block its case store is frozen, so the block's
episodes are independent and run in parallel; after the block, every episode record is added to the store. For each
task the 8 most similar past cases are retrieved with Okapi BM25 (k1 = 1.5, b = 0.75) over the task text without its
timestamp line, and shown in the system prompt under `# Past cases`, each with its score and stream position. Test
tasks retrieve from the store but are never added to it.

## Adding a method

Subclass `servelearnbench.methods.Method`:

```python
from servelearnbench.methods import Method, EpisodeResult


class MyNotes(Method):
    name = "notes"
    learns = True          # run the serving stream
    block_size = 12        # serving episodes between updates

    def setup(self, bundle):
        super().setup(bundle)
        self.notes = ""

    def context(self, task, phase, window):
        """Text appended to the system prompt for this episode."""
        return f"# Notes\n{self.notes}" if self.notes else ""

    def update(self, results: list[EpisodeResult]):
        """Called after each block of serving episodes."""
        for r in results:
            ...   # r.instruction, r.timestamp, r.reward, r.record
```

Run it from Python:

```python
from servelearnbench.llm import ModelConfig
from servelearnbench.runner import run

run("banking_l2", MyNotes(), ModelConfig(model="MODEL", base_url="URL"), "results/notes_banking_l2.jsonl")
```

or register it in `servelearnbench/methods/__init__.py` to use it with `slb run`.

Rules that keep results comparable:

- Learn only from `update()`. Do not read `task.z`, `task.actions`, `task.answer` or the scenario bundle's
  `rules`; they are the evaluator's.
- `context()` may depend on the task text but not on `phase` or `window` (they are passed for bookkeeping; only
  Oracle uses the window).
- If the method calls a model outside the agent loop, report that usage with your results.
