"""A minimal learning method: keep the last few failed instructions as notes.

    python examples/custom_method.py --model MODEL --scenario banking_l1
"""

import argparse

from servelearnbench.llm import ModelConfig
from servelearnbench.methods import EpisodeResult, Method
from servelearnbench.protocol import strip_timestamp
from servelearnbench.runner import run


class RecentFailures(Method):
    name = "recent_failures"
    learns = True

    def setup(self, bundle):
        super().setup(bundle)
        self.failures: list[str] = []

    def context(self, task, phase, window):
        if not self.failures:
            return ""
        lines = "\n".join(f"- {f}" for f in self.failures[-5:])
        return f"# Recent requests you got wrong\n{lines}"

    def update(self, results: list[EpisodeResult]):
        self.failures += [strip_timestamp(r.instruction)[:300] for r in results if r.reward < 1]


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--base-url", default="https://api.fireworks.ai/inference/v1")
    p.add_argument("--scenario", default="banking_l1")
    a = p.parse_args()
    run(a.scenario, RecentFailures(), ModelConfig(model=a.model, base_url=a.base_url),
        f"results/{a.scenario}_recent_failures.jsonl")
