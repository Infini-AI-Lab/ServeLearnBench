# Protocol

## The agent

Every method uses the same acting agent: a loop over the provider's native tool-calling API.

- The system prompt is the domain's prompt (identity, a note that actual practice may differ from the documents
  in either direction, the document index, tool signatures, protocol rules), followed by whatever the method adds.
- The user message is the task, prefixed with `Current timestamp: t`.
- One tool call per turn. A turn with several calls executes none of them; a turn with no call gets a corrective
  message. The episode ends when the agent calls `finish`.
- The model's reasoning is passed back in the conversation on later turns.
- Budgets per episode: 50 tool calls, 100 turns, 256k output tokens. Every task is attempted once.

## What a method may see

After each **serving** task, a method may learn from the episode record: the instruction as the agent received it,
every step (reasoning, reply, action, observation), and the outcome line `Score: n/100.`
It never receives the hidden policy, the correct answer or action, the task's category, window boundaries, or any
explanation. **Test** tasks return nothing and never update the method.

The timestamp is the task's position in the serving stream (1, 2, 3, … continuing across windows); a window's test
tasks carry the position of that window's last serving task. It lets a method order its experience in time but does
not reveal where windows begin or end.

Oracle is the one exception by design: it is told the active hidden policy of each window and serves as an upper
reference.

## Scoring

- Retail and Banking rewards are binary. Pitch rewards are the rubric score divided by 100.
- A **setting score** is the mean reward × 100 over one scenario's held-out test tasks of one category
  (Hidden-Dependent or Fully Specified). In Retail and Banking, a task whose rule has returned to its documented
  behavior counts as Fully Specified; every Pitch task counts as Hidden-Dependent.
- **Avg. H** is the unweighted mean of the nine Hidden scores; **Avg. F** the unweighted mean of the six Retail and
  Banking Fully Specified scores. Every setting counts once, whatever its number of tasks.
- Costs are totals over tasks: per-task cost is total spend divided by the number of tasks.
