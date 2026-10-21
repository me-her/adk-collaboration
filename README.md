# ADK collaboration experiment

An experimental Python extension for supervising background Google ADK agents.
The coordinator receives task handles immediately, keeps working, observes
progress, sends queued messages, cancels individual workers, and explicitly
merges proposed state updates. The implementation uses public ADK Runner,
session, plugin, and tool APIs. It does not require a patched ADK or replace its
model/provider layer.

This is an independent experiment, not a Google release or a production-ready
durable runtime. It complements upstream
[AgentDispatcherToolset #6592](https://github.com/google/adk-python/pull/6592) and
[background event observation #7094](https://github.com/google/adk-python/issues/7094).
ADK already supports parallel and dynamic workflows; this project explores the
task lifecycle around that execution.

## Run it

Python 3.11 or later is required. The tests and demo use deterministic local
models, make no provider calls, and need no credentials.

```sh
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e '.[test]'
.venv/bin/python -m pytest -q
.venv/bin/python examples/team_demo.py
.venv/bin/python benchmarks/worker_capacity.py
```

The demo runs a real ADK coordinator whose model calls the team tools. It starts
four children, continues while all four are blocked on simulated I/O, reads live
progress, steers two workers, cancels one, collects the first result before the
remaining workers finish, and merges selected state. A conflicting two-key
proposal is rejected without applying either key. Assertions and full observer
traces are written to `evidence/team-demo.json`.

The benchmark executes identical child work with capacities 1, 2, and 4. It
reports wall time and process CPU time, verifies overlap and output completeness,
rotates execution order, and retains raw samples. These are simulated I/O
measurements, not production LLM speedups or a Python/Rust comparison.

## Python API

Wrap a fresh-agent factory with `AdkWorker` and declare the state keys it may
propose. Each run receives a private session and a detached snapshot of team
state. Child writes do not modify the coordinator's ADK session.

```python
from google.adk import Agent
from adk_collaboration import Supervisor
from adk_collaboration.adk import AdkWorker

def researcher(context):
    return Agent(
        name="researcher",
        instruction="Research the assigned question and report your findings.",
        output_key="findings",
    )

async def research(question):
    workers = {"researcher": AdkWorker(researcher, export_keys=("findings",))}
    async with Supervisor(workers, max_workers=4) as team:
        task_id = team.spawn("researcher", question)
        # The caller can do independent work here.
        completed = await team.wait([task_id], timeout=30)
        if completed and completed[0]["status"] == "succeeded":
            team.merge(task_id, keys=["findings"])
        return team.state
```

This example uses the normal ADK model configuration and therefore needs model
access. Use `examples/team_demo.py` for a fully offline example. Exiting the
supervisor scope cancels and awaits unfinished work, including after a timeout
in the example.

For an LLM coordinator, `coordinator_tools(team, app_name=..., user_id=...,
session_id=...)` exposes `spawn_task`, `get_task`, `wait_tasks`, `send_message`,
`get_message_receipt`, `cancel_task`, `merge_result`, and `read_team_state`.
All tools verify the actual caller's parent-session identity. Create one team
per coordinator session and close it with that session's lifecycle. The model
cannot name arbitrary Python callables; it chooses from the supplied worker
registry. `worker_message_tool(context)` optionally lets a worker message a peer
using its bound task identity.

## Contract

| Area | Behavior |
| --- | --- |
| Handles | UUIDs are stable for this supervisor lifetime. Results and receipts remain available until the supervisor is discarded. |
| Lifecycle | `queued` → `running` → `succeeded` or `failed`; cancellation passes through `cancelling` until termination is confirmed. |
| Waiting | `wait(..., mode="any" or "all")` returns terminal records in supplied handle order. Timeouts and cancelled waiters do not cancel workers. |
| Capacity | `max_workers` bounds active worker tasks. `max_tasks` bounds all accepted and retained handles, including queued and terminal tasks. Admission rejects overload. |
| Messages | FIFO admission returns a receipt. Delivery occurs before the top-level worker agent's next model request and is retained in its private session. Nested agents do not consume that mailbox. |
| Final-response race | A message accepted during the final model call triggers a follow-up invocation on the same child session. Follow-ups have a finite limit. Each invocation must produce its own final result. |
| Receipts | `queued`, `delivered`, or `undelivered`. Delivered means inserted into context, not understood, acted on, or an external side effect completed. |
| Cancellation | Cancels one owned asyncio task. It does not roll back tools or stop an uncooperative thread/process. A timeout can return `cancelling`; cleanup reports incomplete termination instead of claiming success. |
| Progress | `events(after=cursor)` and `observe(after=cursor)` read a separate bounded journal. An expired cursor raises `EventGap`. Observers cannot consume or cancel worker execution. |
| Event content | ADK events are attributed to task IDs. Typed thought parts/signatures are removed. Oversized ADK events become explicit metadata-only summaries. Progress is not automatically copied into the coordinator's model history. |
| State | Only allowlisted exported keys become proposed updates. `merge` compares each selected key against the starting snapshot, rejects all writes on conflict, and makes repeated accepted merges no-ops. JSON types are compared distinctly. |
| Isolation | Every ADK worker has its own in-memory service/app/user/session. Team state is separate from ADK session state. Reserved `:` state namespaces and deletion patches are unsupported. |

The journal bounds both record count and serialized record bytes. Task input,
snapshots, results, and team state have payload limits. Message receipts are
bounded over the whole task lifetime, and pending messages have an aggregate
byte limit. These are logical data limits, not a process RSS guarantee: ADK's
private session history and model/tool implementation memory are separate.

## Deliberate boundaries

- In-process ownership only. Handles, queued messages, and in-flight execution
  do not survive process restart. Durable sessions alone would not solve this.
- Completion emits an event; the host chooses whether to wake a coordinator.
  The extension does not autonomously start another parent model turn.
- Queued steering targets the worker's top-level model agent. A workflow-only
  root has no model boundary at which to consume those messages.
- Child workers can message peers but cannot recursively spawn through their
  worker capability. That avoids parents holding every slot while awaiting
  queued descendants. A nested-team scheduling policy remains design work.
- A child that pauses for human input or an unresolved long-running tool fails
  explicitly. This prototype does not claim resumable human-in-the-loop support.
- Concurrent tools inside one worker are still governed by ADK; `max_workers`
  is a worker limit, not a global model-call or tool-call rate limiter.
- State merging is synchronous and atomic on this supervisor's single event
  loop. It is not a distributed database transaction or an automatic write to
  the parent ADK session. Factories must not introduce mutable global business
  state or share unsafe tools between children.
- Tests establish runtime behavior with scripted decisions. They do not
  establish reasoning quality, task success on real coding benchmarks, provider
  compatibility beyond ADK's own adapters, or production reliability.

## Upstream path

The small `Agent(parallel_worker=True, max_parallel_workers=...)` change lives
in [upstream issue #7641](https://github.com/google/adk-python/issues/7641).
[RFC #7642](https://github.com/google/adk-python/issues/7642) requests feedback
on the broader contract and links the executable evidence. This experiment asks for agreement on lifecycle
semantics before proposing core APIs. See [the design proposal](docs/proposal.md)
for acceptance criteria and links to existing work. Possible focused pieces
include progress observation, cancellation states, or acknowledged messaging,
depending on maintainer direction.

The separate [installed-wheel smoke test](upstream/agent_worker_smoke.py)
checks the proposed agent-worker limit. It requires the patched ADK wheel;
the collaboration extension itself runs against the published package.
