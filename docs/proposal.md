This design is being discussed in [upstream RFC #7642](https://github.com/google/adk-python/issues/7642).

### Is your feature request related to a specific problem?

ADK 2.x already supports runtime agent dispatch through `Workflow` and
`ctx.run_node()`, including branch isolation and event streaming, as documented
in the resolution of [#4759](https://github.com/google/adk-python/issues/4759). I want to build on those primitives with a reusable
task lifecycle contract for coordinators that continue working while children
run, react to individual completions, steer a child, and explicitly accept its
results.

The missing application-level contract is a stable handle for each submitted
task, observable progress, selective cancellation, acknowledged queued
messages, and explicit merging of a child's proposed state changes. A custom
`asyncio` wrapper can implement these, but applications otherwise have to agree
on their own behavior for cancellation races, follow-ups, and conflicting
writes.

This proposal complements existing work:

- [#6592](https://github.com/google/adk-python/pull/6592) proposes `AgentDispatcherToolset` with background dispatch, persistent
  child sessions, poll/await, callbacks, and follow-ups. The current
  `message_agent` waits for an active task before starting the follow-up. This
  RFC asks for nonblocking message admission with explicit delivery status and
  a documented safe execution boundary.
- [#7094](https://github.com/google/adk-python/issues/7094) requests the child's live intermediate events for external observers,
  separate from the final result entering the parent's context. That separation
  is part of this contract; this is not a competing event-only API request.
- [#6214](https://github.com/google/adk-python/issues/6214) proposes GroupChat and dialogue scheduling. Task handles and attributed
  events can support its fan-out/join use cases without defining speaker
  selection or a shared conversation-history policy here.

### Describe the Solution You'd Like

Agree on the following semantics before choosing the final API surface. An
experimental extension can demonstrate the contract on public ADK Runner APIs
while maintaining a small upstream review scope.

1. **Task handles and lifecycle.** Submission returns a stable task ID without
   waiting for the child to finish. A handle is scoped to its coordinator,
   user, and parent session. Follow-up turns retain the child session identity;
   each run/message has a separate correlation ID. Expose queued, running,
   completed, failed, and cancelled outcomes. Waiting supports the first of a
   selected group of tasks; a timeout leaves their execution untouched.
2. **Bounded execution.** A coordinator has an explicit maximum active worker
   count and finite admission capacity. Reject overload clearly; never silently
   create unlimited pending work. Define the limit's scope precisely and do
   not claim a global nested-team budget unless all nested work shares it.
3. **Progress.** External observers receive attributed child events as they are
   produced, with task/run IDs and a monotonic cursor. They can resume reading
   after a cursor. Retention limits and stale cursors must be explicit so event
   loss is detectable. Observing progress does not inject every child event
   into the parent's model history or publish private chain-of-thought.
4. **Selective cancellation.** Cancelling one queued or running task leaves
   siblings running. Cancellation is idempotent and distinguishes a request
   from confirmed termination. Cancellation stops future model/tool work at a
   cooperative boundary; it does not claim to undo a completed external side
   effect. Shutdown waits for owned tasks and closes owned resources.
5. **Queued messages.** Sending to a running child returns an admission receipt
   immediately, with a message ID. Messages are FIFO and are processed only at
   a defined safe boundary; never mutate an in-flight model request. Admission
   and delivery are separate states. Cancellation or failure reports accepted
   but undelivered messages rather than silently losing them. A follow-up uses
   the existing child session and serializes runs for that session.
6. **Explicit result/state merge.** Child results carry task identity and a
   proposed state delta. Children do not silently overwrite parent state.
   The parent chooses when and which keys to merge. Detect conflicting parent
   changes against the child's starting snapshot; support disjoint changes and
   require an explicit resolution for conflicts. Reapplying an accepted result
   is idempotent. State merging does not automatically publish child tool
   transcripts to the parent's model context.

### Impact on your work

I am building an orchestrator that starts several research or implementation
tasks, keeps working, consumes the first useful result, redirects unfinished
tasks, and integrates selected outputs. I want applications to share reliable
lifecycle semantics rather than each maintaining a separate task registry and
message/state protocol. No fixed release deadline.

### Willingness to contribute

Yes. This repository contains an executable extension and reproducible
conformance tests. I would like to contribute small pieces in the maintainer-preferred
location. I am not requesting acceptance of a second core runtime or claiming
ownership of the existing proposals.

### Describe Alternatives You've Considered

- `ctx.run_node()` plus application `asyncio` tasks: the appropriate execution
  primitive, but callers still write the lifecycle, message and merge contract.
- Extending [#6592](https://github.com/google/adk-python/pull/6592): plausible if maintainers want a chat-oriented toolset; the
  task contract should remain usable by application code and Workflow nodes.
- A monolithic GroupChat implementation: broader than the task lifecycle and
  overlaps with [#6214](https://github.com/google/adk-python/issues/6214)'s conversational policy decisions.
- A Rust rewrite: does not establish scheduling, cancellation, or state
  semantics. First measure equivalent successful work under fixed limits;
  language-level performance changes can be evaluated separately.

### Proposed API / Implementation

API spelling is deliberately open. The conceptual operations are `spawn`,
`inspect`, `wait_any`, `events(after_cursor)`, `send`, `cancel`, and
`merge_result`. Adapters may expose these as orchestrator tools or ordinary
Python methods. Prefer reusing `Runner`, existing session services, events,
and `ctx.run_node()` rather than introducing another provider layer.

The initial experiment is in-process. A durable child session is not proof
that an in-flight task survives a process restart. Durable execution would need
leases/checkpoints, ownership transfer, message acknowledgements, replay rules,
and side-effect idempotency; those should be a separately agreed extension.

### Runnable acceptance criteria

The experiment should ship an executable `pytest` suite using scripted models,
real ADK Runners, instrumented tools, and explicit asyncio barriers. These are
acceptance criteria. The current tests and demo provide concrete evidence;
review them before interpreting a requirement as proven. Each
test must fail if the corresponding behavior is removed.

| Scenario | Required assertion |
| --- | --- |
| Four tasks with a limit of two | Submission returns before any blocked child completes; exactly two start; releasing one admits the next; all results correlate to their original IDs. |
| Parent continues | A parent checkpoint occurs while child work is still blocked; collecting the first finished task does not wait for the slowest sibling. |
| Live progress | A tool-start event can be read before the tool is released; cursors are monotonic, resuming after a cursor does not repeat earlier events, and expired cursors fail explicitly. |
| Queued steering | Two messages sent while a child is blocked are admitted promptly, retain their IDs, and are delivered FIFO at the documented boundary on the same child session. |
| Cancellation | Cancel one blocked child, observe its cleanup and terminal status, and prove a sibling still completes; repeated cancellation is harmless; undelivered messages remain inspectable. |
| Failure isolation | One child's error is observable and does not erase successful sibling results or leak unhandled task exceptions. |
| Explicit merge | Disjoint deltas merge; concurrent conflicting changes raise without partial mutation; repeated application is idempotent; no merge occurs before acceptance. |
| Isolation and cleanup | One coordinator cannot read or control another's handles; shutdown leaves no owned live tasks or unclosed Runners. |
| Observer failure | A slow, cancelled, or failing observer cannot silently alter a child's final outcome; overflow/backpressure follows the documented policy. |

A CLI demo should run the same four-worker scenario and print assertions plus
task/event IDs. A benchmark should use the same inputs, scripted delays, limits,
and output checks for serial and concurrent execution and report CPU time and
wall time separately. Simulated I/O gains must not be presented as production
model performance gains; correctness and task completion remain prerequisites.

### Additional Context / Requested Direction

Would maintainers prefer the lifecycle pieces as additions to [#6592](https://github.com/google/adk-python/pull/6592), a small
Workflow helper around `ctx.run_node()`, or an external extension until the
contract settles? Which of the existing proposals should own the event
observer interface from [#7094](https://github.com/google/adk-python/issues/7094)? I would like to contribute focused pieces in
that order and avoid duplicating implementation already in progress.
