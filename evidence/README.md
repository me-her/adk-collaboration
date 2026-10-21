# Reproduction evidence

The recorded demonstration and capacity measurements use Python 3.12.12 on
macOS arm64 with the published `google-adk==2.12.0` wheel, rather than the
concurrency-limit patch being proposed separately. All model decisions are
scripted and no provider requests are made.

- `team-demo.json`: four real ADK worker runners controlled through a real ADK
  coordinator's tools, including progress, queued messages, cancellation,
  first-result collection, and explicit conflict-checked merges.
- `capacity.json`: nine measured repetitions per capacity, one excluded warm-up,
  rotated order, per-run wall/CPU measurements and overlap intervals. Each trial
  executes four identical 100 ms simulated I/O tasks with full output checks.

Regenerate both from the repository root:

```sh
python examples/team_demo.py
python benchmarks/worker_capacity.py
```

The capacity run recorded median wall times of about 421 ms, 213 ms, and 110 ms
for capacities 1, 2, and 4. Those measurements establish concurrency on this
synthetic workload. They do not predict provider latency, model reasoning
quality, production throughput, or the benefit of another implementation language.

The conformance suite has 48 passing cases at the initial publication. It
covers the runtime contract, real ADK integration, and regressions discovered
during review. The CI matrix reruns tests and the executable demo independently.
