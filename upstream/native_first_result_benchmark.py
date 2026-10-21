"""Offline comparison of native gather versus first-acceptable-result policy.

Run against an installed wheel containing Context.parallel_tasks, for example:
  uv run --no-project --python /tmp/adk-agent-worker-wheel-venv/bin/python \
    upstream/native_first_result_benchmark.py

Both policies start the same four real ADK Agents and four local model calls.
Fixed async sleeps simulate I/O; the 20 ms answer is rejected, the 60 ms answer
is acceptable. first() cancels the remaining work; gather() completes it all.
This is a selection-policy comparison, not a Python/Rust, token, or billing test.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib.metadata
import json
import logging
import platform
import statistics
import sys
import time
from collections.abc import AsyncGenerator
from datetime import UTC, datetime
from pathlib import Path

from google.adk import Agent, Context, Workflow
from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.runners import InMemoryRunner
from google.adk.workflow import START, _parallel_tasks, node
from google.genai import types
from pydantic import BaseModel

WORKERS = (("reject_20", 20), ("accept_60", 60), ("accept_90", 90), ("accept_120", 120))
MODES = ("gather", "first")


class Answer(BaseModel):
    worker: str
    acceptable: bool


async def trial(mode: str, worker_order: list[tuple[str, int]]) -> dict:
    wall_start, cpu_start = time.perf_counter(), time.process_time()
    entered: asyncio.Queue[str] = asyncio.Queue()
    release = asyncio.Event()
    activity: dict[str, dict] = {}
    emitted: list[str] = []
    closed: set[str] = set()
    observations = {}
    active = peak = 0

    def elapsed_ms() -> float:
        return (time.perf_counter() - wall_start) * 1000

    class ScriptedModel(BaseLlm):
        model: str = "offline-simulated-io"
        worker: str
        delay_ms: int

        async def generate_content_async(
            self, llm_request: LlmRequest, stream: bool = False
        ) -> AsyncGenerator[LlmResponse, None]:
            nonlocal active, peak
            assert self.worker not in activity, "Unexpected extra model request"
            activity[self.worker] = {"request_entered_ms": elapsed_ms()}
            active += 1
            peak = max(peak, active)
            entered.put_nowait(self.worker)
            try:
                await release.wait()
                activity[self.worker]["simulated_io_started_ms"] = elapsed_ms()
                await asyncio.sleep(self.delay_ms / 1000)
                emitted.append(self.worker)
                activity[self.worker]["answer_emitted_ms"] = elapsed_ms()
                answer = {"worker": self.worker, "acceptable": self.delay_ms != 20}
                yield LlmResponse(
                    content=types.Content(role="model", parts=[types.Part(text=json.dumps(answer))])
                )
            finally:
                active -= 1
                closed.add(self.worker)
                activity[self.worker]["generator_closed_ms"] = elapsed_ms()

    agents = [
        Agent(
            name=name,
            model=ScriptedModel(worker=name, delay_ms=delay),
            output_schema=Answer,
        )
        for name, delay in worker_order
    ]
    expected_names = {name for name, _ in WORKERS}

    @node(rerun_on_resume=True)
    async def coordinate(ctx: Context) -> dict:
        async with ctx.parallel_tasks(max_parallel_workers=4) as group:
            handles = [group.spawn(agent, "Evaluate the same scripted case") for agent in agents]
            assert {await entered.get() for _ in handles} == expected_names
            assert all(handle.status == "running" for handle in handles)
            selection_start = time.perf_counter()
            release.set()
            if mode == "gather":
                outputs = await group.gather()
                assert outputs == [
                    {"worker": name, "acceptable": delay != 20} for name, delay in worker_order
                ]
                result = {"all_outputs": outputs}
            else:
                winner = await group.first(accept=lambda output: output["acceptable"])
                assert winner.output == {"worker": "accept_60", "acceptable": True}
                result = winner.output
            observations["selection_and_drain_ms"] = (time.perf_counter() - selection_start) * 1000
            observations["statuses"] = {handle.name: handle.status for handle in handles}
            assert closed == expected_names, "Policy returned before model cleanup"
            return result

    runner = InMemoryRunner(
        node=Workflow(name="native_policy_benchmark", edges=[(START, coordinate)])
    )
    try:
        session = await runner.session_service.create_session(
            app_name=runner.app_name, user_id="benchmark"
        )
        async for _ in runner.run_async(
            user_id="benchmark",
            session_id=session.id,
            new_message=types.Content(role="user", parts=[types.Part(text="start")]),
        ):
            pass
    finally:
        await runner.close()
    wall_ms, cpu_ms = elapsed_ms(), (time.process_time() - cpu_start) * 1000
    expected_emitted = expected_names if mode == "gather" else {"reject_20", "accept_60"}
    assert set(emitted) == expected_emitted and len(emitted) == len(expected_emitted)
    assert set(activity) == closed == expected_names and active == 0 and peak == 4
    assert observations["statuses"] == {
        name: "completed" if name in expected_emitted else "cancelled" for name in expected_names
    }
    return {
        "mode": mode,
        "worker_order": [name for name, _ in worker_order],
        "wall_ms": wall_ms,
        "cpu_ms": cpu_ms,
        "model_requests_started": len(activity),
        "model_answers_emitted": emitted,
        "peak_active_models": peak,
        "closed_workers": sorted(closed),
        **observations,
        "worker_activity_ms": activity,
    }


async def main(repeats: int, output: Path) -> None:
    installed_module = Path(_parallel_tasks.__file__).resolve()
    assert "site-packages" in installed_module.parts, "Use the installed patched ADK wheel"
    for mode in MODES:
        await asyncio.wait_for(trial(mode, list(WORKERS)), 5)
    runs = []
    for repetition in range(repeats):
        offset = repetition % len(WORKERS)
        workers = list(WORKERS[offset:] + WORKERS[:offset])
        modes = MODES if repetition % 2 == 0 else MODES[::-1]
        for mode in modes:
            measured = await asyncio.wait_for(trial(mode, workers), 5)
            runs.append({"repetition": repetition + 1, **measured})
    summary = {}
    for mode in MODES:
        samples = [run for run in runs if run["mode"] == mode]
        summary[mode] = {
            **{
                f"median_{metric}": statistics.median(run[metric] for run in samples)
                for metric in ("wall_ms", "cpu_ms", "selection_and_drain_ms")
            },
            "min_wall_ms": min(run["wall_ms"] for run in samples),
            "max_wall_ms": max(run["wall_ms"] for run in samples),
            "requests_started_per_trial": 4,
            "answers_emitted_per_trial": 4 if mode == "gather" else 2,
            "cancelled_workers_per_trial": 0 if mode == "gather" else 2,
        }
    result = {
        "metadata": {
            "recorded_at_utc": datetime.now(UTC).isoformat(),
            "python": platform.python_version(),
            "python_executable": sys.executable,
            "platform": platform.platform(),
            "adk_version": importlib.metadata.version("google-adk"),
            "installed_module": str(installed_module),
            "installed_module_sha256": hashlib.sha256(installed_module.read_bytes()).hexdigest(),
            "benchmark_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "repetitions_per_mode": repeats,
            "excluded_warmups_per_mode": 1,
            "worker_delay_ms": dict(WORKERS),
            "ordering": "Worker spawn order rotates; policy order alternates each repetition.",
            "wall_timing": "Fresh Agent/Runner/session creation through runner.close; excludes imports/process startup.",
            "selection_timing": "Common release of all four model requests through policy result and worker cleanup.",
            "provider_calls": 0,
            "interpretation": "Selection policy: gather completes all work; first rejects the 20 ms answer, accepts the 60 ms answer, and cancels 90/120 ms losers.",
            "limitations": "Scripted local models and simulated I/O only. Both modes start four requests. No Python/Rust, reasoning quality, production throughput, token, or provider-billing claim.",
        },
        "summary": summary,
        "runs": runs,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print(f"Raw evidence: {output}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.ERROR)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=12)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "evidence" / "native-first-result.json",
    )
    args = parser.parse_args()
    if not 2 <= args.repeats <= 1000:
        parser.error("Use between 2 and 1000 repetitions per mode")
    asyncio.run(main(args.repeats, args.output))
