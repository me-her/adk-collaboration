"""Compare identical ADK child work at capacities 1, 2, and 4, without API calls."""

from __future__ import annotations

import argparse
import asyncio
import importlib.metadata
import json
import logging
import platform
import statistics
import time
from pathlib import Path

from google.adk import Agent
from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_response import LlmResponse
from google.genai import types
from pydantic import PrivateAttr

from adk_collaboration import Supervisor
from adk_collaboration.adk import AdkWorker


class Model(BaseLlm):
    _index: int = PrivateAttr(default=0)

    async def generate_content_async(self, llm_request, stream=False):
        parts = (
            [types.Part.from_function_call(name="io_work", args={})]
            if self._index == 0
            else [types.Part.from_text(text="completed")]
        )
        self._index += 1
        yield LlmResponse(content=types.Content(role="model", parts=parts))


async def trial(capacity: int, delay: float) -> dict:
    active = 0
    peak = 0
    intervals = []

    def factory(context):
        async def io_work() -> str:
            nonlocal active, peak
            start = time.perf_counter()
            active += 1
            peak = max(peak, active)
            try:
                await asyncio.sleep(delay)
                return "I/O complete"
            finally:
                active -= 1
                intervals.append(
                    {"task_id": context.task_id, "start": start, "end": time.perf_counter()}
                )

        return Agent(name="worker", model=Model(model="scripted-local"), tools=[io_work])

    cpu_start, wall_start = time.process_time(), time.perf_counter()
    async with Supervisor({"worker": AdkWorker(factory)}, max_workers=capacity) as team:
        handles = [team.spawn("worker", "Do the same I/O work.") for _ in range(4)]
        records = await team.wait(handles, timeout=10)
        assert len(records) == 4 and all(record["status"] == "succeeded" for record in records)
        assert all(record["result"]["output"] == "completed" for record in records)
    wall_ms = (time.perf_counter() - wall_start) * 1000
    cpu_ms = (time.process_time() - cpu_start) * 1000
    assert active == 0 and peak == capacity
    # Explicitly prove overlap from independently recorded I/O intervals.
    overlap = max(
        sum(other["start"] <= interval["start"] < other["end"] for other in intervals)
        for interval in intervals
    )
    assert overlap == capacity
    return {
        "wall_ms": wall_ms,
        "cpu_ms": cpu_ms,
        "peak": peak,
        "overlap": overlap,
        "intervals_ms": [
            {
                "task_id": interval["task_id"],
                "start": (interval["start"] - wall_start) * 1000,
                "end": (interval["end"] - wall_start) * 1000,
            }
            for interval in intervals
        ],
    }


async def main(repeats: int, delay: float, output: Path) -> None:
    capacities = [1, 2, 4]
    for capacity in capacities:
        await trial(capacity, delay)
    runs = {str(capacity): [] for capacity in capacities}
    for iteration in range(repeats):
        rotated = capacities[iteration % 3 :] + capacities[: iteration % 3]
        for capacity in rotated:
            runs[str(capacity)].append(await trial(capacity, delay))
    result = {
        "metadata": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "adk": importlib.metadata.version("google-adk"),
            "extension": importlib.metadata.version("adk-collaboration"),
            "repeats": repeats,
            "simulated_io_ms": delay * 1000,
            "tasks_per_trial": 4,
            "provider_calls": 0,
            "includes": "Supervisor, fresh child agents/sessions, model/tool dispatch, cleanup",
            "excludes": "Process startup and imports; one warm-up per capacity",
            "limitations": "Scripted models and async sleeps; no reasoning quality, real provider, "
            "Rust, RSS, or production throughput claims",
        },
        "summary": {
            capacity: {
                "median_wall_ms": statistics.median(run["wall_ms"] for run in samples),
                "median_cpu_ms": statistics.median(run["cpu_ms"] for run in samples),
                "min_wall_ms": min(run["wall_ms"] for run in samples),
                "max_wall_ms": max(run["wall_ms"] for run in samples),
            }
            for capacity, samples in runs.items()
        },
        "runs": runs,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["summary"], indent=2))
    print(f"Raw measurements: {output}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.ERROR)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=9)
    parser.add_argument("--delay", type=float, default=0.1)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "evidence" / "capacity.json",
    )
    args = parser.parse_args()
    if args.repeats < 1 or not 0 < args.delay < 2:
        parser.error("Use positive repeats and 0 < delay < 2 seconds")
    asyncio.run(main(args.repeats, args.delay, args.output))
