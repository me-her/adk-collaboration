"""Exercise Agent worker limits through an installed ADK wheel and Runner.

Requires the proposed patch from google/adk-python issue #7641. This script
is separate from the extension's tests, which work with published ADK 2.12.0.
Build the patched ADK wheel, install it in a clean environment, and run this
script with that environment's Python. No credentials or provider calls.
"""

import asyncio
import json
from collections.abc import AsyncGenerator

from google.adk import Agent, Workflow
from google.adk.agents import llm_agent
from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.runners import InMemoryRunner
from google.adk.workflow import START
from google.genai import types


async def check(limit: int | None) -> dict:
    items = ["alpha", "beta", "gamma", "delta"]
    expected = limit or len(items)
    barrier = asyncio.Barrier(expected)
    active = 0
    peak = 0

    class EchoModel(BaseLlm):
        model: str = "local-echo"

        async def generate_content_async(
            self, llm_request: LlmRequest, stream: bool = False
        ) -> AsyncGenerator[LlmResponse, None]:
            nonlocal active, peak
            active += 1
            peak = max(peak, active)
            try:
                assert active <= expected
                await asyncio.wait_for(barrier.wait(), timeout=5)
                text = llm_request.contents[-1].parts[0].text
                yield LlmResponse(
                    content=types.Content(
                        role="model", parts=[types.Part(text=f"processed:{text}")]
                    )
                )
            finally:
                active -= 1

    captured = []

    def produce() -> list[str]:
        return items

    def capture(node_input: list[str]) -> list[str]:
        captured.append(node_input)
        return node_input

    settings = {} if limit is None else {"max_parallel_workers": limit}
    worker = Agent(name="worker", model=EchoModel(), parallel_worker=True, **settings)
    workflow = Workflow(name="wheel_check", edges=[(START, produce, worker, capture)])
    runner = InMemoryRunner(node=workflow)
    session = await runner.session_service.create_session(app_name=runner.app_name, user_id="test")
    async for _ in runner.run_async(
        user_id="test",
        session_id=session.id,
        new_message=types.Content(role="user", parts=[types.Part(text="start")]),
    ):
        pass
    await runner.close()
    assert peak == expected, (peak, expected)
    assert active == 0
    assert captured == [[f"processed:{item}" for item in items]], captured
    return {"limit": limit, "peak_active": peak, "ordered_results": captured[0]}


async def main() -> None:
    assert "/site-packages/" in llm_agent.__file__, llm_agent.__file__
    results = [await check(limit) for limit in (1, 2, None)]
    print(json.dumps({"installed_module": llm_agent.__file__, "cases": results}, indent=2))


asyncio.run(main())
