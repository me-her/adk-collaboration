"""Exercise the proposed native collaboration API through an installed wheel.

No credentials or provider calls are required. Four actual ADK Agents use local
scripted models; this verifies execution and cleanup, not provider billing.
"""

import asyncio
import json
from collections.abc import AsyncGenerator

from google.adk import Agent, Context, Workflow
from google.adk.agents import llm_agent
from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.runners import InMemoryRunner
from google.adk.workflow import START, node
from google.genai import types
from pydantic import BaseModel


async def check() -> dict:
    entered: asyncio.Queue[str] = asyncio.Queue()
    release = asyncio.Event()
    rejected = asyncio.Event()
    never = asyncio.Event()
    closed = set()
    requests = {}
    emitted = []
    observations = {}

    class Answer(BaseModel):
        answer: str

    class ScriptedModel(BaseLlm):
        model: str = "local-scripted"
        worker: str

        async def generate_content_async(
            self, llm_request: LlmRequest, stream: bool = False
        ) -> AsyncGenerator[LlmResponse, None]:
            requests[self.worker] = llm_request.model_copy(deep=True)
            entered.put_nowait(self.worker)
            try:
                await release.wait()
                if self.worker == "winner":
                    await rejected.wait()
                    answer = "accepted"
                elif self.worker == "early":
                    answer = "rejected"
                else:
                    await never.wait()
                    raise AssertionError("Cancelled worker resumed")
                emitted.append(self.worker)
                yield LlmResponse(
                    content=types.Content(
                        role="model", parts=[types.Part(text=json.dumps({"answer": answer}))]
                    )
                )
            finally:
                closed.add(self.worker)

    agents = [
        Agent(name=name, model=ScriptedModel(worker=name), output_schema=Answer)
        for name in ("early", "winner", "loser_a", "loser_b")
    ]

    def accept(output):
        if output["answer"] == "rejected":
            rejected.set()
            return False
        return output["answer"] == "accepted"

    @node(rerun_on_resume=True)
    async def coordinate(ctx: Context) -> dict:
        async with ctx.parallel_tasks(max_parallel_workers=4) as tasks:
            handles = [tasks.spawn(agent, "Solve the problem") for agent in agents]
            receipt = handles[1].send_message("Include independently checked evidence.")
            started = [await entered.get() for _ in handles]
            assert len(set(started)) == 4
            assert all(handle.status == "running" for handle in handles)
            observations["coordinator_work"] = sum(range(10))
            release.set()
            winner = await tasks.first(accept=accept)
            assert winner is handles[1]
            assert closed == set(started), closed
            assert [handle.status for handle in handles[2:]] == ["cancelled", "cancelled"]
            await tasks.merge_result(winner, keys=["answer"])
            observations["receipt"] = handles[1].message_receipt(receipt["id"])["status"]
            observations["statuses"] = [handle.status for handle in handles]
            return winner.output

    workflow = Workflow(name="collaboration_wheel_check", edges=[(START, coordinate)])
    runner = InMemoryRunner(node=workflow)
    session = await runner.session_service.create_session(app_name=runner.app_name, user_id="test")
    events = []
    try:
        async for event in runner.run_async(
            user_id="test",
            session_id=session.id,
            new_message=types.Content(role="user", parts=[types.Part(text="start")]),
        ):
            events.append(event)
        stored = await runner.session_service.get_session(
            app_name=runner.app_name, user_id="test", session_id=session.id
        )
        assert stored.state["answer"] == "accepted"
    finally:
        await runner.close()

    assert observations["receipt"] == "delivered"
    assert observations["coordinator_work"] == 45
    assert emitted == ["early", "winner"]
    assert len(requests) == 4
    assert any(
        "independently checked evidence" in (part.text or "")
        for content in requests["winner"].contents
        for part in content.parts or []
    )
    assert any("parallel_task" in (event.custom_metadata or {}) for event in events)
    return {
        **observations,
        "model_calls": len(requests),
        "model_answers": emitted,
        "closed_workers": sorted(closed),
        "merged_answer": stored.state["answer"],
    }


async def main() -> None:
    assert "/site-packages/" in llm_agent.__file__, llm_agent.__file__
    result = await asyncio.wait_for(check(), timeout=10)
    print(json.dumps({"installed_module": llm_agent.__file__, **result}, indent=2))


asyncio.run(main())
