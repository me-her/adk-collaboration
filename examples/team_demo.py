"""Run a real ADK coordinator and four children with deterministic local models.

Run: python examples/team_demo.py
No provider calls or credentials. Gates represent external I/O and make the
assertions independent of model timing. Scripted decisions test the runtime
contract, not model reasoning quality.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

from google.adk import Agent
from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_response import LlmResponse
from google.adk.runners import InMemoryRunner
from google.adk.tools.tool_context import ToolContext
from google.genai import types
from pydantic import PrivateAttr

from adk_collaboration import Supervisor
from adk_collaboration.adk import AdkWorker, coordinator_tools


class LocalModel(BaseLlm):
    decide: Callable
    _step: int = PrivateAttr(default=0)

    async def generate_content_async(self, llm_request, stream=False):
        parts = await self.decide(self._step, llm_request)
        self._step += 1
        yield LlmResponse(content=types.Content(role="model", parts=parts))


def call(name: str, **args: Any) -> types.Part:
    return types.Part.from_function_call(name=name, args=args)


def answer(text: str) -> list[types.Part]:
    return [types.Part.from_text(text=text)]


async def demonstrate() -> dict[str, Any]:
    names = ["researcher", "critic", "writer", "slow"]
    started = {name: asyncio.Event() for name in names}
    release = {name: asyncio.Event() for name in names}
    active = 0
    peak = 0
    messages_seen = {}
    observed = {}

    def factory(name: str):
        def build(context):
            async def do_work() -> str:
                nonlocal active, peak
                active += 1
                peak = max(peak, active)
                context.emit("work_started", {"worker": name})
                started[name].set()
                try:
                    await release[name].wait()
                    return "I/O completed"
                finally:
                    active -= 1

            async def propose(tool_context: ToolContext) -> str:
                if name == "researcher":
                    tool_context.state["decision"] = "use primary evidence"
                elif name == "critic":
                    tool_context.state["decision"] = "different recommendation"
                    tool_context.state["review"] = "review complete"
                elif name == "writer":
                    tool_context.state["draft"] = "draft using delivered evidence"
                return "proposed privately"

            async def decide(step, request):
                if step == 0:
                    return [call("do_work")]
                if step == 1:
                    contents = json.dumps(
                        [content.model_dump(mode="json") for content in request.contents]
                    )
                    if name == "researcher":
                        assert "prioritize primary sources" in contents
                        messages_seen[name] = True
                    if name == "writer":
                        assert "incorporate primary evidence" in contents
                        messages_seen[name] = True
                    return [call("propose")]
                assert step == 2
                return answer(f"{name} completed")

            return Agent(
                name=name,
                model=LocalModel(model="local-worker", decide=decide),
                tools=[do_work, propose],
            )

        return build

    workers = {
        name: AdkWorker(factory(name), export_keys=("decision", "review", "draft"))
        for name in names
    }
    async with Supervisor(workers, initial_state={"decision": "pending"}, max_workers=4) as team:
        handles = {}

        async def root_decide(step, request):
            for content in request.contents:
                for part in content.parts or []:
                    response = part.function_response
                    if response and response.name == "spawn_task":
                        payload = response.response
                        assert "task_id" in payload, payload
                        handles[payload["worker"]] = payload["task_id"]
            if step == 0:
                return [
                    call("spawn_task", worker=name, request="Work on your assigned role.")
                    for name in names
                ]
            if step == 1:
                await asyncio.wait_for(
                    asyncio.gather(*(event.wait() for event in started.values())), 5
                )
                assert active == 4
                observed["parent_continued_with_four_live_children"] = True
                observed["progress_visible_before_completion"] = any(
                    event["kind"] == "work_started" for event in team.events()["events"]
                )
                return [
                    call(
                        "send_message",
                        task_id=handles["researcher"],
                        text="prioritize primary sources",
                    ),
                    call("cancel_task", task_id=handles["slow"]),
                ]
            if step == 2:
                assert team.get(handles["slow"])["status"] == "cancelled"
                # Test harness releases simulated I/O; this is not a runtime API.
                release["researcher"].set()
                return [
                    call(
                        "wait_tasks",
                        task_ids=[handles[name] for name in names[:3]],
                        mode="any",
                        timeout_seconds=5,
                    )
                ]
            if step == 3:
                assert team.get(handles["researcher"])["status"] == "succeeded"
                assert team.get(handles["writer"])["status"] == "running"
                observed["first_completion_observed_before_siblings"] = True
                return [
                    call("merge_result", task_id=handles["researcher"], keys=["decision"]),
                    call(
                        "send_message",
                        task_id=handles["writer"],
                        text="incorporate primary evidence",
                    ),
                ]
            if step == 4:
                release["critic"].set()
                release["writer"].set()
                return [
                    call(
                        "wait_tasks", task_ids=list(handles.values()), mode="all", timeout_seconds=5
                    )
                ]
            if step == 5:
                return [
                    call("merge_result", task_id=handles["critic"], keys=["decision", "review"]),
                    call("merge_result", task_id=handles["writer"], keys=["draft"]),
                ]
            if step == 6:
                assert team.state == {
                    "decision": "use primary evidence",
                    "draft": "draft using delivered evidence",
                }
                observed["conflict_rejected_without_partial_write"] = "review" not in team.state
                return [call("read_team_state")]
            assert step == 7
            return answer(
                "Team complete: evidence accepted, draft merged, conflicting review held."
            )

        tools = coordinator_tools(team, app_name="demo", user_id="demo_user", session_id="parent")
        root = Agent(
            name="coordinator",
            tools=tools,
            model=LocalModel(model="local-coordinator", decide=root_decide),
        )
        async with InMemoryRunner(agent=root, app_name="demo") as runner:
            session = await runner.session_service.create_session(
                app_name="demo", user_id="demo_user", session_id="parent"
            )
            async with asyncio.timeout(20):
                async for _ in runner.run_async(
                    user_id=session.user_id,
                    session_id=session.id,
                    new_message=types.Content(role="user", parts=answer("Collaborate.")),
                ):
                    pass
        assert peak == 4 and active == 0
        assert messages_seen == {"researcher": True, "writer": True}
        assert all(observed.values())
        records = [team.get(handles[name]) for name in names]
        assert [record["status"] for record in records] == [
            "succeeded",
            "succeeded",
            "succeeded",
            "cancelled",
        ]
        return {
            "provider_calls": 0,
            "peak_active_workers": peak,
            "assertions": observed,
            "messages_seen_in_actual_model_requests": messages_seen,
            "tasks": records,
            "merged_state": team.state,
            "observer_journal": team.events(),
        }


if __name__ == "__main__":
    logging.basicConfig(level=logging.ERROR)
    result = asyncio.run(demonstrate())
    output = Path(__file__).resolve().parents[1] / "evidence" / "team-demo.json"
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {key: value for key, value in result.items() if key != "observer_journal"}, indent=2
        )
    )
    print(f"Assertions passed; full progress trace: {output}")
