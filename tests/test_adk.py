"""Integration checks against public ADK runners, models, tools and sessions.

No provider requests are made. Event barriers establish ordering; timeouts only
bound a broken test so a scheduling regression cannot hang the suite.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncGenerator, Awaitable, Callable
from typing import Any

import pytest
from google.adk import Agent
from google.adk.agents.base_agent import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events.event import Event
from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.runners import InMemoryRunner
from google.adk.tools.function_tool import FunctionTool
from google.adk.tools.long_running_tool import LongRunningFunctionTool
from google.adk.tools.tool_context import ToolContext
from google.genai import types
from pydantic import PrivateAttr

from adk_collaboration.adk import AdkWorker, coordinator_tools
from adk_collaboration.runtime import Supervisor, TaskResult, WorkerContext


class ScriptedModel(BaseLlm):
    """Exercise ADK's actual model boundary with deterministic responses."""

    responses: list[LlmResponse]
    before_response: Callable[[int, LlmRequest], Awaitable[None]] | None = None
    _requests: list[LlmRequest] = PrivateAttr(default_factory=list)

    @property
    def requests(self) -> list[LlmRequest]:
        return self._requests

    async def generate_content_async(
        self,
        llm_request: LlmRequest,
        stream: bool = False,
    ) -> AsyncGenerator[LlmResponse, None]:
        index = len(self._requests)
        self._requests.append(llm_request.model_copy(deep=True))
        assert index < len(self.responses), "Unexpected extra model request"
        if self.before_response:
            await self.before_response(index, llm_request)
        yield self.responses[index].model_copy(deep=True)


def answer(text: str) -> LlmResponse:
    return LlmResponse(content=types.Content(role="model", parts=[types.Part.from_text(text=text)]))


def call(name: str, **arguments: Any) -> LlmResponse:
    return LlmResponse(
        content=types.Content(
            role="model", parts=[types.Part.from_function_call(name=name, args=arguments)]
        )
    )


def mailbox_messages(request: LlmRequest) -> list[dict[str, Any]]:
    messages = []
    for content in request.contents:
        for part in content.parts or []:
            if part.text and part.text.startswith("Queued team messages."):
                messages.extend(json.loads(part.text.split("\n", 1)[1]))
    return messages


async def terminal(supervisor: Supervisor, task_id: str) -> dict[str, Any]:
    async with asyncio.timeout(5):
        results = await supervisor.wait([task_id])
    assert len(results) == 1
    return results[0]


async def test_child_state_isolated_until_selected_exports_are_merged():
    entered = asyncio.Event()
    release = asyncio.Event()
    seen: dict[str, dict[str, Any]] = {}
    identities: set[tuple[str, str, str]] = set()

    def factory(context: WorkerContext) -> Agent:
        async def write_result(tool_context: ToolContext) -> str:
            seen[context.task_id] = tool_context.state.to_dict()
            identities.add(
                (
                    tool_context.session.app_name,
                    tool_context.session.user_id,
                    tool_context.session.id,
                )
            )
            tool_context.state["exported"] = {"owner": context.task_id}
            tool_context.state["private"] = "child-only"
            if len(seen) == 2:
                entered.set()
            await release.wait()
            return "written"

        return Agent(
            name="worker",
            tools=[write_result],
            model=ScriptedModel(
                model="scripted-local",
                responses=[
                    call("write_result"),
                    answer("done"),
                ],
            ),
        )

    baseline = {"exported": {"owner": "parent"}, "untouched": [1]}
    async with Supervisor(
        {"writer": AdkWorker(factory, export_keys=("exported",))},
        initial_state=baseline,
    ) as supervisor:
        first = supervisor.spawn("writer", "first")
        second = supervisor.spawn("writer", "second")
        await asyncio.wait_for(entered.wait(), 5)
        assert seen[first] == seen[second] == baseline
        assert len(identities) == 2
        assert supervisor.state == baseline
        release.set()
        results = await asyncio.gather(terminal(supervisor, first), terminal(supervisor, second))
        for task_id, result in zip((first, second), results, strict=True):
            assert result["status"] == "succeeded", result
            assert result["result"] == {
                "output": "done",
                "updates": {"exported": {"owner": task_id}},
            }
        assert supervisor.state == baseline
        supervisor.merge(first, keys=["exported"])
        assert supervisor.state == {"exported": {"owner": first}, "untouched": [1]}


async def test_mailbox_delivers_fifo_before_next_request_and_retains_history():
    entered = asyncio.Event()
    release = asyncio.Event()

    async def block() -> str:
        entered.set()
        await release.wait()
        return "released"

    def next_step() -> str:
        return "second step"

    model = ScriptedModel(
        model="scripted-local",
        responses=[
            call("block"),
            call("next_step"),
            answer("steered result"),
        ],
    )
    worker = AdkWorker(lambda _: Agent(name="worker", model=model, tools=[block, next_step]))
    async with Supervisor({"worker": worker}) as supervisor:
        task_id = supervisor.spawn("worker", "original task")
        await asyncio.wait_for(entered.wait(), 5)
        first = supervisor.send(task_id, "Prefer option A")
        second = supervisor.send(task_id, "Include the evidence")
        assert first["status"] == second["status"] == "queued"
        assert supervisor.get(task_id)["status"] == "running"
        assert len(model.requests) == 1
        # Child events have reached an observer while its tool is still blocked.
        assert any(
            event["kind"] == "adk_event"
            and any(
                part.get("function_call", {}).get("name") == "block"
                for part in event["payload"].get("content", {}).get("parts", [])
            )
            for event in supervisor.events()["events"]
        )
        release.set()
        result = await terminal(supervisor, task_id)
        assert result["status"] == "succeeded", result
        assert result["result"]["output"] == "steered result"
        assert len(model.requests) == 3
        expected = [first["id"], second["id"]]
        assert mailbox_messages(model.requests[0]) == []
        for request in model.requests[1:]:
            delivered = mailbox_messages(request)
            assert [message["id"] for message in delivered] == expected
            assert [message["text"] for message in delivered] == [
                "Prefer option A",
                "Include the evidence",
            ]
        for receipt in (first, second):
            assert supervisor.message_receipt(task_id, receipt["id"])["status"] == "delivered"


async def test_accepted_message_during_final_model_request_gets_followup():
    in_model = asyncio.Event()
    release = asyncio.Event()

    async def before_response(index: int, _: LlmRequest) -> None:
        if index == 0:
            in_model.set()
            await release.wait()

    model = ScriptedModel(
        model="scripted-local",
        responses=[
            answer("initial answer"),
            answer("revised answer"),
        ],
        before_response=before_response,
    )
    worker = AdkWorker(lambda _: Agent(name="worker", model=model))
    async with Supervisor({"worker": worker}) as supervisor:
        task_id = supervisor.spawn("worker", "original task")
        await asyncio.wait_for(in_model.wait(), 5)
        receipt = supervisor.send(task_id, "Revise the final answer")
        release.set()
        result = await terminal(supervisor, task_id)
        assert result["status"] == "succeeded", result
        assert result["result"]["output"] == "revised answer"
        assert len(model.requests) == 2
        assert mailbox_messages(model.requests[0]) == []
        assert [message["id"] for message in mailbox_messages(model.requests[1])] == [receipt["id"]]
        history = [
            part.text
            for content in model.requests[1].contents
            for part in content.parts or []
            if part.text
        ]
        assert "original task" in history
        assert "initial answer" in history
        assert supervisor.message_receipt(task_id, receipt["id"])["status"] == "delivered"


async def test_final_response_race_honors_followup_limit_and_records_undelivered():
    in_model = asyncio.Event()
    release = asyncio.Event()

    async def before_response(_: int, __: LlmRequest) -> None:
        in_model.set()
        await release.wait()

    model = ScriptedModel(
        model="scripted-local", responses=[answer("initial")], before_response=before_response
    )
    worker = AdkWorker(lambda _: Agent(name="worker", model=model), max_followups=0)
    async with Supervisor({"worker": worker}) as supervisor:
        task_id = supervisor.spawn("worker", "task")
        await asyncio.wait_for(in_model.wait(), 5)
        receipt = supervisor.send(task_id, "More work")
        release.set()
        result = await terminal(supervisor, task_id)
        assert result["status"] == "failed", result
        assert "follow-up limit" in result["error"]
        assert "result" not in result
        assert supervisor.message_receipt(task_id, receipt["id"])["status"] == "undelivered"


async def test_model_error_event_is_not_a_successful_result():
    model = ScriptedModel(
        model="scripted-local",
        responses=[
            LlmResponse(error_code="UNAVAILABLE", error_message="scripted model unavailable"),
        ],
    )
    worker = AdkWorker(lambda _: Agent(name="worker", model=model))
    async with Supervisor({"worker": worker}) as supervisor:
        result = await terminal(supervisor, supervisor.spawn("worker", "task"))
        assert result["status"] == "failed", result
        assert "scripted model unavailable" in result["error"]
        assert "result" not in result


@pytest.mark.parametrize("pause_kind", ["long_running", "confirmation"])
async def test_unresolved_human_input_is_not_a_successful_result(pause_kind: str):
    def needs_input() -> None:
        """Wait for an external answer."""

    if pause_kind == "long_running":
        tool = LongRunningFunctionTool(needs_input)
    else:
        tool = FunctionTool(needs_input, require_confirmation=True)
    model = ScriptedModel(
        model="scripted-local",
        responses=[
            call("needs_input"),
            answer("must not report success"),
        ],
    )
    worker = AdkWorker(lambda _: Agent(name="worker", model=model, tools=[tool]))
    async with Supervisor({"worker": worker}) as supervisor:
        task_id = supervisor.spawn("worker", "task")
        result = await terminal(supervisor, task_id)
        assert result["status"] == "failed", result
        assert "unresolved human input or tool response" in result["error"]
        assert "result" not in result


@pytest.mark.parametrize("mode", ["chat", "nested_single_turn"])
async def test_thought_parts_do_not_become_final_output(mode: str):
    model = ScriptedModel(
        model="scripted-local",
        responses=[
            LlmResponse(
                content=types.Content(
                    role="model",
                    parts=[
                        types.Part(text="internal reasoning", thought=True),
                        types.Part.from_text(text="visible answer"),
                    ],
                ),
            )
        ],
    )

    def factory(_: WorkerContext) -> Agent:
        if mode == "chat":
            return Agent(name="worker", model=model)
        nested = Agent(name="nested", model=model, mode="single_turn")
        parent_model = ScriptedModel(
            model="scripted-local",
            responses=[call("nested", request="Answer the task"), answer("visible answer")],
        )
        return Agent(name="worker", model=parent_model, sub_agents=[nested])

    worker = AdkWorker(factory)
    async with Supervisor({"worker": worker}) as supervisor:
        result = await terminal(supervisor, supervisor.spawn("worker", "task"))
        assert result["status"] == "succeeded", result
        assert result["result"]["output"] == "visible answer"
        assert "internal reasoning" not in json.dumps(supervisor.events())


async def test_nested_agent_cannot_consume_messages_addressed_to_worker():
    entered = asyncio.Event()
    release = asyncio.Event()

    async def block() -> str:
        entered.set()
        await release.wait()
        return "released"

    nested_model = ScriptedModel(
        model="scripted-local", responses=[call("block"), answer("nested answer")]
    )
    root_model = ScriptedModel(
        model="scripted-local",
        responses=[call("nested", request="Do a subtask"), answer("worker answer")],
    )

    def factory(_: WorkerContext) -> Agent:
        nested = Agent(name="nested", mode="single_turn", model=nested_model, tools=[block])
        return Agent(name="worker", model=root_model, sub_agents=[nested])

    async with Supervisor({"worker": AdkWorker(factory)}) as supervisor:
        task_id = supervisor.spawn("worker", "original task")
        await asyncio.wait_for(entered.wait(), 5)
        receipt = supervisor.send(task_id, "Worker: revise your synthesis")
        release.set()
        result = await terminal(supervisor, task_id)
        assert result["status"] == "succeeded", result
        assert result["result"]["output"] == "worker answer"
        assert len(nested_model.requests) == len(root_model.requests) == 2
        assert mailbox_messages(nested_model.requests[1]) == []
        assert [message["id"] for message in mailbox_messages(root_model.requests[1])] == [
            receipt["id"]
        ]
        assert supervisor.message_receipt(task_id, receipt["id"])["status"] == "delivered"


@pytest.mark.parametrize("failure", ["error", "cancel"])
async def test_runner_closes_if_session_initialization_fails(
    monkeypatch: pytest.MonkeyPatch, failure: str
):
    import adk_collaboration.adk as adapter

    entered = asyncio.Event()
    never_release = asyncio.Event()
    closed: list[InMemoryRunner] = []

    def failing_runner(*args: Any, **kwargs: Any) -> InMemoryRunner:
        runner = InMemoryRunner(*args, **kwargs)
        original_close = runner.close

        async def create_session(**_: Any) -> None:
            entered.set()
            if failure == "error":
                raise RuntimeError("scripted session initialization failure")
            await never_release.wait()

        async def close() -> None:
            closed.append(runner)
            await original_close()

        monkeypatch.setattr(runner.session_service, "create_session", create_session)
        monkeypatch.setattr(runner, "close", close)
        return runner

    monkeypatch.setattr(adapter, "InMemoryRunner", failing_runner)
    model = ScriptedModel(model="scripted-local", responses=[answer("unused")])
    worker = AdkWorker(lambda _: Agent(name="worker", model=model))
    async with Supervisor({"worker": worker}) as supervisor:
        task_id = supervisor.spawn("worker", "task")
        await asyncio.wait_for(entered.wait(), 5)
        if failure == "cancel":
            await supervisor.cancel(task_id)
        result = await terminal(supervisor, task_id)
        assert result["status"] == ("failed" if failure == "error" else "cancelled"), result
        assert len(closed) == 1
        assert model.requests == []


@pytest.mark.parametrize("container", ["direct", "list", "dict", "tuple"])
async def test_observer_filters_thought_parts_in_typed_content_output(container: str):
    class ContentOutputAgent(BaseAgent):
        async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
            content = types.Content(
                role="model",
                parts=[
                    types.Part(text="typed output internal reasoning", thought=True),
                    types.Part(text="visible structured answer", thought_signature=b"opaque"),
                ],
            )
            output = {
                "direct": content,
                "list": [content],
                "dict": {"answer": content},
                "tuple": (content,),
            }[container]
            yield Event(
                author=self.name,
                invocation_id=ctx.invocation_id,
                output=output,
            )

    worker = AdkWorker(lambda _: ContentOutputAgent(name="worker"))
    async with Supervisor({"worker": worker}) as supervisor:
        task_id = supervisor.spawn("worker", "task")
        result = await terminal(supervisor, task_id)
        journal = json.dumps(supervisor.events())
        assert "typed output internal reasoning" not in journal
        assert "thought_signature" not in journal
        assert result["status"] == "succeeded", result
        assert "visible structured answer" in json.dumps(result["result"]["output"])
        assert "typed output internal reasoning" not in json.dumps(result)
        assert "thought_signature" not in json.dumps(result)


@pytest.mark.parametrize("nested", [False, True])
async def test_export_detects_boolean_to_number_change(nested: bool):
    baseline = {"value": {"enabled": False} if nested else False}
    updated = {"enabled": 0} if nested else 0

    def update_value(tool_context: ToolContext) -> str:
        tool_context.state["value"] = updated
        return "updated"

    model = ScriptedModel(model="scripted-local", responses=[call("update_value"), answer("done")])
    worker = AdkWorker(
        lambda _: Agent(name="worker", model=model, tools=[update_value]),
        export_keys=("value",),
    )
    async with Supervisor({"worker": worker}, initial_state=baseline) as supervisor:
        result = await terminal(supervisor, supervisor.spawn("worker", "task"))
        assert result["status"] == "succeeded", result
        assert "value" in result["result"]["updates"]
        actual = result["result"]["updates"]["value"]
        assert json.dumps(actual) == json.dumps(updated)


async def test_followup_without_visible_answer_does_not_reuse_stale_result():
    in_model = asyncio.Event()
    release = asyncio.Event()

    async def before_response(index: int, _: LlmRequest) -> None:
        if index == 0:
            in_model.set()
            await release.wait()

    model = ScriptedModel(
        model="scripted-local",
        responses=[
            answer("obsolete answer"),
            LlmResponse(
                content=types.Content(
                    role="model", parts=[types.Part(text="reasoning only", thought=True)]
                )
            ),
        ],
        before_response=before_response,
    )
    worker = AdkWorker(lambda _: Agent(name="worker", model=model))
    async with Supervisor({"worker": worker}) as supervisor:
        task_id = supervisor.spawn("worker", "task")
        await asyncio.wait_for(in_model.wait(), 5)
        receipt = supervisor.send(task_id, "That answer is incorrect; revise it")
        release.set()
        result = await terminal(supervisor, task_id)
        assert result["status"] == "failed", result
        assert "result" not in result
        assert supervisor.message_receipt(task_id, receipt["id"])["status"] == "delivered"


async def test_nested_tool_answer_does_not_replace_missing_worker_answer():
    nested_model = ScriptedModel(model="scripted-local", responses=[answer("nested-only answer")])
    root_model = ScriptedModel(
        model="scripted-local",
        responses=[
            call("nested", request="Do a subtask"),
            LlmResponse(
                content=types.Content(
                    role="model", parts=[types.Part(text="unfinished synthesis", thought=True)]
                )
            ),
        ],
    )

    def factory(_: WorkerContext) -> Agent:
        nested = Agent(name="nested", mode="single_turn", model=nested_model)
        return Agent(name="worker", model=root_model, sub_agents=[nested])

    async with Supervisor({"worker": AdkWorker(factory)}) as supervisor:
        result = await terminal(supervisor, supervisor.spawn("worker", "task"))
        assert result["status"] == "failed", result
        assert "result" not in result


@pytest.mark.parametrize("wrong_identity", ["app", "user", "session"])
async def test_all_coordinator_tools_reject_other_parent_sessions(wrong_identity: str):
    entered = asyncio.Event()
    release = asyncio.Event()

    async def worker(_: WorkerContext, task_input: Any) -> TaskResult:
        if task_input == "block":
            entered.set()
            await release.wait()
        return TaskResult("private result", {"secret": "child"})

    async with Supervisor({"worker": worker}, initial_state={"secret": "parent"}) as supervisor:
        complete = supervisor.spawn("worker", "finish")
        assert (await terminal(supervisor, complete))["status"] == "succeeded"
        running = supervisor.spawn("worker", "block")
        await asyncio.wait_for(entered.wait(), 5)
        receipt = supervisor.send(running, "existing private message")
        bound = {"app_name": "parent_app", "user_id": "owner", "session_id": "parent_session"}
        actual = dict(bound)
        actual[{"app": "app_name", "user": "user_id", "session": "session_id"}[wrong_identity]] += (
            "_foreign"
        )
        calls = [
            ("spawn_task", {"worker": "worker", "request": "unwanted task"}),
            ("get_task", {"task_id": complete}),
            ("wait_tasks", {"task_ids": [running], "mode": "all", "timeout_seconds": 0}),
            ("send_message", {"task_id": running, "text": "unwanted message"}),
            ("get_message_receipt", {"task_id": running, "message_id": receipt["id"]}),
            ("cancel_task", {"task_id": running}),
            ("merge_result", {"task_id": complete, "keys": ["secret"]}),
            ("read_team_state", {}),
        ]
        model = ScriptedModel(
            model="scripted-local",
            responses=[
                LlmResponse(
                    content=types.Content(
                        role="model",
                        parts=[
                            types.Part.from_function_call(name=name, args=arguments)
                            for name, arguments in calls
                        ],
                    )
                ),
                answer("done"),
            ],
        )
        parent = Agent(
            name="coordinator", model=model, tools=coordinator_tools(supervisor, **bound)
        )
        runner = InMemoryRunner(agent=parent, app_name=actual["app_name"])
        try:
            session = await runner.session_service.create_session(**actual)
            async with asyncio.timeout(5):
                events = [
                    event
                    async for event in runner.run_async(
                        user_id=session.user_id,
                        session_id=session.id,
                        new_message=types.Content(
                            role="user", parts=[types.Part.from_text(text="try")]
                        ),
                    )
                ]
        finally:
            await runner.close()
        responses = {
            response.name: response.response
            for event in events
            for response in event.get_function_responses()
        }
        assert responses.keys() == {name for name, _ in calls}
        for name, result in responses.items():
            assert result == {
                "error": "CollaborationError",
                "message": "These team tools belong to a different parent session",
            }, (name, result)
        assert supervisor.state == {"secret": "parent"}
        assert supervisor.get(running)["status"] == "running"
        journal = supervisor.events()["events"]
        assert sum(event["kind"] == "task_queued" for event in journal) == 2
        assert sum(event["kind"] == "message_queued" for event in journal) == 1
        release.set()
