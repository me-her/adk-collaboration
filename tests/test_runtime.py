"""Behavioral checks for owned task lifetimes and explicit collaboration state.

Workers are coordinated by events rather than elapsed sleeps, so concurrency and
ordering assertions describe actual execution instead of scheduler timing.
"""

import asyncio

import pytest

from adk_collaboration.runtime import (
    CollaborationError,
    EventGap,
    MergeConflict,
    Supervisor,
    TaskResult,
)


async def reached(event):
    await asyncio.wait_for(event.wait(), timeout=2)


async def results(team, handles):
    completed = await team.wait(handles, timeout=2)
    assert len(completed) == len(handles), "Tasks did not reach a terminal state"
    return completed


async def test_parallel_workers_obey_capacity_and_queued_cancellation():
    entered = asyncio.Queue()
    release = asyncio.Event()
    active = 0
    peak = 0

    async def worker(context, name):
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        await entered.put(name)
        try:
            await release.wait()
            return TaskResult(name)
        finally:
            active -= 1

    async with Supervisor({"worker": worker}, max_workers=2) as team:
        handles = [team.spawn("worker", name) for name in range(4)]
        first = await asyncio.wait_for(entered.get(), 2)
        second = await asyncio.wait_for(entered.get(), 2)
        assert {first, second} == {0, 1}
        assert active == 2
        assert team.get(handles[2])["status"] == "queued"
        assert team.get(handles[3])["status"] == "queued"
        assert (await team.cancel(handles[2]))["status"] == "cancelled"
        release.set()
        completed = await results(team, handles)
        assert [item["status"] for item in completed] == [
            "succeeded",
            "succeeded",
            "cancelled",
            "succeeded",
        ]
        assert await asyncio.wait_for(entered.get(), 2) == 3
        assert entered.empty()
        assert peak == 2
        assert active == 0


async def test_cancel_before_coroutine_starts_is_terminal_and_never_runs():
    called = False

    async def worker(context, value):
        nonlocal called
        called = True
        return TaskResult(value)

    async with Supervisor({"worker": worker}, max_workers=1) as team:
        handle = team.spawn("worker", None)
        assert (await team.cancel(handle))["status"] == "cancelled"
        assert not called
        assert (await results(team, [handle]))[0]["status"] == "cancelled"
        assert (await team.cancel(handle))["status"] == "cancelled"
        terminals = [event for event in team.events()["events"] if event["kind"] == "task_terminal"]
        assert len(terminals) == 1


async def test_running_cancellation_is_selective_and_runs_finally():
    started = {name: asyncio.Event() for name in ("target", "sibling")}
    cleaned = {name: asyncio.Event() for name in started}
    release = asyncio.Event()

    async def worker(context, name):
        started[name].set()
        try:
            await release.wait()
            return TaskResult(name)
        finally:
            cleaned[name].set()

    async with Supervisor({"worker": worker}) as team:
        target = team.spawn("worker", "target")
        sibling = team.spawn("worker", "sibling")
        await reached(started["target"])
        await reached(started["sibling"])
        assert (await team.cancel(target))["status"] == "cancelled"
        await reached(cleaned["target"])
        assert team.get(sibling)["status"] == "running"
        assert not cleaned["sibling"].is_set()
        release.set()
        assert (await results(team, [sibling]))[0]["status"] == "succeeded"


async def test_cancel_timeout_does_not_report_uncooperative_worker_stopped():
    started = asyncio.Event()
    cancelled = asyncio.Event()
    release = asyncio.Event()

    async def worker(context, value):
        started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled.set()
            await release.wait()
        return TaskResult("returned after cancellation")

    async with Supervisor({"worker": worker}) as team:
        handle = team.spawn("worker", None)
        await reached(started)
        response = await team.cancel(handle, timeout=0.01)
        await reached(cancelled)
        assert response["status"] == "cancelling"
        assert await team.wait([handle], timeout=0) == []
        release.set()
        assert (await results(team, [handle]))[0]["status"] == "cancelled"


async def test_wait_any_all_timeout_and_caller_cancellation_preserve_workers():
    gates = {name: asyncio.Event() for name in ("first", "second")}
    starts = {name: asyncio.Event() for name in gates}

    async def worker(context, name):
        starts[name].set()
        await gates[name].wait()
        return TaskResult(name)

    async with Supervisor({"worker": worker}) as team:
        first = team.spawn("worker", "first")
        second = team.spawn("worker", "second")
        await reached(starts["first"])
        await reached(starts["second"])
        assert await team.wait([first, second], mode="any", timeout=0) == []
        assert await team.wait([first, second], mode="all", timeout=0) == []
        waiter_started = asyncio.Event()

        async def wait_forever():
            waiter_started.set()
            return await team.wait([first, second])

        waiter = asyncio.create_task(wait_forever())
        await reached(waiter_started)
        waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiter
        assert team.get(first)["status"] == team.get(second)["status"] == "running"
        gates["first"].set()
        ready = await team.wait([second, first], mode="any", timeout=2)
        assert [item["task_id"] for item in ready] == [first]
        assert [item["task_id"] for item in await team.wait([first, second], timeout=0)] == [first]
        gates["second"].set()
        assert [item["task_id"] for item in await results(team, [second, first])] == [second, first]


@pytest.mark.parametrize(
    "behavior, expected_error",
    [
        ("raise", "RuntimeError: worker exploded"),
        ("wrong_result", "Worker must return TaskResult"),
        ("invalid_output", "Payload must be finite JSON data"),
        ("reserved_state", "reserved"),
    ],
)
async def test_worker_failures_are_observable_and_do_not_poison_siblings(behavior, expected_error):
    async def worker(context, value):
        if value == "raise":
            raise RuntimeError("worker exploded")
        if value == "wrong_result":
            return "invalid contract"
        if value == "invalid_output":
            return TaskResult(float("nan"))
        if value == "reserved_state":
            return TaskResult(None, {"user:secret": True})
        return TaskResult("healthy")

    async with Supervisor({"worker": worker}) as team:
        failed = team.spawn("worker", behavior)
        healthy = team.spawn("worker", "healthy")
        failures, successes = await results(team, [failed, healthy])
        assert failures["status"] == "failed"
        assert expected_error in failures["error"]
        assert successes["status"] == "succeeded"
        with pytest.raises(CollaborationError, match="Only succeeded"):
            team.merge(failed)


async def test_message_receipts_acknowledgement_order_sealing_and_detached_copies():
    started = asyncio.Event()
    drain = asyncio.Event()
    sealed = asyncio.Event()
    release = asyncio.Event()
    seen = []

    async def worker(context, value):
        # Worker capabilities deliberately do not include recursive task creation.
        assert not hasattr(context, "spawn")
        started.set()
        await drain.wait()
        pending = context.pending_messages()
        seen.extend(pending)
        pending[0]["text"] = "mutated copy"
        assert context.pending_messages()[0]["text"] == "first"
        with pytest.raises(CollaborationError, match="queue order"):
            context.acknowledge_messages([pending[1]["id"]])
        with pytest.raises(CollaborationError, match="queued messages"):
            context.seal_messages()
        context.acknowledge_messages([item["id"] for item in pending])
        context.seal_messages()
        sealed.set()
        await release.wait()
        return TaskResult("done")

    async with Supervisor({"worker": worker}) as team:
        handle = team.spawn("worker", None)
        await reached(started)
        first = team.send(handle, "first")
        second = team.send(handle, "second")
        first["status"] = "changed by caller"
        assert team.message_receipt(handle, first["id"])["status"] == "queued"
        drain.set()
        await reached(sealed)
        assert [item["sender"] for item in seen] == ["coordinator", "coordinator"]
        for receipt in (first, second):
            assert team.message_receipt(handle, receipt["id"])["status"] == "delivered"
        with pytest.raises(CollaborationError, match="no longer accepting"):
            team.send(handle, "too late")
        release.set()
        assert (await results(team, [handle]))[0]["status"] == "succeeded"


@pytest.mark.parametrize("cancel", [False, True])
async def test_unconsumed_messages_are_explicitly_undelivered(cancel):
    started = asyncio.Event()
    release = asyncio.Event()

    async def worker(context, value):
        started.set()
        await release.wait()
        return TaskResult("did not read inbox")

    async with Supervisor({"worker": worker}) as team:
        handle = team.spawn("worker", None)
        await reached(started)
        receipt = team.send(handle, "late correction")
        if cancel:
            await team.cancel(handle)
        else:
            release.set()
        result = (await results(team, [handle]))[0]
        assert result["status"] == ("cancelled" if cancel else "failed")
        if not cancel:
            assert "without delivering" in result["error"]
        assert team.message_receipt(handle, receipt["id"])["status"] == "undelivered"


async def test_worker_to_worker_message_uses_sender_handle():
    receive = asyncio.Event()
    receipt_ready = asyncio.Event()
    receipt = None

    async def receiver(context, value):
        await receive.wait()
        messages = context.pending_messages()
        context.acknowledge_messages([message["id"] for message in messages])
        return TaskResult(messages)

    async def sender(context, target):
        nonlocal receipt
        receipt = context.send(target, "peer finding")
        receipt_ready.set()
        return TaskResult("sent")

    async with Supervisor({"receiver": receiver, "sender": sender}) as team:
        target = team.spawn("receiver", None)
        source = team.spawn("sender", target)
        await reached(receipt_ready)
        receive.set()
        target_result, _ = await results(team, [target, source])
        assert target_result["result"]["output"][0]["sender"] == source
        assert team.message_receipt(target, receipt["id"])["status"] == "delivered"


async def test_admission_and_message_lifetime_caps_include_completed_records():
    drained = asyncio.Event()
    deliver = asyncio.Event()
    release = asyncio.Event()

    async def worker(context, value):
        await deliver.wait()
        context.acknowledge_messages([message["id"] for message in context.pending_messages()])
        drained.set()
        await release.wait()
        return TaskResult(value)

    async with Supervisor({"worker": worker}, max_tasks=1, max_messages=1) as team:
        handle = team.spawn("worker", None)
        with pytest.raises(CollaborationError, match="admission"):
            team.spawn("worker", None)
        receipt = team.send(handle, "one")
        with pytest.raises(CollaborationError, match="message limit"):
            team.send(handle, "two")
        deliver.set()
        await reached(drained)
        assert team.message_receipt(handle, receipt["id"])["status"] == "delivered"
        with pytest.raises(CollaborationError, match="message limit"):
            team.send(handle, "three")
        release.set()
        await results(team, [handle])
        with pytest.raises(CollaborationError, match="admission"):
            team.spawn("worker", None)


async def test_journal_has_monotonic_cursors_gaps_detachment_and_observer_wakeup():
    progress = asyncio.Event()
    emitted = asyncio.Event()
    release = asyncio.Event()

    async def worker(context, value):
        await progress.wait()
        for index in range(5):
            context.emit("progress", {"index": index})
        emitted.set()
        await release.wait()
        return TaskResult("done")

    async with Supervisor({"worker": worker}, max_events=3) as team:
        handle = team.spawn("worker", None)
        first = team.events()
        assert [event["cursor"] for event in first["events"]] == [1]
        progress.set()
        await reached(emitted)
        with pytest.raises(EventGap, match="earliest cursor"):
            team.events(after=first["cursor"])
        page = team.events(after=4)
        assert [event["cursor"] for event in page["events"]] == [5, 6, 7]
        page["events"][0]["payload"]["index"] = "caller edit"
        assert team.events(after=4)["events"][0]["payload"]["index"] == 2
        cursor = page["cursor"]
        assert await team.observe(after=cursor, timeout=0) == {"cursor": cursor, "events": []}
        observer = asyncio.create_task(team.observe(after=cursor, timeout=2))
        release.set()
        terminal = await observer
        assert terminal["events"][-1]["kind"] == "task_terminal"
        assert terminal["events"][-1]["task_id"] == handle
        with pytest.raises(CollaborationError, match="Invalid event cursor"):
            team.events(after=terminal["cursor"] + 1)


async def test_inputs_snapshots_results_and_team_state_are_detached():
    payload = {"items": ["original"]}
    initial = {"nested": {"value": 1}}
    gate = asyncio.Event()
    worker_result = TaskResult({"items": ["answer"]}, {"nested": {"value": 2}})

    async def worker(context, value):
        await gate.wait()
        assert value == {"items": ["original"]}
        snapshot = context.initial_state
        snapshot["nested"]["value"] = "worker edit"
        assert context.initial_state == {"nested": {"value": 1}}
        return worker_result

    async with Supervisor({"worker": worker}, initial_state=initial) as team:
        initial["nested"]["value"] = "external edit"
        handle = team.spawn("worker", payload)
        payload["items"].append("later")
        gate.set()
        completed = (await results(team, [handle]))[0]
        worker_result.output["items"].append("worker later edit")
        worker_result.updates["nested"]["value"] = "worker later edit"
        completed["result"]["output"]["items"].append("caller edit")
        completed["result"]["updates"]["nested"]["value"] = "caller edit"
        assert team.get(handle)["result"] == {
            "output": {"items": ["answer"]},
            "updates": {"nested": {"value": 2}},
        }
        assert team.state == {"nested": {"value": 1}}
        team.merge(handle)
        state = team.state
        state["nested"]["value"] = "caller edit"
        assert team.state == {"nested": {"value": 2}}


async def test_merge_conflicts_are_atomic_and_successful_retries_idempotent():
    async def worker(context, updates):
        return TaskResult("proposal", updates)

    async with Supervisor({"worker": worker}, initial_state={"shared": 0}) as team:
        first = team.spawn("worker", {"shared": 1})
        second = team.spawn("worker", {"independent": "must not leak", "shared": 2})
        await results(team, [first, second])
        assert team.state == {"shared": 0}
        assert team.merge(first)["merged_keys"] == ["shared"]
        with pytest.raises(MergeConflict) as error:
            team.merge(second)
        assert error.value.keys == ["shared"]
        assert team.state == {"shared": 1}
        assert team.merge(second, keys=["independent"])["merged_keys"] == ["independent"]
        third = team.spawn("worker", {"shared": 3})
        await results(team, [third])
        team.merge(third)
        retry = team.merge(first)
        assert retry["merged_keys"] == []
        assert retry["already_merged_keys"] == ["shared"]
        assert team.state == {"shared": 3, "independent": "must not leak"}
        with pytest.raises(CollaborationError, match="undeclared"):
            team.merge(first, keys=["injected"])


async def test_merge_distinguishes_missing_from_null_and_null_is_not_deletion():
    async def worker(context, updates):
        return TaskResult(None, updates)

    async with Supervisor({"worker": worker}, initial_state={"existing": "value"}) as team:
        null = team.spawn("worker", {"absent": None, "existing": None})
        conflicting = team.spawn("worker", {"absent": "other"})
        await results(team, [null, conflicting])
        team.merge(null)
        assert team.state == {"absent": None, "existing": None}
        with pytest.raises(MergeConflict) as error:
            team.merge(conflicting)
        assert error.value.keys == ["absent"]


async def test_merge_rejects_oversized_combined_state_without_partial_commit():
    async def worker(context, updates):
        return TaskResult(None, updates)

    async with Supervisor(
        {"worker": worker}, initial_state={"old": "a" * 45}, max_payload_bytes=100
    ) as team:
        handle = team.spawn("worker", {"new": "b" * 45})
        await results(team, [handle])
        with pytest.raises(CollaborationError, match="exceeds"):
            team.merge(handle)
        assert team.state == {"old": "a" * 45}


async def test_cleanup_awaits_workers_and_wakes_idle_observers():
    started = asyncio.Event()
    cleaned = asyncio.Event()

    async def worker(context, value):
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            cleaned.set()

    team = Supervisor({"worker": worker}, max_workers=1)
    running = team.spawn("worker", None)
    queued = team.spawn("worker", None)
    await reached(started)
    await team.close()
    assert cleaned.is_set()
    assert team.get(running)["status"] == team.get(queued)["status"] == "cancelled"
    with pytest.raises(CollaborationError, match="closed"):
        team.spawn("worker", None)
    await team.close()
    empty = Supervisor({})
    observer = asyncio.create_task(empty.observe(timeout=2))
    await empty.close()
    assert await observer == {"cursor": 0, "events": []}


async def test_handle_and_sender_are_rejected_outside_their_supervisor():
    async def worker(context, value):
        return TaskResult(None)

    async with Supervisor({"worker": worker}) as first, Supervisor({"worker": worker}) as second:
        handle = first.spawn("worker", None)
        own = second.spawn("worker", None)
        with pytest.raises(CollaborationError, match="Unknown task handle"):
            second.get(handle)
        with pytest.raises(CollaborationError, match="Unknown task handle"):
            second.send(own, "spoofed sender", sender=handle)


async def test_minimum_event_budget_still_records_large_errors_and_wakes_waiters():
    async def worker(context, value):
        raise RuntimeError("large failure diagnostic " * 1000)

    with pytest.raises(ValueError, match="4096"):
        Supervisor({"worker": worker}, max_event_bytes=128)
    async with Supervisor({"worker": worker}, max_event_bytes=4096) as team:
        handle = team.spawn("worker", None)
        completed = (await results(team, [handle]))[0]
        assert completed["status"] == "failed"
        assert "RuntimeError" in completed["error"]
        assert team.events()["events"][-1]["kind"] == "task_terminal"


async def test_aggregate_mailbox_budget_rejects_before_acceptance_and_delivers_accepted_message():
    drain = asyncio.Event()

    async def worker(context, value):
        await drain.wait()
        messages = context.pending_messages()
        context.acknowledge_messages([message["id"] for message in messages])
        return TaskResult(len(messages))

    async with Supervisor({"worker": worker}, max_payload_bytes=256) as team:
        handle = team.spawn("worker", None)
        receipt = team.send(handle, "message " + "x" * 100)
        before_rejected_send = team.events()["cursor"]
        with pytest.raises(CollaborationError, match="exceeds"):
            team.send(handle, "message " + "x" * 100)
        assert team.events()["cursor"] == before_rejected_send
        drain.set()
        completed = (await results(team, [handle]))[0]
        assert completed["status"] == "succeeded", completed["error"]
        assert completed["result"]["output"] == 1
        assert team.message_receipt(handle, receipt["id"])["status"] == "delivered"


async def test_event_kind_cannot_bypass_byte_cap_and_poison_observer_journal():
    async def worker(context, value):
        context.emit("x" * 65536, {})
        return TaskResult("oversized observer record")

    async with Supervisor({"worker": worker}, max_event_bytes=4096) as team:
        handle = team.spawn("worker", None)
        completed = (await results(team, [handle]))[0]
        assert completed["status"] == "failed"
        journal = team.events()
        assert journal["events"][-1]["kind"] == "task_terminal"


async def test_merge_detects_nested_boolean_to_number_change_as_a_conflict():
    async def worker(context, updates):
        return TaskResult(None, updates)

    async with Supervisor(
        {"worker": worker}, initial_state={"setting": {"enabled": False}}
    ) as team:
        numeric = team.spawn("worker", {"setting": {"enabled": 0}})
        boolean = team.spawn("worker", {"setting": {"enabled": True}})
        await results(team, [numeric, boolean])
        team.merge(numeric)
        assert type(team.state["setting"]["enabled"]) is int
        with pytest.raises(MergeConflict) as error:
            team.merge(boolean)
        assert error.value.keys == ["setting"]
        assert type(team.state["setting"]["enabled"]) is int
