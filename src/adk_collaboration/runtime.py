"""Owned task lifetimes, bounded admission, mailboxes, and explicit state merges.

One Supervisor belongs to one coordinator and one asyncio event loop. Workers
cannot recursively spawn children through WorkerContext: waiting parents must
not occupy every admission slot while their children queue behind them.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections import deque
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4


class CollaborationError(Exception):
    """A rejected collaboration operation."""


class EventGap(CollaborationError):
    """The requested event cursor has been evicted from the bounded journal."""


class MergeConflict(CollaborationError):
    def __init__(self, keys: list[str]):
        self.keys = keys
        super().__init__(f"State changed since task snapshot: {', '.join(keys)}")


TERMINAL = frozenset({"succeeded", "failed", "cancelled"})
_MISSING = object()


def _json_equal(left: Any, right: Any) -> bool:
    if left is _MISSING or right is _MISSING:
        return left is right
    # Python equality conflates booleans with numbers, including inside containers.
    return json.dumps(left, sort_keys=True, allow_nan=False) == json.dumps(
        right, sort_keys=True, allow_nan=False
    )


def _copy_json(value: Any, limit: int = 1_048_576) -> Any:
    """Validate and detach JSON payloads. Never truncate semantic data."""

    def check_keys(item: Any) -> None:
        if isinstance(item, dict):
            if any(not isinstance(key, str) for key in item):
                raise CollaborationError("JSON object keys must be strings")
            for child in item.values():
                check_keys(child)
        elif isinstance(item, (list, tuple)):
            for child in item:
                check_keys(child)

    check_keys(value)
    try:
        encoded = json.dumps(value, allow_nan=False, ensure_ascii=False)
    except (ValueError, TypeError) as error:
        raise CollaborationError("Payload must be finite JSON data") from error
    if len(encoded.encode()) > limit:
        raise CollaborationError(f"Payload exceeds {limit} bytes")
    return json.loads(encoded)


def _state(value: Mapping[str, Any], limit: int) -> dict[str, Any]:
    result = _copy_json(dict(value), limit)
    if any(":" in key for key in result):
        raise CollaborationError("Team state keys cannot use ADK reserved ':' namespaces")
    return result


@dataclass(frozen=True)
class TaskResult:
    output: Any
    updates: dict[str, Any] = field(default_factory=dict)


@dataclass
class _Record:
    id: str
    worker: str
    input: Any
    snapshot: dict[str, Any]
    status: str = "queued"
    result: TaskResult | None = None
    error: str | None = None
    task: asyncio.Task | None = None
    done: asyncio.Event = field(default_factory=asyncio.Event)
    inbox: deque = field(default_factory=deque)
    receipts: dict[str, dict[str, Any]] = field(default_factory=dict)
    accepting_messages: bool = True
    merged_keys: set[str] = field(default_factory=set)


Worker = Callable[["WorkerContext", Any], Awaitable[TaskResult]]


class WorkerContext:
    """A worker's identity, snapshot, progress, and message capabilities."""

    def __init__(self, supervisor: Supervisor, record: _Record):
        self._supervisor = supervisor
        self._record = record

    @property
    def task_id(self) -> str:
        return self._record.id

    @property
    def initial_state(self) -> dict[str, Any]:
        return _copy_json(self._record.snapshot, self._supervisor.max_payload_bytes)

    def emit(self, kind: str, payload: Any) -> None:
        self._supervisor._emit(kind, self.task_id, payload)

    def pending_messages(self) -> list[dict[str, Any]]:
        return _copy_json(list(self._record.inbox), self._supervisor.max_payload_bytes)

    def acknowledge_messages(self, ids: list[str]) -> None:
        """Acknowledge only after successfully adding messages to worker context."""
        queued = [message["id"] for message in self._record.inbox]
        if queued[: len(ids)] != ids:
            raise CollaborationError("Messages must be acknowledged in queue order")
        for message_id in ids:
            self._record.inbox.popleft()
            self._record.receipts[message_id]["status"] = "delivered"
            self._supervisor._emit("message_delivered", self.task_id, {"id": message_id})

    def seal_messages(self) -> None:
        """Atomically finish accepting messages only after the inbox is empty."""
        if self._record.inbox:
            raise CollaborationError("Cannot complete with queued messages")
        self._record.accepting_messages = False

    def send(self, task_id: str, text: str) -> dict[str, Any]:
        return self._supervisor.send(task_id, text, sender=self.task_id)


class Supervisor:
    """Manage a bounded team. Use as an async context manager to own cleanup.

    Capacity bounds active workers; max_tasks also bounds queued and retained
    completed handles for this lifetime. State merging is an in-process
    compare-and-swap over this supervisor's own team state, never a transaction
    against an external database or a parent's ADK session.
    """

    def __init__(
        self,
        workers: Mapping[str, Worker],
        *,
        initial_state: Mapping[str, Any] | None = None,
        max_workers: int = 4,
        max_tasks: int = 128,
        max_events: int = 1000,
        max_messages: int = 128,
        max_payload_bytes: int = 1_048_576,
        max_event_bytes: int = 65_536,
    ):
        for value in (
            max_workers,
            max_tasks,
            max_events,
            max_messages,
            max_payload_bytes,
            max_event_bytes,
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError("All limits must be positive integers")
        if max_event_bytes < 4096:
            raise ValueError("max_event_bytes must be at least 4096 for lifecycle records")
        if any(not isinstance(name, str) or not name or len(name) > 128 for name in workers):
            raise ValueError("Worker names must contain 1 to 128 characters")
        self.workers = dict(workers)
        self.max_tasks = max_tasks
        self.max_messages = max_messages
        self.max_payload_bytes = max_payload_bytes
        self.max_event_bytes = max_event_bytes
        self._state = _state(initial_state or {}, max_payload_bytes)
        self._semaphore = asyncio.Semaphore(max_workers)
        self._records: dict[str, _Record] = {}
        self._journal: deque = deque(maxlen=max_events)
        self._sequence = 0
        self._changed = asyncio.Event()
        self._closed = False
        self._loop: asyncio.AbstractEventLoop | None = None

    def _check_loop(self) -> None:
        loop = asyncio.get_running_loop()
        if self._loop is None:
            self._loop = loop
        elif loop is not self._loop:
            raise CollaborationError("Supervisor cannot be shared across event loops")

    def _record(self, task_id: str) -> _Record:
        self._check_loop()
        try:
            return self._records[task_id]
        except KeyError as error:
            raise CollaborationError("Unknown task handle for this supervisor") from error

    def _emit(self, kind: str, task_id: str | None, payload: Any) -> None:
        if not isinstance(kind, str) or not 1 <= len(kind) <= 128:
            raise CollaborationError("Event kind must contain 1 to 128 characters")
        event = _copy_json(
            {
                "cursor": self._sequence + 1,
                "time": time.time(),
                "kind": kind,
                "task_id": task_id,
                "payload": payload,
            },
            self.max_event_bytes,
        )
        self._sequence += 1
        self._journal.append(event)
        self._changed.set()

    @property
    def state(self) -> dict[str, Any]:
        return _copy_json(self._state, self.max_payload_bytes)

    def spawn(self, worker: str, task_input: Any) -> str:
        self._check_loop()
        if self._closed:
            raise CollaborationError("Supervisor is closed")
        if worker not in self.workers:
            raise CollaborationError(f"Unknown worker: {worker}")
        if len(self._records) >= self.max_tasks:
            raise CollaborationError("Task admission limit reached")
        record = _Record(
            uuid4().hex, worker, _copy_json(task_input, self.max_payload_bytes), self.state
        )
        self._records[record.id] = record
        self._emit("task_queued", record.id, {"worker": worker})
        record.task = asyncio.create_task(self._run(record), name=f"adk-team:{record.id}")
        record.task.add_done_callback(lambda task: self._task_done(record, task))
        return record.id

    def _finish(self, record: _Record, status: str, error: str | None = None) -> None:
        if record.status in TERMINAL:
            return
        record.accepting_messages = False
        while record.inbox:
            message = record.inbox.popleft()
            record.receipts[message["id"]]["status"] = "undelivered"
            self._emit("message_undelivered", record.id, {"id": message["id"]})
        record.status = status
        record.error = error
        self._emit("task_terminal", record.id, {"status": status, "error": error})
        record.done.set()

    def _task_done(self, record: _Record, task: asyncio.Task) -> None:
        # A coroutine cancelled before its first instruction cannot run finally.
        if task.cancelled():
            self._finish(record, "cancelled")
        elif (error := task.exception()) is not None:
            self._finish(record, "failed", f"{type(error).__name__}: {error}"[:300])

    async def _run(self, record: _Record) -> None:
        try:
            async with self._semaphore:
                if record.status == "cancelling":
                    raise asyncio.CancelledError
                record.status = "running"
                self._emit("task_running", record.id, {"worker": record.worker})
                context = WorkerContext(self, record)
                result = await self.workers[record.worker](context, record.input)
                if record.status == "cancelling":
                    raise asyncio.CancelledError
                if not isinstance(result, TaskResult):
                    raise CollaborationError("Worker must return TaskResult")
                if record.inbox:
                    raise CollaborationError("Worker returned without delivering queued messages")
                output = _copy_json(result.output, self.max_payload_bytes)
                updates = _state(result.updates, self.max_payload_bytes)
                record.result = TaskResult(output, updates)
                self._finish(record, "succeeded")
        except asyncio.CancelledError:
            self._finish(record, "cancelled")
            raise
        except Exception as error:
            self._finish(record, "failed", f"{type(error).__name__}: {error}"[:300])

    def get(self, task_id: str, *, include_result: bool = True) -> dict[str, Any]:
        record = self._record(task_id)
        result = {
            "task_id": record.id,
            "worker": record.worker,
            "status": record.status,
            "error": record.error,
        }
        if include_result and record.result is not None:
            result["result"] = {"output": record.result.output, "updates": record.result.updates}
        # Two independently bounded payloads may both occur in a result.
        return _copy_json(result, 2 * self.max_payload_bytes + 4096)

    async def wait(
        self,
        task_ids: list[str],
        *,
        mode: str = "all",
        timeout: float | None = None,
    ) -> list[dict[str, Any]]:
        if mode not in {"any", "all"}:
            raise ValueError("mode must be 'any' or 'all'")
        if not task_ids or len(task_ids) != len(set(task_ids)):
            raise ValueError("Pass a nonempty list of unique task handles")
        records = [self._record(task_id) for task_id in task_ids]
        waiters = [asyncio.create_task(record.done.wait()) for record in records]
        try:
            await asyncio.wait(
                waiters,
                timeout=timeout,
                return_when=asyncio.FIRST_COMPLETED if mode == "any" else asyncio.ALL_COMPLETED,
            )
            return [self.get(record.id) for record in records if record.done.is_set()]
        finally:
            for waiter in waiters:
                waiter.cancel()
            await asyncio.gather(*waiters, return_exceptions=True)

    async def cancel(self, task_id: str, *, timeout: float = 5.0) -> dict[str, Any]:
        record = self._record(task_id)
        if record.status not in TERMINAL and record.status != "cancelling":
            record.status = "cancelling"
            record.accepting_messages = False
            self._emit("task_cancelling", task_id, {})
            assert record.task is not None
            record.task.cancel()
        if not record.done.is_set():
            try:
                await asyncio.wait_for(record.done.wait(), timeout)
            except TimeoutError:
                pass  # Still cancelling: never claim an uncooperative worker stopped.
        return self.get(task_id)

    def send(self, task_id: str, text: str, *, sender: str = "coordinator") -> dict[str, Any]:
        record = self._record(task_id)
        if sender != "coordinator":
            self._record(sender)
        if not isinstance(text, str) or not text.strip():
            raise CollaborationError("Message text must be nonempty")
        if not record.accepting_messages or record.status in TERMINAL:
            raise CollaborationError("Task is no longer accepting messages")
        # Bound receipts over the task's entire lifetime, not just pending messages.
        if len(record.receipts) >= self.max_messages:
            raise CollaborationError("Task message limit reached")
        message = _copy_json(
            {"id": uuid4().hex, "sender": sender, "text": text},
            min(self.max_payload_bytes, self.max_event_bytes),
        )
        _copy_json([*record.inbox, message], self.max_payload_bytes)
        record.inbox.append(message)
        receipt = {"id": message["id"], "task_id": task_id, "status": "queued"}
        record.receipts[message["id"]] = receipt
        self._emit("message_queued", task_id, {"id": message["id"], "sender": sender})
        return dict(receipt)

    def message_receipt(self, task_id: str, message_id: str) -> dict[str, Any]:
        record = self._record(task_id)
        if message_id not in record.receipts:
            raise CollaborationError("Unknown message receipt")
        return dict(record.receipts[message_id])

    def merge(self, task_id: str, *, keys: list[str] | None = None) -> dict[str, Any]:
        """Atomically merge selected proposed updates into the team's own state.

        All comparisons precede all writes, and this synchronous operation has no
        await point on the owned event loop. Retrying already-merged keys is a
        no-op; null is a value, not a deletion instruction.
        """
        record = self._record(task_id)
        if record.status != "succeeded" or record.result is None:
            raise CollaborationError("Only succeeded tasks can be merged")
        selected = set(record.result.updates if keys is None else keys)
        if not selected <= record.result.updates.keys():
            raise CollaborationError("Cannot merge undeclared result keys")
        pending = selected - record.merged_keys
        conflicts = sorted(
            key
            for key in pending
            if not _json_equal(self._state.get(key, _MISSING), record.snapshot.get(key, _MISSING))
        )
        if conflicts:
            raise MergeConflict(conflicts)
        candidate = self.state
        for key in pending:
            candidate[key] = record.result.updates[key]
        candidate = _state(candidate, self.max_payload_bytes)
        # Validate the observation event before committing the update as well.
        payload = _copy_json({"keys": sorted(pending)}, self.max_event_bytes)
        self._state = candidate
        record.merged_keys |= pending
        self._emit("state_merged", task_id, payload)
        return {
            "task_id": task_id,
            "merged_keys": sorted(pending),
            "already_merged_keys": sorted(selected - pending),
        }

    def events(self, *, after: int = 0) -> dict[str, Any]:
        self._check_loop()
        if after < 0 or after > self._sequence:
            raise CollaborationError("Invalid event cursor")
        if self._journal and after < self._journal[0]["cursor"] - 1:
            raise EventGap(f"Events evicted; earliest cursor is {self._journal[0]['cursor']}")
        return {
            "cursor": self._sequence,
            "events": _copy_json(
                [event for event in self._journal if event["cursor"] > after],
                (self.max_event_bytes + 1024) * len(self._journal) + 1024,
            ),
        }

    async def observe(self, *, after: int = 0, timeout: float = 30.0) -> dict[str, Any]:
        page = self.events(after=after)
        if page["events"] or self._closed:
            return page
        self._changed.clear()
        try:
            await asyncio.wait_for(self._changed.wait(), timeout)
        except TimeoutError:
            pass
        return self.events(after=after)

    async def close(self, *, timeout: float = 5.0) -> None:
        self._check_loop()
        self._closed = True
        self._changed.set()
        pending = [record for record in self._records.values() if record.status not in TERMINAL]
        if pending:
            await asyncio.gather(*(self.cancel(record.id, timeout=timeout) for record in pending))
        # Wait for coroutine finally blocks, not merely the terminal event.
        tasks = [
            record.task
            for record in self._records.values()
            if record.task is not None and not record.task.done()
        ]
        if tasks:
            _, remaining = await asyncio.wait(tasks, timeout=timeout)
            if remaining:
                raise CollaborationError("Workers are still cancelling; cleanup is incomplete")

    async def __aenter__(self) -> Supervisor:
        self._check_loop()
        return self

    async def __aexit__(self, *_: Any) -> None:
        await self.close()
