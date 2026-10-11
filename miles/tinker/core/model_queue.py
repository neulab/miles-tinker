"""Per-model command ordering and batch runs separated by barriers."""

import time
from collections import deque
from dataclasses import dataclass, field

from miles.tinker.core.types import Command


@dataclass
class PendingRequest:
    """One submitted command and its completion accounting."""

    command: Command
    datums: list[dict] = field(default_factory=list)  # batch ops only
    num_issued_datums: int = 0
    outputs: list[dict | None] = field(default_factory=list)
    remaining: int = 0
    enqueued_at: float = field(default_factory=time.monotonic)  # when it became schedulable

    @property
    def is_batch_op(self) -> bool:
        return self.command.op.is_batch()

    def record_output(self, local_index: int, output: dict) -> bool:
        """Store one datum's result; True once every datum has reported."""
        self.outputs[local_index] = output
        self.remaining -= 1
        return self.remaining == 0

    def pack_key(self) -> tuple:
        """Datums pack into one BatchUnit only within the same (op, loss_fn, config)."""
        config = self.command.payload.get("loss_fn_config") or {}
        return (self.command.op, self.command.payload["loss_fn"], tuple(sorted(config.items())))


class ModelRequestQueue:
    def __init__(self, model_id: str, tenant: str, slot: int) -> None:
        self.model_id = model_id
        self.tenant = tenant
        self.slot = slot
        # seq_ids are 1-based
        self.last_enqueued_seq_id = 0
        self.pending_by_seq: dict[int, Command] = {}
        self.queue: deque[PendingRequest] = deque()
        self.last_submit_at = 0.0

    def submit(self, command: Command) -> None:
        """Accept one deduplicated command; feed the queue in seq order."""
        self.last_submit_at = time.monotonic()
        self.pending_by_seq[command.seq_id] = command
        self._enqueue_contiguous_commands()

    def _enqueue_contiguous_commands(self) -> None:
        while self.last_enqueued_seq_id + 1 in self.pending_by_seq:
            self.last_enqueued_seq_id += 1
            next_command = self.pending_by_seq.pop(self.last_enqueued_seq_id)
            pending = PendingRequest(command=next_command)
            if pending.is_batch_op and pending.command.validation_error is None:
                pending.datums = next_command.payload["datums"]
                pending.remaining = len(pending.datums)
                pending.outputs = [None] * len(pending.datums)
            self.queue.append(pending)

    def open_batch_run(self) -> list[PendingRequest]:
        """The leading run of batch-op commands; their datums are all issuable."""
        run = []
        for pending in self.queue:
            if not pending.is_batch_op or pending.command.validation_error is not None:
                break
            run.append(pending)
        return run

    def batch_run_may_grow(self) -> bool:
        """True while only batch ops are queued: a later chunk of the same run can still arrive and pack with them."""
        return bool(self.queue) and len(self.open_batch_run()) == len(self.queue)

    def ready_barrier(self) -> PendingRequest | None:
        """The head barrier, executable once every batch op ahead of it completed."""
        if self.queue and not self.queue[0].is_batch_op and self.queue[0].command.validation_error is None:
            return self.queue[0]
        return None

    def finish(self, pending: PendingRequest) -> None:
        self.queue.remove(pending)
