import queue
import traceback
from typing import Optional

from deserve_worker.engine.event.exec import NewExecEvent
from deserve_worker.kvcache.paged_kvcache import PagedKVCache

from ..execution.exec import BatchDecode, BatchExec, BatchPrefill, SingleTrace
from ..execution.result import BatchPersist, ExecResult
from .event.base import EngineEvent, MoreSpaceEvent

EOS_TOKEN_ID = 128001  # for llama 3 only
STOP_TOKEN_IDS = [128001, 128009]


class LLMEngine:
    def __init__(
        self,
        max_total_bsz: int,
        max_total_seqlen: int,
        receiver: queue.Queue[EngineEvent],
        sender: queue.Queue[ExecResult],
        is_scheduler: bool,
    ):
        self.max_total_bsz = max_total_bsz
        self.max_total_seqlen = max_total_seqlen
        self.receiver = receiver
        self.sender = sender
        self.prefill_buffer: list[BatchPrefill] = []
        self.decode_buffer: list[BatchDecode] = []
        self.trace_buffer: list[SingleTrace] = []
        self.pending_buffer: list[BatchExec] = []
        self.is_scheduler = is_scheduler

    def run(self) -> None:
        while True:
            self.check_buffer()
            responses = None
            if len(self.trace_buffer) != 0:
                responses = self.handle_trace(self.trace_buffer)
            if responses is None and len(self.prefill_buffer) != 0:
                responses = self.handle_prefill(self.prefill_buffer)
            if responses is None and len(self.decode_buffer) != 0:
                responses = self.handle_decode(self.decode_buffer)
            if (
                responses is None
                and len(self.trace_buffer) != 0
                and len(self.prefill_buffer) != 0
                and len(self.decode_buffer) != 0
            ):
                raise RuntimeError(
                    "Unable to handle any request due to lack of resources"
                )
            if responses is not None:
                for response in responses:
                    self.sender.put(response)

    def sort_exec(self, exec: BatchExec) -> None:
        if isinstance(exec, BatchDecode):
            self.decode_buffer.append(exec)
        elif isinstance(exec, BatchPrefill):
            self.prefill_buffer.append(exec)
        elif isinstance(exec, SingleTrace):
            self.trace_buffer.append(exec)
        else:
            raise NotImplementedError(f"Unknown request type {type(exec)}")

    def sort_event(self, event: EngineEvent) -> None:
        if isinstance(event, NewExecEvent):
            self.sort_exec(event.exec)
        elif isinstance(event, MoreSpaceEvent):
            while len(self.pending_buffer) > 0:
                exec = self.pending_buffer[-1]
                num_avails = exec.page_pool.num_avails
                if exec.check_kvcache_available(num_avails):
                    num_avails -= exec.get_extended_pages_num()
                    self.pending_buffer.pop()
                    self.sort_exec(exec)
                else:
                    break

    def check_buffer(self) -> None:
        if (
            len(self.prefill_buffer) == 0
            and len(self.decode_buffer) == 0
            and len(self.trace_buffer) == 0
        ):
            self.sort_event(self.receiver.get())
        while True:
            try:
                self.sort_event(self.receiver.get(block=False))
            except queue.Empty:
                break

    def handle_prefill(
        self, prefills: list[BatchPrefill]
    ) -> Optional[list[ExecResult]]:
        prefills.sort(key=lambda task: task.total_seqlen())
        total_seqlen = 0
        total_bsz = 0
        todos: list[BatchPrefill] = []
        for i in reversed(range(len(prefills))):
            # TODO: after chunk prefill is introduced, we don't need to check whether todos is empty
            if (
                total_seqlen + prefills[i].total_seqlen() <= self.max_total_seqlen
                and total_bsz + prefills[i].bsz() <= self.max_total_bsz
            ) or len(todos) == 0:
                total_bsz += prefills[i].bsz()
                total_seqlen += prefills[i].total_seqlen()
                todos.append(prefills.pop(i))
        todo = BatchPrefill.merge(todos)
        if todo.check_kvcache_available():
            return todo.step()
        else:
            self.pending_buffer.append(todo)
            print("Pending prefill due to lack of KV cache, waiting for more space")
            return None

    def handle_decode(self, decodes: list[BatchDecode]) -> Optional[list[ExecResult]]:
        decodes.sort(key=lambda task: task.bsz())
        total_bsz = 0
        todos: list[BatchDecode] = []
        for i in reversed(range(len(decodes))):
            if total_bsz + decodes[i].bsz() <= self.max_total_bsz:
                total_bsz += decodes[i].bsz()
                todos.append(decodes.pop(i))
        todo = BatchDecode.merge(todos)
        if self.is_scheduler:
            persists: list[BatchExec] = []
            while not todo.check_kvcache_available():
                pending = todo.pop()
                pending.persist()
                print(f"Offload {pending.task_datas[0].task_id} to CPU")
                persists.append(pending)
            results: list[ExecResult] = [
                BatchPersist(
                    [
                        task_data
                        for persist in persists
                        for task_data in persist.task_datas
                    ]
                )
            ]
            self.pending_buffer.extend(persists)
            if todo.bsz() > 0:
                results.extend(todo.step())
            return results
        else:
            if todo.check_kvcache_available():
                return todo.step()
            else:
                self.pending_buffer.append(todo)
                print("Pending decode due to lack of KV cache, waiting for more space")
                return None

    def handle_trace(self, traces: list[SingleTrace]) -> Optional[list[ExecResult]]:
        return traces.pop().step()  # TODO: deal with pending

    def add_event(self, event: EngineEvent) -> None:
        self.receiver.put(event)
