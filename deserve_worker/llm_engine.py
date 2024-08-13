import queue
from typing import Optional, cast

import torch

from deserve_worker.execution.result import BatchUpdate, ExecResult, TraceResult
from deserve_worker.layer_storage import LayerStorage
from deserve_worker.task import TaskData
from deserve_worker.trace import OpId

from .execution.exec import BatchDecode, BatchExec, BatchPrefill, SingleTrace
from .kvcache.kvcache import KVCacheManager, main_device

EOS_TOKEN_ID = 128001  # for llama 3 only
STOP_TOKEN_IDS = [128001, 128009]


class LLMEngine:
    def __init__(
        self,
        max_total_bsz: int,
        max_total_seqlen: int,
        sender: queue.Queue[ExecResult],
    ):
        self.max_total_bsz = max_total_bsz
        self.max_total_seqlen = max_total_seqlen
        self.sender = sender
        self.receiver = queue.Queue[BatchExec]()
        self.prefill_buffer: list[BatchPrefill] = []
        self.decode_buffer: list[BatchDecode] = []
        self.trace_buffer: list[SingleTrace] = []

    def run(self) -> None:
        while True:
            self.fill_buffer()
            if len(self.trace_buffer) != 0:
                responses = self.handle_trace(self.trace_buffer)
            elif len(self.prefill_buffer) != 0:
                responses = self.handle_prefill(self.prefill_buffer)
            elif len(self.decode_buffer) != 0:
                responses = self.handle_decode(self.decode_buffer)
            else:
                raise NotImplementedError("Unknown request type")
            for response in responses:
                self.sender.put(response)

    def sort_request(self, request: BatchExec) -> None:
        if isinstance(request, BatchDecode):
            self.decode_buffer.append(request)
        elif isinstance(request, BatchPrefill):
            self.prefill_buffer.append(request)
        elif isinstance(request, SingleTrace):
            self.trace_buffer.append(request)
        else:
            raise NotImplementedError("Unknown request type")

    def fill_buffer(self) -> None:
        if (
            len(self.prefill_buffer) == 0
            and len(self.decode_buffer) == 0
            and len(self.trace_buffer) == 0
        ):
            self.sort_request(self.receiver.get())
        while True:
            try:
                self.sort_request(self.receiver.get(block=False))
            except queue.Empty:
                break

    def handle_prefill(self, prefills: list[BatchPrefill]) -> list[ExecResult]:
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
        return todo.step()

    def handle_decode(self, decodes: list[BatchDecode]) -> list[ExecResult]:
        decodes.sort(key=lambda task: task.bsz())

        total_bsz = 0
        todos: list[BatchDecode] = []
        for i in reversed(range(len(decodes))):
            if total_bsz + decodes[i].bsz() <= self.max_total_bsz:
                total_bsz += decodes[i].bsz()
                todos.append(decodes.pop(i))
        todo = BatchDecode.merge(todos)
        return todo.step()

    def handle_trace(self, traces: list[SingleTrace]) -> list[ExecResult]:
        return traces.pop().step()

    def add_request(self, request: BatchExec) -> None:
        self.receiver.put(request)
