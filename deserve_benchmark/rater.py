import datetime
import math
import threading
from dataclasses import dataclass
from typing import Any, Optional

import tiktoken
from pydantic import BaseModel
from transformers import AutoTokenizer  # type: ignore

from deserve_benchmark.workload.utils import Workload

llama3_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-70B")
openai_tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")


class RaterRuntimeError(Exception):
    pass


class RaterTimeLimitExceeded(Exception):
    pass


@dataclass
class Request:
    id: int
    history: str


def count_llama_tokens(payload: str) -> int:
    return len(llama3_tokenizer.tokenize(payload))


def count_openai_tokens(payload: str) -> int:
    return len(openai_tokenizer.encode(payload))


@dataclass
class Response:
    id: int
    payload: str
    finished: bool

    def count_openai_tokens(self) -> int:
        return count_openai_tokens(self.payload)

    def count_llama_tokens(self) -> int:
        return count_llama_tokens(self.payload)


class BufferedString(BaseModel):
    buffer: str = ""
    time_last_added: Optional[datetime.datetime] = None
    finished: bool = False

    def insert(
        self, offset: int, payload: str, time: datetime.datetime, finished: bool
    ) -> None:
        self.time_last_added = time
        self.finished = finished
        if offset == -1:
            self.buffer = self.buffer + payload
        elif offset <= len(self.buffer):
            self.buffer = self.buffer[:offset] + payload
        else:
            self.buffer = self.buffer + " " * (offset - len(self.buffer)) + payload

    def is_empty(self) -> bool:
        return self.buffer == ""


class BufferedResponses(BaseModel):
    content: list[BufferedString] = []

    def insert(
        self,
        index: int,
        offset: int,
        payload: str,
        time: datetime.datetime,
        finished: bool,
    ) -> None:
        while index >= len(self.content):
            self.content.append(BufferedString())
        self.content[index].insert(offset, payload, time, finished)

    def is_empty(self, index: int) -> bool:
        if index >= len(self.content):
            return True
        return self.content[index].is_empty()

    def into_timestamped_responses(
        self,
    ) -> list[tuple[str, Optional[datetime.datetime]]]:
        return list(map(lambda x: (x.buffer, x.time_last_added), self.content))


class Rater:
    def __init__(
        self, workload: Workload, time_limit: int, trace: bool, warmup: int = 0
    ):
        self.workload = workload
        self.ptr = 0
        self.time_limit = time_limit
        self.time_first_get: Optional[datetime.datetime] = None
        self.time_last_post: Optional[datetime.datetime] = None
        self.buffered_responses = BufferedResponses()
        self.input_throughput_history: dict[int, int] = {}
        self.output_throughput_history: dict[int, int] = {}
        self.requests_finished_history: dict[int, int] = {}
        self.input_throughput_total = 0
        self.output_throughput_total = 0
        self.requests_finished_total = 0
        self.lock = threading.Lock()
        self.trace = trace
        assert (warmup >= 0 and warmup < time_limit) or time_limit < 0
        self.warmup = warmup

    def get(self, size: int) -> list[Request]:
        if self.time_first_get is None:
            self.time_first_get = datetime.datetime.now()
        texts = self.workload.get(self.ptr, size)
        self.ptr += len(texts)
        return [
            Request(id=id, history=history)
            for id, history in enumerate(texts, start=self.ptr - len(texts))
        ]

    def post(self, response: Response) -> None:
        if self.time_first_get is None:
            raise RaterRuntimeError("First get must be called before post")
        now = datetime.datetime.now()
        self.time_last_post = now
        if self.time_limit > 0 and now - self.time_first_get > datetime.timedelta(
            seconds=self.time_limit
        ):
            raise RaterTimeLimitExceeded("Time limit exceeded")
        time_delta = int((now - self.time_first_get).total_seconds())

        with self.lock:
            if time_delta not in self.input_throughput_history:
                self.input_throughput_history[time_delta] = 0
            if self.buffered_responses.is_empty(response.id):
                tokens = count_llama_tokens(self.workload.get(response.id, 1)[0]) + 1
                self.input_throughput_history[time_delta] += tokens
                self.input_throughput_total += tokens

            if time_delta not in self.output_throughput_history:
                self.output_throughput_history[time_delta] = 0
            self.output_throughput_history[time_delta] += response.count_llama_tokens()
            self.output_throughput_total += response.count_llama_tokens()
            if time_delta not in self.requests_finished_history:
                self.requests_finished_history[time_delta] = 0
            if response.finished:
                self.requests_finished_history[time_delta] += 1
                self.requests_finished_total += 1

            self.buffered_responses.insert(
                response.id,
                -1,
                response.payload,
                now,
                response.finished,
            )

    def dump(self) -> dict[str, Any]:
        time_first_get = (
            self.time_first_get.timestamp() if self.time_first_get else None
        )
        time_last_post = (
            self.time_last_post.timestamp() if self.time_last_post else None
        )
        output_throughput_prefix_sum = {0: 0}
        input_throughput_prefix_sum = {0: 0}
        warmup_input_throughput = 0.0
        warmup_output_throughput = 0.0

        if time_first_get is None or time_last_post is None:
            time_used = None
            real_input_throughput = None
            real_output_throughput = None
        else:
            time_used = time_last_post - time_first_get
            time_used_int = math.ceil(time_used)
            real_input_throughput = self.input_throughput_total / time_used
            real_output_throughput = self.output_throughput_total / time_used

            for i in range(1, time_used_int):
                output_throughput_prefix_sum[i] = output_throughput_prefix_sum[
                    i - 1
                ] + self.output_throughput_history.get(i, 0)

            for i in range(1, time_used_int):
                input_throughput_prefix_sum[i] = input_throughput_prefix_sum[
                    i - 1
                ] + self.input_throughput_history.get(i, 0)

            for i in range(self.warmup, time_used_int):
                warmup_input_throughput += self.input_throughput_history.get(i, 0)
                warmup_output_throughput += self.output_throughput_history.get(i, 0)
            if time_used > self.warmup:
                warmup_input_throughput /= time_used - self.warmup
                warmup_output_throughput /= time_used - self.warmup

        result: dict[str, Any] = {
            "time_limit": self.time_limit,
            "time_warmup": self.warmup,
            "time_first_get": time_first_get,
            "time_last_post": time_last_post,
            "time_used": time_used,
            "real_input_throughput": real_input_throughput,
            "real_output_throughput": real_output_throughput,
            "standard_input_throughput": self.input_throughput_total / self.time_limit,
            "standard_output_throughput": self.output_throughput_total
            / self.time_limit,
            "warmup_input_throughput": warmup_input_throughput,
            "warmup_output_throughput": warmup_output_throughput,
            "input_throughput_total": self.input_throughput_total,
            "input_throughput_history": self.input_throughput_history,
            "input_throughput_prefix_sum": input_throughput_prefix_sum,
            "output_throughput_total": self.output_throughput_total,
            "output_throughput_history": self.output_throughput_history,
            "output_throughput_prefix_sum": output_throughput_prefix_sum,
            "requests_finished_total": self.requests_finished_total,
            "requests_finished_history": self.requests_finished_history,
        }
        if self.trace:
            result["trace"] = [
                {
                    "request": history,
                    "response": response.buffer,
                }
                for history, response in zip(
                    self.workload.get(0, len(self.buffered_responses.content)),
                    self.buffered_responses.content,
                )
            ]
        return result
