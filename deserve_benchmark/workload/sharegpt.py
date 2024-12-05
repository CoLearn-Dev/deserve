import json
from typing import Any

from transformers import PreTrainedTokenizer  # type: ignore

from .utils import Workload, cache, llama3_tokenizer


class ShareGptWorkload(Workload):
    def __init__(
        self, raw_data: list[list[dict[str, str]]], tokenizer: PreTrainedTokenizer
    ) -> None:
        self.raw_data = raw_data
        self.tokenizer = tokenizer

    def get(self, offset: int, length: int) -> list[str]:
        if offset >= len(self.raw_data):
            return []
        limit = min(offset + length, len(self.raw_data))
        return [
            self.tokenizer.apply_chat_template(
                conversation,
                tokenize=False,
                add_generation_timestamp_token=False,
            )
            for conversation in self.raw_data[offset:limit]
        ]

    def size(self) -> int:
        return len(self.raw_data)


class ShareGptDataset:
    def __init__(self) -> None:
        with open(
            "datasets/ShareGPT_Vicuna_unfiltered/ShareGPT_V3_unfiltered_cleaned_split_no_imsorry.json",
            "r",
        ) as f:
            self.raw_data = json.load(f)

    @cache()
    def into_workload(self) -> Workload:
        workload = []
        for conversation in self.raw_data:
            messages = []
            for message in conversation["conversations"]:
                role = "user" if message["from"] == "human" else "assistant"
                messages.append({"content": message["value"], "role": role})
            workload.append(messages)
        return ShareGptWorkload(
            workload,
            llama3_tokenizer,
        )


if __name__ == "__main__":
    dataset = ShareGptDataset()
    workload = dataset.into_workload()
