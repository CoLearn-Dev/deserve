from typing import Optional

from transformers import PreTrainedTokenizer  # type: ignore

from datasets import load_dataset  # type: ignore

from .utils import Workload, cache, llama3_tokenizer


class OasstNode:
    def __init__(self, message: dict[str, str]):
        if message["role"] == "prompter":
            message["role"] = "user"
        self.message = message
        self.children: list[OasstNode] = []

    def add_child(self, node: "OasstNode") -> None:
        self.children.append(node)

    def traverse_for_workload(
        self, messages: list[dict[str, str]], workload: list[list[dict[str, str]]]
    ) -> None:
        messages.append(self.message)
        if self.message["role"] == "user":
            workload.append(messages)
        for child in self.children:
            child.traverse_for_workload(messages.copy(), workload)


class OasstTree:
    def __init__(self) -> None:
        self.roots: list[OasstNode] = []
        self.id2node: dict[str, OasstNode] = {}

    def add_node(
        self, message_id: str, parent_id: Optional[str], text: str, role: str
    ) -> None:
        message = {"content": text, "role": role}
        node = OasstNode(message)
        self.id2node[message_id] = node
        if parent_id is None:
            self.roots.append(node)
        else:
            parent = self.id2node[parent_id]
            parent.add_child(node)


class Oasst1Workload(Workload):
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


class Oasst1Dataset:
    def __init__(self) -> None:
        self.raw_data = load_dataset("OpenAssistant/oasst1")

    @cache()
    def into_workload(self) -> Workload:
        merged_data = list(self.raw_data["train"]) + list(self.raw_data["validation"])  # type: ignore
        tree = OasstTree()
        for row in merged_data:
            tree.add_node(row["message_id"], row["parent_id"], row["text"], row["role"])  # type: ignore

        workload: list[list[dict[str, str]]] = []
        for root in tree.roots:
            root.traverse_for_workload([], workload)
        return Oasst1Workload(workload, llama3_tokenizer)


if __name__ == "__main__":
    workload = Oasst1Dataset().into_workload()
