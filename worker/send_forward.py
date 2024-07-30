import requests
from fleece_network import dumps
import uuid

plan_8b = [
    ["local", [
        "llama-3-8b-instruct-slice/tok_embeddings",
        *[f"llama-3-8b-instruct-slice/layers.{i}" for i in range(0, 32)],
        "llama-3-8b-instruct-slice/norm",
        "llama-3-8b-instruct-slice/output",
    ]]
]

plan_70b = [
    ["4d902f27-8a42-4b9a-b7c4-67a6d3b3ed52", [
        "llama-3-70b-instruct-slice/tok_embeddings",
        *[f"llama-3-70b-instruct-slice/layers.{i}" for i in range(0, 40)],
    ]],
    ["3b9669de-113d-48b5-8ec4-0ed4d26bc083", [
        *[f"llama-3-70b-instruct-slice/layers.{i}" for i in range(40, 80)],
        "llama-3-70b-instruct-slice/norm",
        "llama-3-70b-instruct-slice/output",
    ]],
]

input = [[128000, 128006, 882, 128007, 271, 12840, 374, 279, 11363, 315, 1253, 13767, 1082, 30, 128009, 128006, 78191, 128007, 271], [128000, 128006, 9125, 128007, 271, 38195, 4320, 449, 14433, 39342, 128009, 128006, 882, 128007, 271, 40, 1097, 2133, 311, 12366, 11, 1148, 1288, 358, 1518, 30, 128009, 128006, 78191, 128007, 271],
         [128000, 128006, 9125, 128007, 271, 38195, 4320, 449, 100166, 128009, 128006, 882, 128007, 271, 4438, 311, 733, 505, 27647, 311, 12551, 30, 128009, 128006, 78191, 128007, 271], [128000, 128006, 882, 128007, 271, 12840, 374, 279, 11363, 315, 1253, 13767, 1082, 30, 128009, 128006, 78191, 128007, 271]]

tensors = {}
metadata = {
    "task_id": str(uuid.uuid4()),
    "plan": plan_8b,
    "step": 0,
    "round": 0,
    "max_total_len": 1024,
    "temperature": 0,
    "payload": input,
}

data = dumps(tensors, metadata)
r = requests.post("http://127.0.0.1:8080/forward", data=data)
