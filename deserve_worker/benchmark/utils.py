layers = [
    "llama-3-70b-instruct-slice/tok_embeddings",
    *[f"llama-3-70b-instruct-slice/layers.{i}" for i in range(0, 80)],
    "llama-3-70b-instruct-slice/norm",
    "llama-3-70b-instruct-slice/output",
]


def convert_name_to_id(name: str) -> int:
    if name == "emb":
        return 0
    elif name.isdigit():
        return int(name) + 1
    elif name == "norm":
        return 81
    elif name == "output":
        return 82
    else:
        raise ValueError("Invalid layer name")
