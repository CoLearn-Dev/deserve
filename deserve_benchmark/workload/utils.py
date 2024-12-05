from abc import ABC, abstractmethod
from typing import Any, Callable

from transformers import AutoTokenizer  # type: ignore

llama3_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")


class Workload(ABC):
    @abstractmethod
    def get(self, offset: int, length: int) -> list[str]:
        pass

    @abstractmethod
    def size(self) -> int:
        pass


def cache(root_dir: str = "tmp/") -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """
    Cache the return value of a **method function** in disk using pickle.
    The first argument of the function must be `self`.
    If the file does not exist, call the function and store the return value in the file named `{class_name}_{func_name}_{args}_{kwargs}` in `root_dir`.
    if `enable` is False, the function will not be cached.
    Raise error if the `root_dir` does not exist.
    """
    import functools
    import os
    import pickle

    def decorator(f: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(f)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if not os.path.exists(root_dir) and root_dir != "":
                print(f"Cache root dir {root_dir} does not exist. Creating...")
                os.makedirs(root_dir, exist_ok=True)
            cache_path = os.path.join(
                root_dir,
                f"{args[0].__class__.__name__}_{f.__name__}_{args[1:]}_{kwargs}",
            )
            if os.path.exists(cache_path):
                with open(cache_path, "rb") as fi:
                    return pickle.load(fi)
            else:
                ret = f(*args, **kwargs)
                with open(cache_path, "wb") as fi:
                    pickle.dump(ret, fi)
                return ret

        return wrapper

    return decorator
