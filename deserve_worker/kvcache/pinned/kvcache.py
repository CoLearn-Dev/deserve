from deserve_worker.kvcache.pinned.pinned_memory import PinnedMemory


class PinnedKVCache:
    def __init__(self, memory: PinnedMemory, ptr: int, size: int):
        self.memory = memory
        self.ptr = ptr
        self.size = size
