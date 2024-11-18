import time

import torch

from deserve_worker.kvcache.paged.chunk_pool import ChunkHandle, CpuChunkPool
from deserve_worker.kvcache.paged.kvcache import PagedKVCache
from deserve_worker.kvcache.paged.page_pool import CpuPagePool
from deserve_worker.kvcache.virtual import VirtualPagedKVCache, VirtualPagePool


class KVCacheManager:
    def __init__(
        self,
        virtual_page_pool: VirtualPagePool,
        cpu_chunk_pool: CpuChunkPool,
    ):
        self.virtual_page_pool = virtual_page_pool
        self.cpu_chunk_pool = cpu_chunk_pool
        self.stream = torch.cuda.Stream()  # type: ignore

    def copy_gpu_to_cpu(
        self, paged_kvcache: PagedKVCache[VirtualPagePool], task_id: str
    ) -> ChunkHandle:
        """
        Only do the copy. The life cycle of KV cache is managed by the caller.
        """

        gpu_page_table = paged_kvcache.page_table
        page_table_len = gpu_page_table.shape[0]
        cpu_chunk_handle = self.cpu_chunk_pool.alloc(page_table_len)
        if cpu_chunk_handle is None:
            raise RuntimeError("Failed to allocate chunks")
        cpu_chunk = cpu_chunk_handle.retrieve()
        occupied_size = page_table_len * self.cpu_chunk_pool.num_layers
        # print(
        #     f"[{task_id}] copy_gpu_to_cpu {self.virtual_page_pool.pages[:, gpu_page_table, :, :, :, :].max()}"
        # )
        cpu_chunk[:occupied_size].copy_(
            self.virtual_page_pool.pages[:, gpu_page_table, :, :, :, :].view(
                occupied_size,
                *self.cpu_chunk_pool.per_token_shape,
            ),
            # non_blocking=True,
        )

        return cpu_chunk_handle

    def copy_cpu_to_gpu(
        self, chunk_handle: ChunkHandle, task_id: str
    ) -> VirtualPagedKVCache:
        """
        Only do the copy. The life cycle of KV cache is managed by the caller.
        """
        page_table_len = chunk_handle.initial_size
        gpu_page_table = self.virtual_page_pool.alloc(page_table_len)
        if gpu_page_table is None:
            raise RuntimeError("Failed to allocate page tables")
        self.virtual_page_pool.pages[:, gpu_page_table, :, :, :, :] = (
            chunk_handle.retrieve()[
                : page_table_len * self.cpu_chunk_pool.num_layers, :, :, :, :
            ]
            .view(
                self.cpu_chunk_pool.num_layers,
                page_table_len,
                *self.cpu_chunk_pool.per_token_shape,
            )
            .to("cuda")
        )
        return VirtualPagedKVCache(
            page_table=gpu_page_table,
            isswap=gpu_page_table >= self.virtual_page_pool.num_pages_main,
            pool=self.virtual_page_pool,
        )

    def synchronize(self) -> None:
        self.stream.synchronize()  # type: ignore
