import torch

from deserve_worker.kvcache.paged.kvcache import PagedKVCache
from deserve_worker.kvcache.paged.page_pool import CpuPagePool, GpuPagePool


class PortableKVCacheManager:
    def __init__(
        self,
        gpu_page_pool: GpuPagePool,
        cpu_page_pool: CpuPagePool,
    ):
        self.gpu_page_pool = gpu_page_pool
        self.cpu_page_pool = cpu_page_pool
        self.stream = torch.cuda.Stream()  # type: ignore

    def copy_gpu_to_cpu(
        self, paged_kvcache: PagedKVCache[GpuPagePool]
    ) -> PagedKVCache[CpuPagePool]:
        """
        Only do the copy. The life cycle of KV cache is managed by the caller.
        """

        gpu_page_table = paged_kvcache.page_table
        page_table_len = gpu_page_table.shape[0]
        cpu_page_table = self.cpu_page_pool.alloc(page_table_len)
        if cpu_page_table is None:
            raise RuntimeError("Failed to allocate page tables")
        with torch.cuda.stream(self.stream):
            self.cpu_page_pool.pages[:, cpu_page_table, :, :, :, :] = (
                self.gpu_page_pool.pages[:, gpu_page_table, :, :, :, :].to("cpu")
            )

        return PagedKVCache(
            page_table=cpu_page_table,
            pool=self.cpu_page_pool,
        )

    def copy_cpu_to_gpu(
        self, paged_kvcache: PagedKVCache[CpuPagePool]
    ) -> PagedKVCache[GpuPagePool]:
        """
        Only do the copy. The life cycle of KV cache is managed by the caller.
        """
        cpu_page_table = paged_kvcache.page_table
        page_table_len = cpu_page_table.shape[0]
        gpu_page_table = self.gpu_page_pool.alloc(page_table_len)
        if gpu_page_table is None:
            raise RuntimeError("Failed to allocate page tables")
        with torch.cuda.stream(self.stream):
            self.gpu_page_pool.pages[:, gpu_page_table, :, :, :, :] = (
                self.cpu_page_pool.pages[:, cpu_page_table, :, :, :, :].to("cuda")
            )
        return PagedKVCache(
            page_table=gpu_page_table,
            pool=self.gpu_page_pool,
        )

    def synchronize(self) -> None:
        self.stream.synchronize()  # type: ignore
