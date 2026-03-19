from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Optional

import torch

from sglang.srt.mem_cache.base_prefix_cache import (
    DecLockRefParams,
    EvictParams,
    IncLockRefResult,
    InsertParams,
    InsertResult,
    MatchPrefixParams,
    MatchResult,
)
from sglang.srt.mem_cache.hybrid_cache.tree_component import (
    TreeComponent,
    get_last_access_time,
)
from sglang.srt.server_args import get_global_server_args

if TYPE_CHECKING:
    from sglang.srt.mem_cache.hybrid_cache.hybrid_radix_cache import HybridTreeNode


class MambaComponent(TreeComponent):
    @property
    def name(self) -> str:
        return "mamba"

    def create_match_validator(self) -> Callable[[HybridTreeNode], bool]:
        name = self.name
        return lambda node: node.component_value(name) is not None

    def finalize_match_result(
        self,
        result: MatchResult,
        params: MatchPrefixParams,
        value_chunks: list[torch.Tensor],
        best_value_len: int,
    ) -> MatchResult:
        cow_mamba = params.cow_mamba
        req = params.req
        last_node = result.last_device_node

        if len(value_chunks) > best_value_len:
            chunk_size = get_global_server_args().mamba_cache_chunk_size
            aligned_seqlen = (
                sum(len(v) for v in value_chunks) // chunk_size
            ) * chunk_size
            branching_seqlen = aligned_seqlen if aligned_seqlen > 0 else None
        else:
            branching_seqlen = None

        mamba_value = last_node.component_value(self.name)
        if cow_mamba and mamba_value is not None:
            assert req is not None
            if req.mamba_pool_idx is None:
                dst_index = self.cache.req_to_token_pool.mamba_pool.alloc(1)
                if dst_index is None:
                    self.cache.inc_lock_ref(last_node)
                    self.cache.evict(EvictParams(num_tokens=0, mamba_num=1))
                    dst_index = self.cache.req_to_token_pool.mamba_pool.alloc(1)
                    self.cache.dec_lock_ref(last_node)
                    assert dst_index is not None, "Can not alloc mamba cache"
                self.cache.req_to_token_pool.mamba_pool.copy_from(
                    mamba_value, dst_index
                )
                req.mamba_pool_idx = dst_index[0]
            else:
                dst_index = req.mamba_pool_idx.unsqueeze(0)
                self.cache.req_to_token_pool.mamba_pool.copy_from(
                    mamba_value, dst_index
                )

        return result._replace(mamba_branching_seqlen=branching_seqlen)

    def update_component_on_insert_overlap(
        self,
        node: HybridTreeNode,
        prefix_len: int,
        total_prefix_len: int,
        value_slice: torch.Tensor,
        params: InsertParams,
    ) -> None:
        if params.prev_prefix_len < total_prefix_len + prefix_len:
            start = max(0, params.prev_prefix_len - total_prefix_len)
            self.cache.token_to_kv_pool_allocator.free(value_slice[start:])

    def commit_insert_component_data(
        self,
        node: HybridTreeNode,
        is_new_leaf: bool,
        params: InsertParams,
        result: InsertResult,
    ) -> None:
        assert params.mamba_value is not None
        if is_new_leaf:
            node.set_component_value(self.name, params.mamba_value)
            self.cache.lru_lists[self.name].insert_mru(node)
            self.cache.component_evictable_size_[self.name] += len(params.mamba_value)
            return
        if node.component_value(self.name) is None:
            node.set_component_value(self.name, params.mamba_value)
            self.cache.lru_lists[self.name].insert_mru(node)
            self.cache.component_evictable_size_[self.name] += len(params.mamba_value)
            node.last_access_time = get_last_access_time()
            return
        self.cache.lru_lists[self.name].reset_node_mru(node)
        node.last_access_time = get_last_access_time()
        result.mamba_exist = True

    def redistribute_on_node_split(
        self, new_parent: HybridTreeNode, child: HybridTreeNode
    ):
        new_parent.set_component_value(self.name, None)
        new_parent.component(self.name).lock_ref = 0

    def evict_component(self, node: HybridTreeNode, is_leaf: bool) -> int:
        value = node.component_value(self.name)
        self.cache.req_to_token_pool.mamba_pool.free(value)
        freed = len(value)
        self.cache.component_evictable_size_[self.name] -= freed
        if not is_leaf:
            node.set_component_value(self.name, None)
        return freed

    def drive_eviction(self, params: EvictParams, tracker: dict[str, int]) -> None:
        request = params.mamba_num
        lru = self.cache.lru_lists[self.name]
        x = lru.get_lru_no_lock()
        while tracker[self.name] < request and x is not None and lru.in_list(x):
            assert x.component_value(self.name) is not None
            if len(x.children) > 0:
                # Internal: evict self, cascade to equal-priority components
                x_next = lru.get_prev_no_lock(x)
                self.cache._evict_component_and_detach_lru(
                    x, self, is_leaf=False, tracker=tracker
                )
                self.cache._cascade_evict(x, self, tracker)
                x = x_next
            else:
                # Leaf: evict self, cascade to all components
                self.cache._evict_component_and_detach_lru(
                    x, self, is_leaf=True, tracker=tracker
                )
                self.cache._cascade_evict(x, self, tracker)
                x = lru.get_lru_no_lock()

    def acquire_component_lock(
        self, node: HybridTreeNode, result: IncLockRefResult
    ) -> IncLockRefResult:
        value = node.component_value(self.name)
        if value is not None:
            if node.component(self.name).lock_ref == 0:
                self.cache.component_evictable_size_[self.name] -= len(value)
                self.cache.component_protected_size_[self.name] += len(value)
            node.component(self.name).lock_ref += 1
        return result

    def release_component_lock(
        self, node: HybridTreeNode, params: Optional[DecLockRefParams]
    ) -> None:
        value = node.component_value(self.name)
        if value is not None:
            assert node.component(self.name).lock_ref > 0
            if node.component(self.name).lock_ref == 1:
                self.cache.component_evictable_size_[self.name] += len(value)
                self.cache.component_protected_size_[self.name] -= len(value)
            node.component(self.name).lock_ref -= 1
