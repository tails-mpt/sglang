from __future__ import annotations

import dataclasses
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Callable, Optional

import torch
from numpy import float64

from sglang.srt.mem_cache.base_prefix_cache import (
    DecLockRefParams,
    EvictParams,
    IncLockRefResult,
    InsertParams,
    InsertResult,
    MatchPrefixParams,
    MatchResult,
)

if TYPE_CHECKING:
    from sglang.srt.mem_cache.hybrid_cache.hybrid_radix_cache import (
        HybridRadixCache,
        HybridTreeNode,
    )


BASE_COMPONENT_NAME = "full"

_LAST_ACCESS_TIME_COUNTER_FLOAT = float64(1.0)
_COMPONENT_UUID_COUNTER = 1


@dataclasses.dataclass
class ComponentData:
    value: Optional[torch.Tensor] = None
    lock_ref: int = 0
    metadata: dict[str, Any] = dataclasses.field(default_factory=dict)


def get_last_access_time() -> float64:
    global _LAST_ACCESS_TIME_COUNTER_FLOAT
    ret = _LAST_ACCESS_TIME_COUNTER_FLOAT
    _LAST_ACCESS_TIME_COUNTER_FLOAT += 1.0
    return ret


def gen_component_uuid() -> int:
    global _COMPONENT_UUID_COUNTER
    _COMPONENT_UUID_COUNTER += 1
    return _COMPONENT_UUID_COUNTER


class TreeComponent(ABC):
    def __init__(self, cache: HybridRadixCache):
        self.cache = cache

    def node_has_component_data(self, node: HybridTreeNode) -> bool:
        return node.component_value(self.name) is not None

    def value_len(self, node: HybridTreeNode) -> int:
        value = node.component_value(self.name)
        return len(value) if value is not None else 0

    @property
    @abstractmethod
    def name(self) -> str:
        """Component identifier, e.g. "full", "swa", "mamba"."""
        ...

    @abstractmethod
    def create_match_validator(self) -> Callable[[HybridTreeNode], bool]:
        """Return a per-match stateful predicate that decides whether a node
        is a valid match boundary for this component.
        Called once per match_prefix; the returned closure may carry state.
        - Full: always True (every node is valid).
        - SWA: tracks accumulated length since last gap; returns True only
          when the contiguous window reaches swa_sliding_window_size.
        - Mamba: returns True iff the node has mamba component data."""
        ...

    def finalize_match_result(
        self,
        result: MatchResult,
        params: MatchPrefixParams,
        value_chunks: list[torch.Tensor],
        best_value_len: int,
    ) -> MatchResult:
        """Post-process the match result after prefix matching completes.
        - Full: pass through unchanged.
        - Mamba: performs copy-on-write — allocates a new mamba slot, copies
          the matched node's mamba state into the request pool, and records
          branching_seqlen in result."""
        return result

    def update_component_on_insert_overlap(
        self,
        node: HybridTreeNode,
        prefix_len: int,
        total_prefix_len: int,
        value_slice: torch.Tensor,
        params: InsertParams,
    ) -> None:
        """Called per-node when an insert's key overlaps an existing node,
        allowing the component to attach or extend its own data.
        - Full: no-op.
        - SWA: if the insert covers the swa_evicted_seqlen boundary, clones
          the relevant value_slice into the node's swa component value,
          splits the node if needed, and updates LRU / evictable sizes.
        - Mamba: frees the portion of memory beyond the overlap point so the
          new insert can replace it."""
        pass

    def get_tombstone_prefix_len_for_insert(
        self, total_prefix_len: int, new_key_len: int, params: InsertParams
    ) -> int:
        """Return the component-specific prefix length that already has data,
        so the insert logic knows where the tombstone region begins.
        - Full & Mamba: always 0 (no tombstone contribution).
        - SWA: if swa_evicted_seqlen falls within the current insert range,
          returns that position so nodes before it are treated as tombstones
          (no full KV data, only swa component data)."""
        return 0

    def commit_insert_component_data(
        self,
        node: HybridTreeNode,
        is_new_leaf: bool,
        params: InsertParams,
        result: InsertResult,
    ) -> None:
        """Finalize component data on the target (leaf) node after the insert
        walk completes. Called once per insert.
        - Full: no-op (full data is handled by _add_new_node).
        - SWA: for new leaves, sets the swa component value to full_value,
          inserts into the swa LRU list, and increments evictable size.
        - Mamba: sets the mamba component value from params, inserts into
          mamba LRU list, and increments evictable size. If the node already
          has mamba data, resets its LRU position instead."""
        pass

    @abstractmethod
    def redistribute_on_node_split(
        self, new_parent: HybridTreeNode, child: HybridTreeNode
    ):
        """Redistribute component data between new_parent and child when a
        node is split. new_parent is the newly created prefix node.
        - Full: copies child's lock_ref to new_parent.
        - SWA: slices (or clones) the swa value for new_parent, copies
          lock_ref and component_uuid metadata, then syncs child's swa
          value with its (now-trimmed) full_value.
        - Mamba: sets new_parent's mamba value to None and lock_ref to 0
          (mamba data stays on the original leaf, not on prefix nodes)."""
        ...

    @abstractmethod
    def evict_component(self, node: HybridTreeNode, is_leaf: bool) -> int:
        """Free this component's KV resources on a node being evicted.
        For internal (non-leaf) nodes: free memory and tombstone the value
        (set to None); the node structure is kept.
        For leaf nodes: free memory; the node will be deleted by caller.
        Returns the number of tokens/slots freed.
        - Full: frees full_value via token_to_kv_pool_allocator.
        - SWA: frees swa value via swa_token_to_kv_pool_allocator;
          only tombstones on internal nodes.
        - Mamba: frees mamba value via mamba_token_to_kv_pool_allocator;
          only tombstones on internal nodes."""
        ...

    def eviction_priority(self, is_leaf: bool) -> int:
        """Eviction priority on this node type. Higher = evicted later.
        When a component is evicted, all other components with equal or
        lower priority on the same node are also cascade-evicted.
        Leaf: all components equal (0) — evicting any cascades to all.
        Internal: full=1, others=0 — evicting mamba cascades to swa,
        but not full.
        - Full: leaf=0, internal=1.
        - Mamba/SWA: always 0."""
        return 0

    @abstractmethod
    def drive_eviction(self, params: EvictParams, tracker: dict[str, int]) -> None:
        """Drive eviction from this component's LRU list.
        Each component extracts its own request from params, walks its own
        LRU, evicts, and calls cache._cascade_evict for priority cascade.
        Updates the shared tracker with freed amounts for all components.
        - Full: walks leaf LRU, evicts full then cascades entire leaf.
        - Mamba: walks full LRU; tombstones internal nodes (with cascade
          to equal-priority components like swa), cascades leaves to all."""
        ...

    @abstractmethod
    def acquire_component_lock(
        self, node: HybridTreeNode, result: IncLockRefResult
    ) -> IncLockRefResult:
        """Increment lock_ref for this component, protecting nodes from
        eviction. Updates evictable → protected size on first lock.
        - Full: path-lock — walks from node up to root, incrementing
          lock_ref on every ancestor.
        - SWA: path-lock — walks upward collecting swa values until the
          sliding window is filled; records a component_uuid at the
          boundary for release_component_lock to know where to stop.
        - Mamba: single-node lock — only increments lock_ref on the
          node itself (mamba state is per-leaf, not per-path)."""
        ...

    @abstractmethod
    def release_component_lock(
        self, node: HybridTreeNode, params: Optional[DecLockRefParams]
    ) -> None:
        """Decrement lock_ref for this component, un-protecting nodes.
        Updates protected → evictable size when lock_ref drops to 0.
        - Full: path-unlock — walks from node up to root, decrementing
          lock_ref on every ancestor.
        - SWA: path-unlock — walks upward, stopping at the node whose
          component_uuid matches the one recorded during acquire.
        - Mamba: single-node unlock — only decrements lock_ref on the
          node itself."""
        ...
