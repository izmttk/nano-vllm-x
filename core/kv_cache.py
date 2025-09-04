from typing import Optional
from collections import abc
import torch
import math
import heapq
import time

from core.common import Sequence, SequenceStatus

# KVCachePool should be instantiated in each worker
class KVCachePool:
    def __init__(
        self,
        dtype: torch.dtype,
        device: torch.device,
        num_tokens: int,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        start_layer: Optional[int] = None,
        end_layer: Optional[int] = None,
        page_size: int = 1,
    ):
        self.dtype = dtype
        self.device = device
        self.num_tokens = num_tokens
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.start_layer = start_layer or 0
        self.end_layer = end_layer or num_layers - 1

        self.page_size = page_size
        self.num_pages = math.ceil(self.num_tokens / self.page_size)

        self.create_cache_pool()

    def create_cache_pool(self):
        self.k_cache = torch.zeros(
            (self.num_layers, self.num_pages *
             self.page_size, self.num_heads, self.head_dim),
            dtype=self.dtype,
            device=self.device,
        )
        self.v_cache = torch.zeros(
            (self.num_layers, self.num_pages *
             self.page_size, self.num_heads, self.head_dim),
            dtype=self.dtype,
            device=self.device,
        )

    def get_kv_cache(self, layer: int, index: Optional[torch.Tensor] = None):
        if index is None:
            k_cache = self.k_cache[layer - self.start_layer]
            v_cache = self.v_cache[layer - self.start_layer]
        else:
            k_cache = self.k_cache[layer - self.start_layer, index]
            v_cache = self.v_cache[layer - self.start_layer, index]
        return k_cache, v_cache
    
    def set_kv_cache(self, layer: int, index: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor):
        self.k_cache[layer - self.start_layer, index] = k_cache
        self.v_cache[layer - self.start_layer, index] = v_cache


class KVCacheAllocator:
    def __init__(
        self,
        size: int,
        page_size: int
    ):
        assert size % page_size == 0

        self.size = size
        self.page_size = page_size

        self.free_pages = list(range(0, size, page_size))

    def alloc(self, need_size: int):
        if need_size > len(self.free_pages):
            return None

        select_index = self.free_pages[:need_size]
        self.free_pages = self.free_pages[need_size:]
        return select_index

    def free(self, free_index: abc.Iterable[int]):
        self.free_pages.extend(free_index)
        self.free_pages.sort()


class RadixTreeNode:
    def __init__(
        self,
        # child dict 的 key 等于每个子节点 key 的开头 page size 长度的子序列
        children: dict[tuple[int, ...], "RadixTreeNode"] = {},
        parent: Optional["RadixTreeNode"] = None,
        # key 和 value 长度应该为 page size 的整数倍
        key: tuple[int, ...] = tuple(),
        value: tuple[int, ...] = tuple(),
        ref_count: int = 0
    ):
        self.children = children
        self.parent = parent
        self.key = key
        self.value = value

        self.hashs: tuple[int] = tuple()

        self.access_time = time.monotonic()
        self.ref_count = ref_count

    def __lt__(self, other: "RadixTreeNode"):
        return self.access_time < other.access_time


def paged_prefix_len(key1: abc.Sequence[int], key2: abc.Sequence[int], page_size: int):
    prefix_len = 0
    while prefix_len < min(len(key1), len(key2)):
        if key1[prefix_len:prefix_len + page_size] != key2[prefix_len:prefix_len + page_size]:
            break
        prefix_len += page_size
    return prefix_len


class RadixTree:
    def __init__(
        self,
        page_size: int,
        kv_cache_allocator: KVCacheAllocator
    ):
        self.page_size = page_size
        self.root = RadixTreeNode(ref_count=1)
        self.kv_cache_allocator = kv_cache_allocator

    def match_prefix(self, key: list[int]):

        if self.page_size != 1:
            truncated_len = len(key) // self.page_size * self.page_size
            key = key[:truncated_len]
        
        node = self.root
        child_key = tuple(key[:self.page_size])
        value: list[int] = []

        while len(key) > 0 and child_key in node.children.keys():
            child = node.children[child_key]
            child.access_time = time.monotonic()

            # get max matched prefix length
            prefix_len = paged_prefix_len(child.key, key, self.page_size)
            
            if prefix_len < len(child.key):
                new_node = self._split_node(child, prefix_len)
                value.extend(new_node.value)
                node = new_node
                break
            else:
                value.extend(child.value)
                node = child
                key = key[prefix_len:]
                child_key = tuple(key[:self.page_size])

        cache_indices = value
        last_prefix_node = node
        return cache_indices, last_prefix_node
    
    def insert(self, key: list[int], value: list[int]):
        node = self.root
        child_key = tuple(key[:self.page_size])
        total_prefix_len = 0

        while len(key) > 0 and child_key in node.children.keys():
            child = node.children[child_key]
            child.access_time = time.monotonic()

            # get max matched prefix length
            prefix_len = paged_prefix_len(child.key, key, self.page_size)
            total_prefix_len += prefix_len

            key = key[prefix_len:]
            value = value[prefix_len:]
            child_key = tuple(key[:self.page_size])

            if prefix_len < len(child.key):
                new_node = self._split_node(child, prefix_len)
                node = new_node
                break
            else:
                node = child
        # last prefixed node
        last_prefix_node = node
        last_node = node

        # if we have unmatched keys, create a new node
        if len(key) > 0:
            last_node = self._add_node(
                parent=last_prefix_node,
                key=tuple(key),
                value=tuple(value)
            )
        return total_prefix_len, last_node
    
    def inc_ref(self, node: RadixTreeNode):
        while node != self.root:
            node.ref_count += 1
            node = node.parent or self.root

    def dec_ref(self, node: RadixTreeNode):
        while node != self.root:
            node.ref_count -= 1
            node = node.parent or self.root

    # TODO: need optimization, maybe we can introduce a Evictor containing a link list
    def evict(self, num_tokens: int):
        leaves = self._get_leaf_nodes()
        heapq.heapify(leaves)

        num_evicted = 0
        # len(leaves) == 0: no evictable node
        while num_evicted < num_tokens and len(leaves) > 0:
            evict_node = heapq.heappop(leaves)
            if evict_node == self.root:
                # there is no left node can be evicted
                break
            # only evict ref_count == 0 node, meaning no request is using this node
            if evict_node.ref_count > 0:
                # there is still other request using this node
                continue

            self.kv_cache_allocator.free(evict_node.value)
            num_evicted += len(evict_node.value)

            # delete evict_node
            self._remove_node(evict_node)

            if evict_node.parent and len(evict_node.parent.children) == 0:
                heapq.heappush(leaves, evict_node.parent)
    
    def get_node_prefix_len(self, node: RadixTreeNode):
        length = 0
        while node != self.root:
            length += len(node.key)
            node = node.parent or self.root
        return length

    def _get_leaf_nodes(self):
        leaf_nodes: list[RadixTreeNode] = []
        def traverse_tree(node: RadixTreeNode):
            if len(node.children) == 0:
                leaf_nodes.append(node)
                return
            for child in node.children.values():
                traverse_tree(child)
            
        traverse_tree(self.root)
        return leaf_nodes
    
    def _remove_node(self, node: RadixTreeNode):
        parent_node = node.parent
        if parent_node:
            child_key = node.key[:self.page_size]
            del parent_node.children[child_key]
    
    def _add_node(
        self,
        parent: RadixTreeNode,
        key: tuple[int, ...] = tuple(),
        value: tuple[int, ...] = tuple(),
        ref_count: int = 0
    ):
        new_node = RadixTreeNode(
            parent=parent,
            key=key,
            value=value,
            ref_count=ref_count
        )
        new_node.parent = parent
        parent.children[new_node.key[:self.page_size]] = new_node
        return new_node

    def _split_node(self, node: RadixTreeNode, split_len: int):
        # parent -> node ==> parent -> new_node -> node, return new_node
        new_node = RadixTreeNode(
            parent=node.parent,
            children={node.key[split_len:split_len + self.page_size]: node},
            key=node.key[:split_len],
            value=node.value[:split_len],
            ref_count=node.ref_count
        )
        node.parent = new_node
        node.key = node.key[split_len:]
        node.value = node.value[split_len:]

        if new_node.parent:
            new_child_key = new_node.key[:self.page_size]
            new_node.parent.children[new_child_key] = new_node

        return new_node

class KVCacheManager:
    def __init__(
        self,
        size: int,
        page_size: int = 1
    ):
        self.page_size = page_size
        self.kv_cache_allocator = KVCacheAllocator(size, page_size)
        self.radix_tree = RadixTree(page_size, self.kv_cache_allocator)
        # sequence -> (prefix_len, last_node)
        self.unfinished_sequences: dict[Sequence, tuple[int, RadixTreeNode]] = {}

    def alloc_slots(self, num_slots: int):
        indices = self.kv_cache_allocator.alloc(num_slots)
        # if no space, evict some slots from radix tree
        if indices is None:
            self.radix_tree.evict(num_slots)
            indices = self.kv_cache_allocator.alloc(num_slots)
        # if still no space, raise error
        if indices is None:
            raise RuntimeError(f"Failed to allocate {num_slots} slots, KV cache is full!")
        return indices

    def cache_sequence(self, sequence: Sequence):
        token_ids = sequence.token_ids
        kv_indices = sequence.kv_indices

        if self.page_size != 1:
            truncated_kv_len = len(kv_indices) // self.page_size * self.page_size
            kv_indices = kv_indices[:truncated_kv_len]
            token_ids = token_ids[:truncated_kv_len]
            self.kv_cache_allocator.free(kv_indices[truncated_kv_len:])

        new_prefix_len, new_last_node = self.radix_tree.insert(token_ids, kv_indices)
        old_prefix_len, old_last_node = self.unfinished_sequences.get(sequence, (0, self.radix_tree.root))

        # 相当于把 token_ids 对应的 kv_indices 取出来
        new_indices, _ = self.radix_tree.match_prefix(token_ids)
        self.unfinished_sequences[sequence] = (len(new_indices), new_last_node)

        # free the unmatched value slots
        # 这种情况发生在存在两个seq都没有插入radix tree，但是包含相同的prefix
        # seq1: [1,2,3,4], kv1: [10,11,12,13]
        # seq2: [1,2,3,5], kv2: [20,21,22,23]
        # 插入seq1后，radix tree中有[1,2,3,4] -> [10,11,12,13]
        # 插入seq2时，匹配到[1,2,3]，4和5不匹配，但是[1,2,3]对应的kv是[10,11,12]和[20,21,22]不匹配
        # 这时需要释放掉kv2中的[20,21,22]，因为这部分kv重复了，只需要保留kv1中的[10,11,12]
        # 于是最后radix tree中有[1,2,3,4] -> [10,11,12,13]和[1,2,3,5] -> [10,11,12,23]
        if new_prefix_len > old_prefix_len and old_prefix_len != 0:
            # 释放重复的kv cache slots
            self.kv_cache_allocator.free(kv_indices[old_prefix_len:new_prefix_len])
            # 更新sequence的kv_indices的重复部分
            sequence.kv_indices[old_prefix_len:new_prefix_len] = new_indices[old_prefix_len:new_prefix_len]

        self.radix_tree.dec_ref(old_last_node)
        self.radix_tree.inc_ref(new_last_node)

        if sequence.status == SequenceStatus.FINISHED:
            self.radix_tree.dec_ref(new_last_node)
            del self.unfinished_sequences[sequence]
            # kv indices 交给 radix tree 管理了，此后 seq 中的 kv_indices 可能不再有效
            sequence.kv_indices.clear()
