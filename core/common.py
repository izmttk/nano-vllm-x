from typing import Optional
from dataclasses import dataclass, field
import enum
import torch

@dataclass
class SamplingParams:
    n: int = 1
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1
    min_p: float = 0.0

    max_new_tokens: int = 128
    ignore_eos: bool = False

class SequenceStatus(enum.Enum):
    WAITING = enum.auto()
    RUNNING = enum.auto()
    FINISHED = enum.auto()

@dataclass
class Sequence:
    seq_id: int
    status: SequenceStatus = SequenceStatus.WAITING
    num_tokens: int = 0
    tokens: list[str] = field(default_factory=list)
    token_ids: list[int] = field(default_factory=list)
    sampling_params: SamplingParams = field(default_factory=SamplingParams)
    kv_indices: list[int] = field(default_factory=list)
    cached_kv_len: int = 0
    stream:  bool = False

class ForwardMode(enum.Enum):
    PREFILL = 0
    DECODE = 1

@dataclass
class ForwardBatch:
    foward_mode: ForwardMode
    num_seqs: int
    seqs: list[Sequence]
    max_bs: int
