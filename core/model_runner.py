import torch
from model_loader import load_model
from models.registry import MODEL_REGISTRY
from transformers import AutoConfig, PretrainedConfig
from layers.sampler import Sampler
from core.kv_cache import KVCachePool
from core.common import ForwardBatch, ForwardMode
from core.cuda_graph import CUDAGraph
from distributed.parallel_state import get_tp_group, get_pp_group
from distributed.utils import get_pp_indices
import torch.distributed as dist
from layers.attention import attention_kv_cache, AttentionBackend
import os

_STR_DTYPE_TO_TORCH_DTYPE = {
    "half": torch.float16,
    "float16": torch.float16,
    "float": torch.float32,
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
}

def set_cuda_arch():
    capability = torch.cuda.get_device_capability()
    arch = f"{capability[0]}.{capability[1]}"
    os.environ["TORCH_CUDA_ARCH_LIST"] = f"{arch}{'+PTX' if arch == '9.0' else ''}"
    
    
def get_model_config_per_gpu(
    hf_config: PretrainedConfig,
    tp_size: int,
    tp_rank: int,
    pp_size: int,
    pp_rank: int,
):
    dtype: torch.dtype = _STR_DTYPE_TO_TORCH_DTYPE[hf_config.dtype or hf_config.torch_dtype] # type: ignore
    
    start_layer, end_layer = get_pp_indices(hf_config.num_hidden_layers, pp_rank, pp_size)
    num_layers = end_layer - start_layer
    num_heads = int(hf_config.num_attention_heads) // tp_size
    num_kv_heads = max(1, hf_config.num_key_value_heads // tp_size)
    
    if hasattr(hf_config, "head_dim"):
        head_dim = int(hf_config.head_dim)
    else:
        head_dim = int(hf_config.hidden_size // hf_config.num_attention_heads)
    
    return (
        dtype,
        start_layer,
        end_layer,
        num_layers,
        num_heads,
        num_kv_heads,
        head_dim
    )

class ModelRunner:
    def __init__(
        self,
        model: str,
        max_bs: int,
        rank: int,
        device: torch.device,
        enforce_eager: bool = False,
        context_len: int = 2048,
    ):
        self.model_path = model
        self.max_bs = max_bs
        self.rank = rank
        self.device = device
        self.enforce_eager = enforce_eager
        self.context_len = context_len
        set_cuda_arch()
    
    def load_model(self):
        
        hf_config = AutoConfig.from_pretrained(
            self.model_path, trust_remote_code=True
        )

        architectures = getattr(hf_config, "architectures", [])
        ModelClass = None
        ConfigClass = None
        for arch in architectures:
            if arch in MODEL_REGISTRY:
                ModelClass, ConfigClass = MODEL_REGISTRY[arch]
                break
        assert ModelClass is not None and ConfigClass is not None, \
            f"Model arch {hf_config.architectures} not supported."
            
        self.hf_config = ConfigClass()
        self.hf_config.update(hf_config.to_dict())
        
        print(f"Rank {self.rank} loading model {self.model_path} with type {ModelClass.__name__}.")
        
        torch_default_dtype = torch.get_default_dtype()
        
        
        (
            self.dtype,
            self.start_layer,
            self.end_layer,
            self.num_layers,
            self.num_heads,
            self.num_kv_heads,
            self.head_dim
        ) = get_model_config_per_gpu(
            self.hf_config,
            get_tp_group().size(),
            get_tp_group().rank(),
            get_pp_group().size(),
            get_pp_group().rank(),
        )
        
        
        torch.set_default_dtype(self.dtype)

        self.model = ModelClass(self.hf_config)
        self.model.to(self.device)

        self.sampler = Sampler()
        
        load_model(self.model, self.model_path)
        
        torch.set_default_dtype(torch_default_dtype)

    def initialize_kv_cache(self, kv_cache_size: int):
        self.kv_cache_size = kv_cache_size
        self.kv_cache = KVCachePool(
            dtype=self.dtype,
            device=self.device,
            num_tokens=self.kv_cache_size,
            num_layers=self.num_layers,
            num_heads=self.num_kv_heads,
            head_dim=self.head_dim,
            start_layer=self.start_layer,
            end_layer=self.end_layer,
        )
        self.attn_backend = AttentionBackend(
            kv_cache=self.kv_cache,
            num_qo_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim,
            dtype=self.dtype,
            device=self.device,
        )
        if not self.enforce_eager:
            self.cuda_graph = CUDAGraph()
            self.capture_graph()
        else:
            self.cuda_graph = None

    def profile_kv_cache_size(self, gpu_memory_utilization: float = 0.9):
        cache_memsize_per_token = self.num_layers * self.num_kv_heads * self.head_dim * 2 * self.dtype.itemsize
        
        torch.cuda.empty_cache()
        free_gpu_memory, total_gpu_memory = torch.cuda.mem_get_info(self.device)
        
        max_num_tokens = int(free_gpu_memory * gpu_memory_utilization) // cache_memsize_per_token
        max_num_tokens = torch.tensor(max_num_tokens, device=self.device)
        dist.all_reduce(max_num_tokens, op= dist.ReduceOp.MIN)
        max_num_tokens = int(max_num_tokens.item())

        torch.cuda.empty_cache()
        return max_num_tokens

    def prepare_input(self, batch: ForwardBatch):
        input_ids: list[int] = []
        positions: list[int] = []

        for seq in batch.seqs:
            input_ids.extend(seq.token_ids[seq.cached_kv_len:])
            positions.extend(range(seq.cached_kv_len, len(seq.token_ids)))

        tensor_input_ids = torch.tensor(input_ids, dtype=torch.long, device=self.device)
        tensor_positions = torch.tensor(positions, dtype=torch.long, device=self.device)

        return (
            tensor_input_ids,
            tensor_positions,
        )
    
    def prepare_sampling_params(self, batch: ForwardBatch):
        vocab_size = self.hf_config.vocab_size
        
        temperatures = []
        min_ps = []
        top_ps = []
        top_ks = []
        for seq in batch.seqs:
            temperatures.append(seq.sampling_params.temperature)
            min_ps.append(seq.sampling_params.min_p)
            top_ps.append(seq.sampling_params.top_p)
            top_k = seq.sampling_params.top_k
            if top_k == -1:
                top_k = vocab_size
            else:
                top_k = min(top_k, vocab_size)
            top_ks.append(top_k)
        
        tensor_temperatures = torch.tensor(temperatures, dtype=torch.float, device=self.device)
        tensor_min_ps = torch.tensor(min_ps, dtype=torch.float, device=self.device)
        tensor_top_ps = torch.tensor(top_ps, dtype=torch.float, device=self.device)
        tensor_top_ks = torch.tensor(top_ks, dtype=torch.long, device=self.device)

        return (
            tensor_temperatures,
            tensor_min_ps,
            tensor_top_ps,
            tensor_top_ks
        )
    
    def prepare_last_hidden_states(self, batch: ForwardBatch, hidden_states: torch.Tensor):
        last_indices = []
        cu_seq_len = 0
        for seq in batch.seqs:
            cu_seq_len += len(seq.token_ids) - seq.cached_kv_len
            last_indices.append(cu_seq_len - 1)
        return hidden_states[..., last_indices, :]

    @torch.inference_mode()
    def execute_model(self, batch: ForwardBatch) -> list[int]:
        assert hasattr(self, 'model') and hasattr(self, 'sampler'), \
            "Model and sampler must be loaded before execution."
        assert hasattr(self, "kv_cache"), "KV Cache not initialized yet."

        if self.cuda_graph is not None and self.cuda_graph.is_captured and batch.forward_mode == ForwardMode.DECODE:
            logits = self._execute_model_cuda_graph(batch)
        else:
            logits = self._execute_model_eager(batch)

        (
            temperatures,
            min_ps,
            top_ps,
            top_ks
        ) = self.prepare_sampling_params(batch)
        output_ids = self.sampler(logits, temperatures, min_ps, top_ps, top_ks)
        return output_ids.tolist()
    
    
    def _execute_model_cuda_graph(self, batch: ForwardBatch) -> torch.Tensor:
        assert self.cuda_graph is not None
        input_ids, positions = self.prepare_input(batch)
        
        input_idx_buffer = self.cuda_graph.get_input_buffer("input_ids")
        positions_buffer = self.cuda_graph.get_input_buffer("positions")
        
        input_idx_buffer[:len(input_ids)] = input_ids
        positions_buffer[:len(positions)] = positions
        
        bs = batch.num_seqs
        padded_bs = self.cuda_graph.match_bs(bs)
        
        attention_metadata = self.attn_backend.build_metadata_for_cuda_graph_replay(
            graph=self.cuda_graph,
            batch=batch,
            padded_bs=padded_bs,
        )
        
        with attention_kv_cache(self.model, attention_metadata):
            self.cuda_graph.replay(bs=padded_bs)
            
        hidden_states_buffer = self.cuda_graph.get_output_buffer("hidden_states")
        hidden_states = hidden_states_buffer[:bs]
    
        hidden_states = self.prepare_last_hidden_states(batch, hidden_states)
        logits = self.model.compute_logits(hidden_states)
        return logits

    def _execute_model_eager(self, batch: ForwardBatch) -> torch.Tensor:
        input_ids, positions = self.prepare_input(batch)
        
        attention_metadata = self.attn_backend.build_metadata(batch=batch)

        with attention_kv_cache(self.model, attention_metadata):
            hidden_states = self.model(input_ids, positions)

        hidden_states = self.prepare_last_hidden_states(batch, hidden_states)
        logits = self.model.compute_logits(hidden_states)
        return logits

    @torch.inference_mode()
    def capture_graph(self):
        assert self.cuda_graph is not None
        graph_bs = [1, 2, 4, 8] + list(range(16, self.max_bs, 16))
        if self.max_bs not in graph_bs:
            graph_bs.append(self.max_bs)

        print("Capturing CUDA graphs for batch sizes:", graph_bs)

        input_ids = torch.zeros(
            (self.max_bs,),
            dtype=torch.long,
            device=self.device
        )
        positions = torch.zeros(
            (self.max_bs,),
            dtype=torch.long,
            device=self.device
        )
        hidden_states = torch.zeros(
            (self.max_bs, self.hf_config.hidden_size),
            dtype=self.dtype,
            device=self.device
        )
        self.cuda_graph.set_input_buffer("input_ids", input_ids)
        self.cuda_graph.set_input_buffer("positions", positions)
        self.cuda_graph.set_output_buffer("hidden_states", hidden_states)
        
        self.attn_backend.prepare_for_cuda_graph_capture(
            graph=self.cuda_graph,
            max_bs=self.max_bs,
            context_len=self.context_len,
        )
        
        for bs in reversed(graph_bs):
            attention_metadata = self.attn_backend.build_metadata_for_cuda_graph_capture(
                graph=self.cuda_graph,
                bs=bs,
                context_len=self.context_len,
            )

            with attention_kv_cache(self.model, attention_metadata):
                # Warmup before capture
                hidden_states[:bs] = self.model(input_ids[:bs], positions[:bs])
                with self.cuda_graph.capture(bs):
                    hidden_states[:bs] = self.model(input_ids[:bs], positions[:bs])
            
            torch.cuda.synchronize()
