import torch
from model_loader import load_model
from models.registry import MODEL_REGISTRY
from transformers import AutoConfig, PretrainedConfig
from layers.sampler import Sampler
from core.kv_cache import KVCachePool
from core.common import ForwardBatch
from distributed.parallel_state import get_tp_group, get_pp_group
from distributed.utils import get_pp_indices
from layers.attention import attention_kv_cache, AttentionMetadata
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
        rank: int,
        device: torch.device,

        # KV cache parameters
        kv_cache_size: int,
    ):
        self.model_path = model
        self.rank = rank
        self.device = device

        self.kv_cache_size = kv_cache_size
        
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
    
    def prepare_last_hiden_states(self, batch: ForwardBatch, hidden_states: torch.Tensor):
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

        input_ids, positions = self.prepare_input(batch)
        
        attention_metadata = AttentionMetadata.build(
            batch=batch,
            kv_cache=self.kv_cache,
            num_qo_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim,
            dtype=self.dtype,
            device=self.device,
        )

        with attention_kv_cache(self.model, attention_metadata):
            hidden_states = self.model(input_ids, positions)

        hidden_states = self.prepare_last_hiden_states(batch, hidden_states)
        logits = self.model.compute_logits(hidden_states)
        
        (
            temperatures,
            min_ps,
            top_ps,
            top_ks
        ) = self.prepare_sampling_params(batch)
        tensor_output_ids = self.sampler(logits, temperatures, min_ps, top_ps, top_ks)

        return tensor_output_ids.tolist()