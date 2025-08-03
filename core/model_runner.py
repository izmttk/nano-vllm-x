import torch
from model_loader import load_model
from models.registry import MODEL_REGISTRY
from transformers import AutoConfig
from layers.sampler import Sampler

class ModelRunner:
    def __init__(
        self,
        model: str,
        rank: int,
        device: torch.device,
    ):
        self.model_path = model
        self.rank = rank
        self.device = device
    
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
        
        self.model = ModelClass(self.hf_config)
        self.model.to(self.device)

        self.sampler = Sampler()
        
        load_model(self.model, self.model_path)

    def execute_model(self, input_ids: torch.Tensor):
        input_device = input_ids.device
        # input_ids: (seq_len)
        input_ids = input_ids.to(self.device)
        assert hasattr(self, 'model') and hasattr(self, 'sampler'), \
            "Model and sampler must be loaded before execution."
        print(f"Rank {self.rank} executing model {self.model_path}.")

        positions = torch.arange(input_ids.shape[-1], dtype=torch.long, device=self.device)

        hidden_states = self.model(input_ids, positions)
        logits = self.model.compute_logits(hidden_states)

        output_ids = self.sampler(logits[..., -1, :])
        return output_ids.to(input_device)
