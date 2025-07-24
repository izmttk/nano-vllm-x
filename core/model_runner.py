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