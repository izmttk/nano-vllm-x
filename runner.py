from models.qwen3 import Qwen3ForCausalLM
from layers.sampler import Sampler
from transformers import AutoConfig, AutoTokenizer
from model_loader import load_model
import torch

from distributed.parallel_state import is_first_rank
def run_model():
    model_path = "/opt/models/Qwen3-8B"
    config = AutoConfig.from_pretrained(
        model_path, trust_remote_code=True
    )
    # if is_first_rank():
    #     print(config)
    
    model = Qwen3ForCausalLM(config).to("cuda")
    sampler = Sampler()

    load_model(model, model_path)

    prompt = "明月几时有，把酒问青天。不知"

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to("cuda")
    positions = torch.arange(input_ids.shape[1], dtype=torch.long).to("cuda")

    if is_first_rank():
        print(input_ids, positions)

    hidden_states = model(input_ids, positions)
    logits = model.compute_logits(hidden_states)

    if is_first_rank():
        print(logits.shape)
        output_ids = sampler(logits[:, -1, :])
        print(output_ids)
        print(tokenizer.decode(output_ids[0]))