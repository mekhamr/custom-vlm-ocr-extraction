# config.py
from transformers import AutoProcessor, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import torch

model_id = "microsoft/Florence-2-base"

print("Loading processor")
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

print("Loading Florence-2-base")
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    trust_remote_code=True,
    torch_dtype=torch.float32,       
    device_map="cpu",
    low_cpu_mem_usage=True
)
model = prepare_model_for_kbit_training(model)


lora_config = LoraConfig(
    r=32,                   
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

print("Model + LoRA loaded ")