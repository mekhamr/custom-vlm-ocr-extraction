# merge_lora.py
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoProcessor
import torch

print("Loading base Florence-2 model")
base_model = AutoModelForCausalLM.from_pretrained(
    "microsoft/Florence-2-base",
    trust_remote_code=True,
    torch_dtype=torch.float32
)

print("Loading best LoRA adapter (checkpoint-60)")
model = PeftModel.from_pretrained(base_model, "./output/checkpoint-60")

print("Merging LoRA weights into base model")
merged_model = model.merge_and_unload()

print("Saving final merged model to model_v1/ ")
merged_model.save_pretrained("model_v1")
AutoProcessor.from_pretrained("microsoft/Florence-2-base", trust_remote_code=True).save_pretrained("model_v1")

print(" final standalone model is in: model_v1/")
