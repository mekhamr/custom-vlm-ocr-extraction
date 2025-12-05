from transformers import Trainer, TrainingArguments, EarlyStoppingCallback
from datasets import load_dataset
from config import model, processor
from PIL import Image
import torch
import wandb

# Load data
train_dataset = load_dataset("json", data_files="dataset/train_metadata.jsonl", split="train")
eval_dataset  = load_dataset("json", data_files="dataset/test_metadata.jsonl", split="train")

print(f"Training samples  : {len(train_dataset)}")
print(f"Validation samples: {len(eval_dataset)}")


def collate_fn(batch):
    images = []
    prompts = []
    full_texts = []

    for item in batch:
        file_name = item["file_name"]

        
        if file_name in ["10.png", "11.png"]:
            img_path = f"test_images/{file_name}"
        else:
            img_path = f"dataset/images/{file_name}"

        image = Image.open(img_path).convert("RGB")
        images.append(image)
        prompts.append("<OCR>")
        full_texts.append(f"<OCR>{item['text']}</OCR>")

    inputs = processor(text=prompts, images=images, return_tensors="pt", padding=True, truncation=True)
    labels = processor.tokenizer(full_texts, padding=True, truncation=True, return_tensors="pt").input_ids
    labels[labels == processor.tokenizer.pad_token_id] = -100
    inputs["labels"] = labels
    return inputs


training_args = TrainingArguments(
    output_dir="./output",
    num_train_epochs=15,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=3e-4,
    weight_decay=0.01,
    warmup_steps=8,
    logging_steps=2,
    eval_strategy="steps",
    eval_steps=6,
    save_steps=12,
    save_total_limit=3,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    load_best_model_at_end=True,
    report_to="wandb",
    run_name="florence2-pharmacy-ocr-v1",
    remove_unused_columns=False,
    dataloader_num_workers=0,
    fp16=False,
    bf16=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=collate_fn,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
)

print("Starting training")
trainer.train()

trainer.save_model("./output")
artifact = wandb.Artifact("florence2-pharmacy-ocr-v1", type="model")
artifact.add_dir("./output")
wandb.log_artifact(artifact)
wandb.finish()
print("Model trained")





