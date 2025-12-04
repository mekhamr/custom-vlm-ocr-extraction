
import argparse
import json
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM
import torch
import os

# Load merged model
print("Loading final pharmacy OCR model...")
processor = AutoProcessor.from_pretrained("model_v1", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("model_v1", trust_remote_code=True)

def ocr_image(image_path):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(text="<OCR>", images=image, return_tensors="pt")

    generated_ids = model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=1024,
        num_beams=3,
        do_sample=False
    )

    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    ocr_result = generated_text.split("<OCR>")[-1].split("</OCR>")[0].strip()
    return ocr_result

def save_to_jsonl(image_path, text):
    output_file = "output.jsonl"   # renamed file

    record = {
        "filename": os.path.basename(image_path),
        "extracted_text": text
    }

    # Append as one-line JSON
    with open(output_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"\nSaved to: {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True, help="Path to invoice image")
    args = parser.parse_args()

    print(f"Running OCR on: {args.image}")
    result = ocr_image(args.image)

    print("\n" + "="*60)
    print("EXTRACTED TEXT:")
    print("="*60)
    print(result)
    print("="*60)

    # Save result to output.jsonl
    save_to_jsonl(args.image, result)
