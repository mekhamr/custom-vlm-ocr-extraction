
import argparse
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM
import torch

# Load merged model
print("Loading  final pharmacy OCR model...")
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