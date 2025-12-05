from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from PIL import Image
import torch
from transformers import AutoProcessor, AutoModelForCausalLM
import io

print("Loading final  invoice OCR model")
processor = AutoProcessor.from_pretrained("model_v1", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("model_v1", trust_remote_code=True)
model.eval()

app = FastAPI(title="Pharmacy OCR VLM API")

def ocr_image(image: Image.Image):
    inputs = processor(text="<OCR>", images=image, return_tensors="pt")

    with torch.no_grad():
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

@app.post("/ocr")
async def ocr_api(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        text = ocr_image(image)

        return JSONResponse({"filename": file.filename, "extracted_text": text})

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
