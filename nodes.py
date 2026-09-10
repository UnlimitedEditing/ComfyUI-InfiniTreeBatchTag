"""
Florence2BatchTagFromURLs -- batch image captioning/tagging in a single
Graydient job call.

Graydient gives every job a fresh, ephemeral container with no state carried
between runs and a ~300s public timeout. That rules out the usual "start a
server, keep the model warm, call it N times" pattern used for local batch
work -- there is no "N times" across jobs here, only within one job. So this
node takes a *list* of image URLs and does the whole loop itself: load the
model once, caption+tag each image in turn, and return whatever finished
before the time budget runs out (rather than timing out mid-batch and losing
everything).

Output is delivered via the same "pixel data image" convention used by this
project's other LLM/VLM Graydient workflows (gen_llm.py, gen_vlm.py): a
4-byte big-endian length header followed by UTF-8 JSON bytes, packed 3 bytes
per pixel (RGB channels only, alpha skipped), decoded by ForgeExpress with:
    const bytes = [...imgData].reduce((a,_,i)=>i%4<3?[...a,imgData[i]]:a,[])
    const n = bytes[0]<<24|bytes[1]<<16|bytes[2]<<8|bytes[3]
    const text = new TextDecoder().decode(new Uint8Array(bytes.slice(4,4+n)))
"""
import io
import json
import math
import os
import time

import numpy as np
import requests
import torch
from PIL import Image

DEFAULT_MODEL = "MiaoshouAI/Florence-2-base-PromptGen-v2.0"
CAPTION_TASK = "<MORE_DETAILED_CAPTION>"
TAGS_TASK = "<GENERATE_TAGS>"

_MODEL_CACHE = {}


def _local_model_dir(model_id: str) -> str | None:
    """Prefer a Graydient concept_mapping pre-staged copy under models/LLM/<name>."""
    try:
        import folder_paths
        base = folder_paths.models_dir
    except ImportError:
        return None
    name = model_id.rsplit("/", 1)[-1]
    candidate = os.path.join(base, "LLM", name)
    if os.path.isfile(os.path.join(candidate, "model.safetensors")):
        return candidate
    return None


def _load_model(model_id: str):
    if model_id in _MODEL_CACHE:
        return _MODEL_CACHE[model_id]

    from transformers import AutoModelForCausalLM, AutoProcessor

    source = _local_model_dir(model_id) or model_id
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        source, trust_remote_code=True, torch_dtype=dtype
    ).to(device).eval()
    processor = AutoProcessor.from_pretrained(source, trust_remote_code=True)

    _MODEL_CACHE[model_id] = (model, processor, device, dtype)
    return _MODEL_CACHE[model_id]


def _run_task(model, processor, device, dtype, image: Image.Image, task: str, max_new_tokens: int) -> str:
    inputs = processor(text=task, images=image, return_tensors="pt").to(device, dtype)
    with torch.no_grad():
        generated_ids = model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=max_new_tokens,
            num_beams=1,
            do_sample=False,
        )
    text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    parsed = processor.post_process_generation(text, task=task, image_size=(image.width, image.height))
    return str(parsed.get(task, "")).strip()


def _encode_data_image(payload: dict) -> np.ndarray:
    body = json.dumps(payload).encode("utf-8")
    header = len(body).to_bytes(4, "big")
    raw = header + body

    pixel_count = math.ceil(len(raw) / 3)
    raw = raw + b"\x00" * (pixel_count * 3 - len(raw))
    rgb = np.frombuffer(raw, dtype=np.uint8).reshape(pixel_count, 3)

    side = max(1, math.ceil(math.sqrt(pixel_count)))
    padded = np.zeros((side * side, 3), dtype=np.uint8)
    padded[:pixel_count] = rgb
    img = padded.reshape(side, side, 3)

    tensor = torch.from_numpy(img).float() / 255.0
    return tensor.unsqueeze(0)  # (1, H, W, 3) -- ComfyUI IMAGE convention


class Florence2BatchTagFromURLs:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_urls": ("STRING", {"multiline": True, "default": ""}),
                "model_id": ("STRING", {"default": DEFAULT_MODEL}),
                "max_seconds": ("INT", {"default": 240, "min": 10, "max": 3600}),
                "caption_max_tokens": ("INT", {"default": 200, "min": 8, "max": 1024}),
                "tags_max_tokens": ("INT", {"default": 128, "min": 8, "max": 512}),
            }
        }

    RETURN_TYPES = ("STRING", "IMAGE")
    RETURN_NAMES = ("response_text", "response_data")
    FUNCTION = "run"
    CATEGORY = "InfiniTree"

    def run(self, image_urls, model_id, max_seconds, caption_max_tokens, tags_max_tokens):
        deadline = time.monotonic() + max_seconds
        urls = [u.strip() for u in image_urls.splitlines() if u.strip()]

        model, processor, device, dtype = _load_model(model_id)

        results = []
        for url in urls:
            if time.monotonic() >= deadline:
                break
            try:
                resp = requests.get(url, timeout=20)
                resp.raise_for_status()
                image = Image.open(io.BytesIO(resp.content)).convert("RGB")

                description = _run_task(model, processor, device, dtype, image, CAPTION_TASK, caption_max_tokens)
                raw_tags = _run_task(model, processor, device, dtype, image, TAGS_TASK, tags_max_tokens)
                tags = [t.strip().lower() for t in raw_tags.split(",") if t.strip()]

                results.append({"url": url, "description": description, "tags": tags})
            except Exception as e:
                results.append({"url": url, "error": str(e)})

        payload = {
            "processed": len(results),
            "requested": len(urls),
            "results": results,
        }
        response_text = json.dumps(payload)
        response_data = _encode_data_image(payload)
        return (response_text, response_data)


NODE_CLASS_MAPPINGS = {"Florence2BatchTagFromURLs": Florence2BatchTagFromURLs}
NODE_DISPLAY_NAME_MAPPINGS = {"Florence2BatchTagFromURLs": "Florence2 Batch Tag From URLs"}
