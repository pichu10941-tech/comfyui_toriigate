"""
ToriiGate – llama.cpp API nodes
================================
These nodes communicate with a llama-server (llama.cpp) that exposes an
OpenAI-compatible HTTP API.  They do NOT import transformers, torch model
loaders, AutoProcessor, AutoModel, or any CUDA initialisation code.

Typical server launch (example):
    llama-server.exe ^
        --model ToriiGate-0.5-Q4_K_M.gguf ^
        --mmproj mmproj-ToriiGate-0.5-f16.gguf ^
        --port 8000

The nodes then send HTTP requests to http://127.0.0.1:8000/v1/chat/completions.
"""

import base64
import io
import json
import logging
import random

import numpy as np
from PIL import Image

from .nodes import CAPTION_TYPES, _empty_grounding
from .prompts import make_user_query, prompts_b, prompts_names_only, system_prompt

logger = logging.getLogger("ToriiGate.API")


# ---------------------------------------------------------------------------
# Shared helper functions
# ---------------------------------------------------------------------------


def image_tensor_to_base64(image_tensor, max_pixels_mp: float = 1.0) -> str:
    """Convert a ComfyUI IMAGE tensor (B, H, W, C float32 in [0,1]) to a
    base64-encoded PNG string suitable for embedding in a data-URI.
    Downscales the image if it exceeds max_pixels_mp to prevent massive TTFT."""
    img_np = (image_tensor[0].detach().cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    img_pil = Image.fromarray(img_np)
    if img_pil.mode != "RGB":
        img_pil = img_pil.convert("RGB")

    current_pixels = img_pil.width * img_pil.height
    max_pixels_count = max_pixels_mp * 1_000_000
    if current_pixels > max_pixels_count:
        scale = (max_pixels_count / current_pixels) ** 0.5
        new_w = max(1, int(img_pil.width * scale))
        new_h = max(1, int(img_pil.height * scale))
        logger.info(f"[ToriiGate API] Downscaling image from {img_pil.width}x{img_pil.height} to {new_w}x{new_h}")
        img_pil = img_pil.resize((new_w, new_h), Image.Resampling.LANCZOS)

    buf = io.BytesIO()
    img_pil.save(buf, format="PNG")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def build_vision_payload(
    model_name: str,
    system_prompt: str,
    user_text: str,
    image_b64: str,
    temperature: float,
    max_tokens: int,
    seed: int = 0,
) -> dict:
    """Build an OpenAI-compatible chat-completions payload with an image."""
    messages = []
    if system_prompt.strip():
        messages.append({"role": "system", "content": system_prompt.strip()})

    messages.append(
        {
            "role": "user",
            "content": [
                {"type": "text", "text": user_text},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{image_b64}"},
                },
            ],
        }
    )

    payload = {
        "model": model_name,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if seed != 0:
        payload["seed"] = seed
    return payload


def build_text_payload(
    model_name: str,
    system_prompt: str,
    user_text: str,
    temperature: float,
    max_tokens: int,
) -> dict:
    """Build an OpenAI-compatible chat-completions payload without an image."""
    messages = []
    if system_prompt.strip():
        messages.append({"role": "system", "content": system_prompt.strip()})

    messages.append({"role": "user", "content": user_text})

    payload = {
        "model": model_name,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    return payload


def send_chat_request(server_url: str, payload: dict, timeout: float) -> str:
    """POST *payload* to ``{server_url}/v1/chat/completions`` and return the
    generated text.  Raises a ``RuntimeError`` with a human-readable message
    on any failure so that ComfyUI can surface the error in the UI."""
    try:
        import requests  # deferred so missing requests gives a clear message
    except ImportError as exc:
        raise RuntimeError(
            "[ToriiGate API] The 'requests' library is not installed. "
            "Run: pip install requests"
        ) from exc

    endpoint = server_url.rstrip("/") + "/v1/chat/completions"
    logger.info("[ToriiGate API] POST → %s  (model=%s)", endpoint, payload.get("model", "?"))

    try:
        response = requests.post(
            endpoint,
            json=payload,
            timeout=timeout,
            headers={"Content-Type": "application/json"},
        )
    except requests.exceptions.ConnectionError as exc:
        raise RuntimeError(
            f"[ToriiGate API] Cannot connect to llama-server at '{server_url}'. "
            "Make sure llama-server is running and the URL is correct.\n"
            f"Detail: {exc}"
        ) from exc
    except requests.exceptions.Timeout as exc:
        raise RuntimeError(
            f"[ToriiGate API] Request timed out after {timeout}s. "
            "Try increasing the timeout or reducing max_tokens.\n"
            f"Detail: {exc}"
        ) from exc
    except requests.exceptions.RequestException as exc:
        raise RuntimeError(f"[ToriiGate API] HTTP error: {exc}") from exc

    if not response.ok:
        try:
            error_body = response.json()
        except Exception:
            error_body = response.text
        raise RuntimeError(
            f"[ToriiGate API] Server returned HTTP {response.status_code}.\n"
            f"Response: {json.dumps(error_body, ensure_ascii=False, indent=2)}"
        )

    try:
        data = response.json()
    except ValueError as exc:
        raise RuntimeError(
            f"[ToriiGate API] Server returned non-JSON response:\n{response.text[:500]}"
        ) from exc

    try:
        text = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(
            f"[ToriiGate API] Unexpected response format:\n"
            f"{json.dumps(data, ensure_ascii=False, indent=2)[:500]}"
        ) from exc

    logger.info("[ToriiGate API] Received %d characters of generated text.", len(text))
    return text


# ---------------------------------------------------------------------------
# All GGUF quantization variants available at:
# https://huggingface.co/DraconicDragon/ToriiGate-0.5-GGUF
# Format used by llama-server router: "repo:quant_tag"
# ---------------------------------------------------------------------------

GGUF_MODEL_NAMES = [
    "DraconicDragon/ToriiGate-0.5-GGUF:Q4_K_M",   # 3.07 GB  ← recommended balance
    "DraconicDragon/ToriiGate-0.5-GGUF:Q4_K_S",   # 2.92 GB
    "DraconicDragon/ToriiGate-0.5-GGUF:Q4_0",     # 2.90 GB
    "DraconicDragon/ToriiGate-0.5-GGUF:Q4_1",     # 3.16 GB
    "DraconicDragon/ToriiGate-0.5-GGUF:IQ4_NL",   # 2.98 GB
    "DraconicDragon/ToriiGate-0.5-GGUF:Q5_K_M",   # 3.51 GB
    "DraconicDragon/ToriiGate-0.5-GGUF:Q5_K_S",   # 3.43 GB
    "DraconicDragon/ToriiGate-0.5-GGUF:Q6_K",     # 3.99 GB
    "DraconicDragon/ToriiGate-0.5-GGUF:Q8_0",     # 5.16 GB
    "DraconicDragon/ToriiGate-0.5-GGUF:Q3_K_L",   # 2.69 GB
    "DraconicDragon/ToriiGate-0.5-GGUF:Q3_K_M",   # 2.54 GB
    "DraconicDragon/ToriiGate-0.5-GGUF:Q3_K_S",   # 2.34 GB
    "DraconicDragon/ToriiGate-0.5-GGUF:Q2_K",     # 2.12 GB
    "DraconicDragon/ToriiGate-0.5-GGUF:bf16",     # 9.70 GB
]



class LlamaCppVisionGenerate:
    """Sends an image + prompt to a llama-server vision endpoint and returns
    the generated caption text.

    The image is converted from a ComfyUI tensor to a base64-encoded PNG and
    embedded in an OpenAI-compatible multimodal chat-completions request.
    No Transformers or PyTorch model loading is performed.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": (
                    "IMAGE",
                    {
                        "tooltip": (
                            "ComfyUI image tensor (B, H, W, C float32). "
                            "Only the first image in the batch is sent."
                        )
                    },
                ),
                "server_url": (
                    "STRING",
                    {
                        "default": "http://127.0.0.1:8080",
                        "tooltip": (
                            "Base URL of the llama-server instance. "
                            "Example: http://127.0.0.1:8080"
                        ),
                    },
                ),
                "model_name": (
                    GGUF_MODEL_NAMES,
                    {
                        "default": "DraconicDragon/ToriiGate-0.5-GGUF:Q4_K_M",
                        "tooltip": (
                            "GGUF quantization to use. The identifier must match what the "
                            "llama-server router registered (shown at startup as 'operator(): ...'). "
                            "Q4_K_M is the recommended balance of quality vs size (3.07 GB). "
                            "Use custom_model_name below to override with any arbitrary string."
                        ),
                    },
                ),
                "timeout": (
                    "FLOAT",
                    {
                        "default": 120.0,
                        "min": 5.0,
                        "max": 600.0,
                        "step": 5.0,
                        "tooltip": (
                            "HTTP request timeout in seconds. "
                            "Increase for slow hardware or very long generations."
                        ),
                    },
                ),
            },
            "optional": {
                "custom_model_name": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": (
                            "Override the model identifier with any custom string. "
                            "Useful when running a non-GGUF backend or a locally "
                            "renamed model. Leave blank to use the dropdown above."
                        ),
                    },
                ),
                "prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                        "tooltip": "Optional prompt. You can connect the text output from the ToriiGate Grounding Builder here, or type your own.",
                        "forceInput": True,
                    },
                ),
                "max_pixels_mp": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.1,
                        "max": 8.0,
                        "step": 0.1,
                        "tooltip": "Resolution limit sent to the model, in megapixels. Lower values drastically reduce prompt evaluation time (Time To First Token) in llama.cpp.",
                    },
                ),
                "max_new_tokens": (
                    "INT",
                    {
                        "default": 512,
                        "min": 64,
                        "max": 8192,
                        "tooltip": "Maximum generated tokens.",
                    },
                ),
                "temperature": (
                    "FLOAT",
                    {
                        "default": 0.5,
                        "min": 0.0,
                        "max": 2.0,
                        "step": 0.01,
                        "tooltip": "Generation randomness. 0 is deterministic.",
                    },
                ),
                "decoding": (
                    ["sample", "greedy_fast"],
                    {
                        "default": "sample",
                        "tooltip": "sample uses temperature-based sampling; greedy_fast sets temperature to 0.0.",
                    },
                ),
                "seed": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 0xFFFFFFFFFFFFFFFF,
                        "tooltip": "Seed for reproducibility. Use 0 for a random seed.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("caption",)
    FUNCTION = "generate"
    CATEGORY = "ToriiGate/API"

    def generate(
        self,
        image,
        server_url="http://127.0.0.1:8080",
        model_name="DraconicDragon/ToriiGate-0.5-GGUF:Q4_K_M",
        timeout=120.0,
        custom_model_name="",
        prompt="",
        max_pixels_mp=1.0,
        max_new_tokens=512,
        temperature=0.5,
        decoding="sample",
        seed=0,
    ):
        resolved_model = custom_model_name.strip() if custom_model_name.strip() else model_name.strip()
        actual_server_url = server_url.strip().rstrip("/")
        
        actual_temperature = 0.0 if decoding == "greedy_fast" else float(temperature)

        print(
            f"[ToriiGate] Caption generation started (Vision API) "
            f"(server={actual_server_url}, model={resolved_model}, max_pixels={max_pixels_mp}MP, "
            f"decoding={decoding}, temperature={actual_temperature:.2f})."
        )

        if not prompt:
            prompt = "Describe this image in detail."

        # Convert image tensor → base64 PNG
        image_b64 = image_tensor_to_base64(image, float(max_pixels_mp))

        payload = build_vision_payload(
            model_name=resolved_model,
            system_prompt="",
            user_text=prompt,
            image_b64=image_b64,
            temperature=actual_temperature,
            max_tokens=int(max_new_tokens),
            seed=int(seed) if seed != 0 else None,
        )

        import time
        start_time = time.perf_counter()
        result = send_chat_request(server_url=actual_server_url, payload=payload, timeout=float(timeout))
        elapsed = time.perf_counter() - start_time
        
        print(f"[ToriiGate] Caption generation finished in {elapsed:.1f}s ({len(result)} chars).")
        return (result.strip(),)


# ---------------------------------------------------------------------------
# Node: LlamaCppTextGenerate
# ---------------------------------------------------------------------------

_DEFAULT_TEXT_PROMPT = "Describe the following topic in detail:"


class LlamaCppTextGenerate:
    """Sends a text-only prompt to a llama-server and returns the generated
    response.  No image is sent; no Transformers or PyTorch code is used.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": _DEFAULT_TEXT_PROMPT,
                        "tooltip": "User prompt sent to the llama-server.",
                    },
                ),
                "server_url": (
                    "STRING",
                    {
                        "default": "http://127.0.0.1:8080",
                        "tooltip": (
                            "Base URL of the llama-server instance. "
                            "Example: http://127.0.0.1:8080"
                        ),
                    },
                ),
                "model_name": (
                    GGUF_MODEL_NAMES,
                    {
                        "default": "DraconicDragon/ToriiGate-0.5-GGUF:Q4_K_M",
                        "tooltip": (
                            "GGUF quantization to use. The identifier must match what the "
                            "llama-server router registered (shown at startup as 'operator(): ...'). "
                            "Q4_K_M is the recommended balance of quality vs size (3.07 GB). "
                            "Use custom_model_name below to override with any arbitrary string."
                        ),
                    },
                ),
                "temperature": (
                    "FLOAT",
                    {
                        "default": 0.7,
                        "min": 0.0,
                        "max": 2.0,
                        "step": 0.01,
                        "tooltip": (
                            "Sampling temperature. 0 is deterministic (greedy); "
                            "higher values introduce more randomness."
                        ),
                    },
                ),
                "max_tokens": (
                    "INT",
                    {
                        "default": 512,
                        "min": 16,
                        "max": 8192,
                        "tooltip": "Maximum number of tokens to generate.",
                    },
                ),
                "timeout": (
                    "FLOAT",
                    {
                        "default": 120.0,
                        "min": 5.0,
                        "max": 600.0,
                        "step": 5.0,
                        "tooltip": (
                            "HTTP request timeout in seconds. "
                            "Increase for slow hardware or very long generations."
                        ),
                    },
                ),
            },
            "optional": {
                "custom_model_name": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": (
                            "Override the model identifier with any custom string. "
                            "Useful when running a non-GGUF backend or a locally "
                            "renamed model. Leave blank to use the dropdown above."
                        ),
                    },
                ),
                "system_prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                        "tooltip": (
                            "Optional system prompt. Leave blank to omit the system turn."
                        ),
                    },
                ),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "generate"
    CATEGORY = "ToriiGate/API"

    def generate(
        self,
        prompt: str,
        server_url: str = "http://127.0.0.1:8080",
        model_name: str = "DraconicDragon/ToriiGate-0.5-GGUF:Q4_K_M",
        temperature: float = 0.7,
        max_tokens: int = 512,
        timeout: float = 120.0,
        custom_model_name: str = "",
        system_prompt: str = "",
    ):
        resolved_model = custom_model_name.strip() if custom_model_name.strip() else model_name.strip()
        actual_server_url = server_url.strip().rstrip("/")

        print(
            f"[ToriiGate] Caption generation started (Text API) "
            f"(server={actual_server_url}, model={resolved_model}, "
            f"max_tokens={max_tokens}, temperature={actual_temperature:.2f})."
        )

        payload = build_text_payload(
            model_name=resolved_model,
            system_prompt=system_prompt,
            user_text=prompt,
            temperature=float(temperature),
            max_tokens=int(max_tokens),
        )

        import time
        start_time = time.perf_counter()
        result = send_chat_request(server_url=actual_server_url, payload=payload, timeout=float(timeout))
        elapsed = time.perf_counter() - start_time
        
        print(f"[ToriiGate] Caption generation finished in {elapsed:.1f}s ({len(result)} chars).")
        return (result.strip(),)


# ---------------------------------------------------------------------------
# ComfyUI registration maps (imported by nodes.py)
# ---------------------------------------------------------------------------

NODE_CLASS_MAPPINGS_API = {
    "ToriiGate_LlamaCppVisionGenerate": LlamaCppVisionGenerate,
    "ToriiGate_LlamaCppTextGenerate": LlamaCppTextGenerate,
}

NODE_DISPLAY_NAME_MAPPINGS_API = {
    "ToriiGate_LlamaCppVisionGenerate": "ToriiGate Llama.cpp Vision Generate",
    "ToriiGate_LlamaCppTextGenerate": "ToriiGate Llama.cpp Text Generate",
}
