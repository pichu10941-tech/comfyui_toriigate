"""
ToriiGate – Embedded llama.cpp GGUF Node
=========================================
Loads the ToriiGate GGUF model directly via llama-cpp-python,
eliminating the need for a separate llama-server process.

Model: DraconicDragon/ToriiGate-0.5-GGUF (HuggingFace)

Usage:
  - Install: pip install llama-cpp-python (with CUDA: --extra-index-url .../whl/cu124)
  - Auto-downloads GGUF + mmproj from HuggingFace on first run
  - Or provide local paths via gguf_path / mmproj_path
"""

import base64
import gc
import io
import os
import random
import time
from pathlib import Path

import numpy as np
from PIL import Image

from .prompts import system_prompt

# ===========================================================================
# Constants
# ===========================================================================

GGUF_REPO = "DraconicDragon/ToriiGate-0.5-GGUF"

GGUF_QUANTS = [
    "Q4_K_M",   # 3.07 GB  ← recommended balance
    "Q4_K_S",   # 2.92 GB
    "Q4_0",     # 2.90 GB
    "Q4_1",     # 3.16 GB
    "IQ4_NL",   # 2.98 GB
    "Q5_K_M",   # 3.51 GB
    "Q5_K_S",   # 3.43 GB
    "Q6_K",     # 3.99 GB
    "Q8_0",     # 5.16 GB
    "Q3_K_L",   # 2.69 GB
    "Q3_K_M",   # 2.54 GB
    "Q3_K_S",   # 2.34 GB
    "Q2_K",     # 2.12 GB
    "bf16",     # 9.70 GB
]

# mmproj files available in the repo — pick the best one for each quant
_MMPROJ = {
    "bf16": "ToriiGate-0.5-bf16.mmproj.gguf",
    "Q8_0": "ToriiGate-0.5-Q8_0.mmproj.gguf",
    "fp16": "ToriiGate-0.5-fp16.mmproj.gguf",  # works with ALL quants
}


def _mmproj_for_quant(quant: str) -> str:
    """Pick the best mmproj file for a given quantization."""
    if quant == "bf16":
        return _MMPROJ["bf16"]
    if quant == "Q8_0":
        return _MMPROJ["Q8_0"]
    return _MMPROJ["fp16"]


def _gguf_filename(quant: str) -> str:
    return f"ToriiGate-0.5-{quant}.gguf"


# ===========================================================================
# ComfyUI model directory detection
# ===========================================================================

def _comfyui_models_dir() -> Path:
    """Auto-detect ComfyUI's models/LLM directory from the node's location.

    nodes_gguf.py lives at:  ComfyUI/custom_nodes/comfyui_toriigate/nodes_gguf.py
    So models/LLM is at:     ../.. /models/LLM/ToriiGate/
    """
    node_dir = Path(__file__).resolve().parent          # .../custom_nodes/comfyui_toriigate/
    comfyui_root = node_dir.parent.parent               # .../ComfyUI/
    return comfyui_root / "models" / "LLM"


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


# ===========================================================================
# HuggingFace download helpers — saves to ComfyUI/models/LLM/ToriiGate/
# ===========================================================================

def _download_from_hf(repo_id: str, filename: str, target_dir: Path) -> str:
    """Download a single file from HuggingFace Hub to *target_dir*."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        raise ImportError(
            "[ToriiGate GGUF] 'huggingface_hub' is required for auto-download.\n"
            "Install: pip install huggingface_hub\n"
            "Or provide local paths via gguf_path / mmproj_path."
        )

    local_path = target_dir / filename
    if local_path.exists():
        print(f"[ToriiGate GGUF] Already exists: {local_path}")
        return str(local_path)

    print(f"[ToriiGate GGUF] Downloading: {repo_id}/{filename}")
    print(f"[ToriiGate GGUF]   → {local_path}")
    # hf_hub_download still uses HF cache internally, but we symlink/copy
    result = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_dir=str(target_dir),
        local_dir_use_symlinks=False,  # actual file, not symlink
        resume_download=True,
    )
    return result


def _resolve_paths(model_quant: str, gguf_path: str, mmproj_path: str):
    """Return (gguf_local, mmproj_local), downloading to models/LLM/ if needed."""
    target_dir = _ensure_dir(_comfyui_models_dir())

    # --- GGUF ---
    if gguf_path and os.path.isfile(gguf_path):
        gguf_local = gguf_path
        print(f"[ToriiGate GGUF] Using local GGUF: {gguf_local}")
    else:
        fname = _gguf_filename(model_quant)
        gguf_local = _download_from_hf(GGUF_REPO, fname, target_dir)

    # --- MMproj ---
    if mmproj_path and os.path.isfile(mmproj_path):
        mmproj_local = mmproj_path
        print(f"[ToriiGate GGUF] Using local mmproj: {mmproj_local}")
    else:
        fname = _mmproj_for_quant(model_quant)
        mmproj_local = _download_from_hf(GGUF_REPO, fname, target_dir)

    return gguf_local, mmproj_local


# ===========================================================================
# ComfyUI Node
# ===========================================================================

class ToriiGateGGUFCaptioner:
    """
    Loads ToriiGate GGUF model directly via llama-cpp-python.
    No external server needed — the model runs inside ComfyUI's Python process.
    """

    _cache = {}   # cache_key -> Llama instance
    _banner_printed = set()

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {
                    "tooltip": "ComfyUI image tensor (B, H, W, C float32). Only the first image is used.",
                }),
            },
            "optional": {
                "model_quant": (GGUF_QUANTS, {
                    "default": "Q4_K_M",
                    "tooltip": "GGUF quantization. Q4_K_M is the recommended balance of quality vs size (3.07 GB).",
                }),
                "gguf_path": ("STRING", {
                    "default": "",
                    "tooltip": "Local path to the GGUF model file. Leave empty to auto-download from HuggingFace.",
                }),
                "mmproj_path": ("STRING", {
                    "default": "",
                    "tooltip": "Local path to the mmproj (multimodal projector) file. Leave empty to auto-download.",
                }),
                "n_gpu_layers": ("INT", {
                    "default": -1,
                    "min": -1,
                    "max": 200,
                    "step": 1,
                    "tooltip": "GPU offload layers. -1 = all on GPU (fastest), 0 = CPU only, 20+ = partial offload.",
                }),
                "n_ctx": ("INT", {
                    "default": 4096,
                    "min": 512,
                    "max": 32768,
                    "step": 512,
                    "tooltip": "Context size in tokens. Higher = more memory.",
                }),
                "max_pixels_mp": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.1,
                    "max": 8.0,
                    "step": 0.1,
                    "tooltip": "Max image resolution in megapixels. Lower = faster prompt processing.",
                }),
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "Caption prompt. Connect the Grounding Builder output here, or type your own.",
                    "forceInput": True,
                }),
                "max_new_tokens": ("INT", {
                    "default": 512,
                    "min": 64,
                    "max": 4096,
                    "tooltip": "Maximum tokens to generate.",
                }),
                "temperature": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.01,
                    "tooltip": "Sampling temperature.",
                }),
                "decoding": (["sample", "greedy_fast"], {
                    "default": "greedy_fast",
                    "tooltip": "sample = temperature sampling; greedy_fast = deterministic (faster).",
                }),
                "keep_model_alive": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Keep model loaded in memory between runs. Disable to free VRAM after each caption.",
                }),
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 0xFFFFFFFFFFFFFFFF,
                    "tooltip": "Random seed. 0 = random (not reproducible).",
                }),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("caption",)
    FUNCTION = "caption"
    CATEGORY = "ToriiGate/GGUF"

    # ── node entry point ──────────────────────────────────────────────────

    def caption(
        self,
        image,
        model_quant="Q4_K_M",
        gguf_path="",
        mmproj_path="",
        n_gpu_layers=-1,
        n_ctx=4096,
        max_pixels_mp=1.0,
        prompt="",
        max_new_tokens=512,
        temperature=0.5,
        decoding="greedy_fast",
        keep_model_alive=False,
        seed=0,
    ):
        # --- deferred import so ComfyUI can still load the rest of the node ---
        try:
            from llama_cpp import Llama
        except ImportError as exc:
            raise ImportError(
                "[ToriiGate GGUF] 'llama-cpp-python' is required.\n"
                "Install with:\n"
                "  pip install llama-cpp-python\n"
                "For CUDA support (recommended on NVIDIA GPUs):\n"
                "  pip install llama-cpp-python --extra-index-url "
                "https://abetlen.github.io/llama-cpp-python/whl/cu124"
            ) from exc

        # ── print banner once ──────────────────────────────────────────
        model_tag = gguf_path or model_quant
        if model_tag not in self._banner_printed:
            self._banner_printed.add(model_tag)
            print(self._BANNER)

        # ── resolve paths ──────────────────────────────────────────────
        actual_temp = 0.0 if decoding == "greedy_fast" else float(temperature)
        ngl = int(n_gpu_layers)
        ctx = int(n_ctx)
        cache_key = (model_quant if not gguf_path else gguf_path, ngl, ctx)

        if not prompt:
            prompt = "Describe this image in detail."

        print(f"\n[ToriiGate GGUF] quant={model_quant}, n_gpu_layers={ngl}, n_ctx={ctx}")
        gguf_local, mmproj_local = _resolve_paths(model_quant, gguf_path, mmproj_path)

        # ── load model ─────────────────────────────────────────────────
        if cache_key not in self._cache:
            print(f"[ToriiGate GGUF] Loading model via llama-cpp-python...")
            t0 = time.perf_counter()
            llm = Llama(
                model_path=gguf_local,
                mmproj=mmproj_local,
                n_gpu_layers=ngl,
                n_ctx=ctx,
                flash_attn=True,
                verbose=False,
            )
            t1 = time.perf_counter()
            print(f"[ToriiGate GGUF] Model loaded in {t1 - t0:.1f}s.")
            self._cache[cache_key] = llm
        else:
            print(f"[ToriiGate GGUF] Reusing cached model.")
            llm = self._cache[cache_key]

        # ── prepare image ──────────────────────────────────────────────
        img_np = (image[0].detach().cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        img_pil = Image.fromarray(img_np)
        if img_pil.mode != "RGB":
            img_pil = img_pil.convert("RGB")

        # optional resize
        current_mp = (img_pil.width * img_pil.height) / 1_000_000
        if current_mp > max_pixels_mp:
            scale = (max_pixels_mp / current_mp) ** 0.5
            new_w = max(1, int(img_pil.width * scale))
            new_h = max(1, int(img_pil.height * scale))
            img_pil = img_pil.resize((new_w, new_h), Image.Resampling.LANCZOS)
            print(f"[ToriiGate GGUF] Resized {img_pil.width}x{img_pil.height} ({max_pixels_mp:.1f} MP)")

        # base64 PNG
        buf = io.BytesIO()
        img_pil.save(buf, format="PNG")
        b64_data = base64.b64encode(buf.getvalue()).decode("utf-8")

        # ── build messages ─────────────────────────────────────────────
        seed_val = int(seed) if seed != 0 else random.SystemRandom().randint(1, 2**63 - 1)

        messages = []
        if system_prompt.strip():
            messages.append({"role": "system", "content": system_prompt.strip()})
        messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_data}"}},
            ],
        })

        # ── generate ──────────────────────────────────────────────────
        print(f"[ToriiGate GGUF] Generating (max_tokens={max_new_tokens}, temp={actual_temp:.2f})...")
        t0 = time.perf_counter()

        response = llm.create_chat_completion(
            messages=messages,
            max_tokens=int(max_new_tokens),
            temperature=float(actual_temp),
            seed=seed_val,
        )

        elapsed = time.perf_counter() - t0

        try:
            caption = response["choices"][0]["message"]["content"].strip()
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(
                f"[ToriiGate GGUF] Unexpected response format:\n{response}"
            ) from exc

        # rough token count
        approx_tokens = max(1, len(caption) // 4)
        print(f"[ToriiGate GGUF] Done in {elapsed:.1f}s ({approx_tokens} tok, {approx_tokens/elapsed:.1f} tok/s).")

        # ── cleanup ───────────────────────────────────────────────────
        if not keep_model_alive:
            print("[ToriiGate GGUF] Unloading model (keep_model_alive=False).")
            self._cache.pop(cache_key, None)
            del llm
            gc.collect()

        return (caption,)

    # ── banner ──────────────────────────────────────────────────────────

    _BANNER = r"""
======================================================================

  TTTTT  OOO  RRRR   III  III   GGG    A    TTTTT EEEEE
    T   O   O R   R   I    I   G      A A     T   E
    T   O   O RRRR    I    I   G  GG AAAAA    T   EEEE
    T   O   O R  R    I    I   G   G A   A    T   E
    T    OOO  R   R  III  III   GGG  A   A    T   EEEEE

                         ToriiGate-0.5
                     Embedded GGUF mode (llama-cpp-python)
======================================================================

"""


# ===========================================================================
# Registration maps (imported by nodes.py)
# ===========================================================================

NODE_CLASS_MAPPINGS_GGUF = {
    "ToriiGate_GGUFCaptioner": ToriiGateGGUFCaptioner,
}

NODE_DISPLAY_NAME_MAPPINGS_GGUF = {
    "ToriiGate_GGUFCaptioner": "ToriiGate GGUF Captioner (Embedded)",
}
