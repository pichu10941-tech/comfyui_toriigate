# comfyui_toriigate

ComfyUI custom nodes for [Minthy/ToriiGate-0.5](https://huggingface.co/Minthy/ToriiGate-0.5), an image captioning model for anime-style and digital art.

Original Model: [Minthy/ToriiGate-0.5](https://huggingface.co/Minthy/ToriiGate-0.5)
GGUF Models: [DraconicDragon/ToriiGate-0.5-GGUF](https://huggingface.co/DraconicDragon/ToriiGate-0.5-GGUF)

---

## Installation

Clone this repository into your `ComfyUI/custom_nodes` folder:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/litch230/comfyui_toriigate.git
```

Install the base requirements, then the backend of your choice:

```bash
# Standard Python / venv
pip install -r ComfyUI/custom_nodes/comfyui_toriigate/requirements.txt

# ComfyUI Portable (Windows)
python_embeded\\python.exe -m pip install -r ComfyUI\\custom_nodes\\comfyui_toriigate\\requirements.txt
```

**If using the Embedded GGUF node**, also install `llama-cpp-python`:

```bash
# CPU-only
pip install llama-cpp-python

# NVIDIA GPU (CUDA 12.x) — recommended
pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124
```

---

## Usage Methods

This node pack provides three different ways to run ToriiGate. 

### 1. Embedded GGUF (NEW — Recommended)
Loads the GGUF model directly inside ComfyUI via `llama-cpp-python`.
- **Node**: `ToriiGate GGUF Captioner (Embedded)`
- **Model format**: `.gguf` (auto-downloaded from HuggingFace, or use local files)
- **Pros**: Fast inference, GPU-accelerated, no external server needed.
- **Cons**: Requires `llama-cpp-python` with CUDA support installed.

### 2. Llama.cpp API
Connects to an external `llama-server.exe` running a GGUF version of the model.
- **Node**: `ToriiGate Llama.cpp Vision Generate`
- **Model format**: `.gguf` only
- **Pros**: Fast inference, lower VRAM usage, bypasses python dependency issues.
- **Cons**: You need to run the llama-server manually before starting ComfyUI.

### 3. Transformers / PyTorch (Local)
Runs the original PyTorch model inside ComfyUI using the `transformers` library.
- **Node**: `ToriiGate Captioner`
- **Model format**: Original unquantized HuggingFace model format (cannot run GGUF).
- **Pros**: No external server needed.
- **Cons**: Slower, uses more VRAM.

---

## Running the Llama.cpp Server

For the best performance on NVIDIA GPUs, follow these steps to run the API server:

1. Download the latest release from the [llama.cpp releases page](https://github.com/ggerganov/llama.cpp/releases). 
   - Choose the `win-cuda` zip matching your CUDA version (e.g. `llama-b4109-bin-win-cuda-12.4-x64.zip`).
   - If you have an AMD GPU, use `win-vulkan` or `win-rocm`.

2. Download the `cudart` zip file from the same release page (e.g. `cudart-llama-bin-win-cuda-12.4-x64.zip`). 
   - Extract the `cudart` files directly into the same folder where you extracted `llama-server.exe`. Without this, inference on NVIDIA GPUs will be very slow.

3. Open CMD in that folder and run the server with the following optimizations. *(Note: The `-m` and `--mmproj` flags are optional if you want to preload local files; if omitted, the ComfyUI node will automatically instruct the server to download and load the correct GGUF model)*:

```cmd
llama-server.exe [-m <model.gguf>] [--mmproj <mmproj.gguf>] -b 2048 -ub 1024 -fa on -fit on -fitt 1024 -ngl 999
```

Command breakdown:
- `-m`: (Optional) Path to your downloaded `.gguf` language model.
- `--mmproj`: (Optional) Path to your downloaded `.gguf` vision projector model (required if preloading a vision model locally).
- `-b 2048 -ub 1024`: Batch sizes.
- `-fa on`: Enables Flash Attention.
- `-fit on -fitt 1024`: Speeds up image processing.
- `-ngl 999`: Offloads all layers to the GPU.

Once the server is running on `http://127.0.0.1:8080`, you can generate captions using the `ToriiGate Llama.cpp Vision Generate` node in ComfyUI.

---

## Running the Embedded GGUF Node (Recommended)

The `ToriiGate GGUF Captioner (Embedded)` node runs the model directly inside ComfyUI — no separate server needed.

### How it works

1. On first use, the node automatically downloads the GGUF model and mmproj from [DraconicDragon/ToriiGate-0.5-GGUF](https://huggingface.co/DraconicDragon/ToriiGate-0.5-GGUF) to `ComfyUI/models/LLM/`.
2. The model is loaded via `llama-cpp-python` with full GPU acceleration (CUDA).
3. Subsequent calls reuse the cached model (unless `keep_model_alive` is disabled).

Files are saved as:
```
ComfyUI/models/LLM/
├── ToriiGate-0.5-Q4_K_M.gguf        ← language model (~3 GB)
└── ToriiGate-0.5-fp16.mmproj.gguf   ← vision projector (~200 MB)
```

### Key parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model_quant` | Q4_K_M | Quantization level. Q4_K_M = 3.07 GB balance. |
| `gguf_path` | (empty) | Local GGUF file path. Empty = auto-download. |
| `mmproj_path` | (empty) | Local mmproj path. Empty = auto-download. |
| `n_gpu_layers` | -1 | GPU offload layers. -1 = all on GPU. |
| `n_ctx` | 4096 | Context window size. |
| `max_pixels_mp` | 1.0 | Image resolution cap in megapixels. Lower = faster. |
| `keep_model_alive` | False | Keep model loaded between runs (saves reload time). |
| `decoding` | greedy_fast | greedy_fast = deterministic; sample = creative. |

### Workflow example

```
Image → ToriiGate GGUF Captioner (Embedded) → caption (STRING)
                ↑
        ToriiGate Grounding Builder (optional)
```

---

## Node Reference

- **ToriiGate Grounding Builder**: Compiles tags, characters, and descriptions into the final prompt string needed by the generator nodes.
- **ToriiGate GGUF Captioner (Embedded)**: (NEW) Direct GGUF inference via `llama-cpp-python`. No external server needed.
- **ToriiGate Captioner**: Local PyTorch generator. Takes the image and the prompt string.
- **ToriiGate Llama.cpp Vision Generate**: API generator. Connects to `llama-server` and passes the image and prompt string.
- **ToriiGate Llama.cpp Text Generate**: Text-only node for chatting with the model.
