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

If you plan to use the local Transformers node instead of the API, install the requirements:

```bash
# Standard Python / venv
pip install -r ComfyUI/custom_nodes/comfyui_toriigate/requirements.txt

# ComfyUI Portable (Windows)
python_embeded\python.exe -m pip install -r ComfyUI\custom_nodes\comfyui_toriigate\requirements.txt
```

---

## Usage Methods

This node pack provides two different ways to run ToriiGate. 

### 1. Llama.cpp API (Recommended)
Connects to an external `llama-server.exe` running a GGUF version of the model.
- **Node**: `ToriiGate Llama.cpp Vision Generate`
- **Model format**: `.gguf` only
- **Pros**: Much faster inference, lower VRAM usage, bypasses python dependency issues.
- **Cons**: You need to run the llama-server manually before starting ComfyUI.

### 2. Transformers / PyTorch (Local)
Runs the model natively inside ComfyUI using the `transformers` library.
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

## Node Reference

- **ToriiGate Grounding Builder**: Compiles tags, characters, and descriptions into the final prompt string needed by the generator nodes.
- **ToriiGate Captioner**: Local PyTorch generator. Takes the image and the prompt string.
- **ToriiGate Llama.cpp Vision Generate**: API generator. Connects to `llama-server` and passes the image and prompt string.
- **ToriiGate Llama.cpp Text Generate**: Text-only node for chatting with the model.
