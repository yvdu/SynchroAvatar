# SynchroAvatar

SynchroAvatar is a digital human video synthesis framework based on **CosyVoice2** and **Real3DPortrait**.

Built upon open-source models, this project provides an integrated pipeline for speech-driven and language-model-driven avatar video generation.

---

## Features

- 🎧 Voice pack generation based on **CosyVoice2**
- 📝 No reference transcript required for the input audio
- 🤖 Supports response narration from **local large language models** or **LLM APIs**
- 🖼️ A simple front-end that allows users to select:
  - Portrait image
  - Reference audio
  - Background image
  - Custom narration text
- 🆕 **Real-time, full-duplex conversation service (`realtime/`)**: microphone → ASR → LLM →
  streaming talking avatar, with **barge-in interruption** and **multi-user concurrency**
  (see "Real-time Conversation Service" below).

---

## Environment Setup

It is recommended to use **Conda** to create an isolated environment.

```bash
conda create -n SynchroAvatar -y python=3.9
conda activate SynchroAvatar
conda install -y -c conda-forge pynini==2.1.5
pip install -r requirements.txt
conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=12.1 -c pytorch -c nvidia
sudo apt-get install sox libsox-dev
```

> Windows users: install system `sox` from the [official SoX package](https://sourceforge.net/projects/sox/);
> for Python deps use `pip install -r requirements_win.txt` (Linux-only packages removed), then install a matching torch:
> `pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121`

## Installation

### 1. CosyVoice2 Models

Run the following commands under the `CosyVoice-main` directory:

```
mkdir -p pretrained_models
git clone https://www.modelscope.cn/iic/CosyVoice2-0.5B.git pretrained_models/CosyVoice2-0.5B
git clone https://www.modelscope.cn/iic/CosyVoice-ttsfrd.git pretrained_models/CosyVoice-ttsfrd
```

------

### 2. (Optional) Whisper Model

Download any Whisper model (e.g., `medium`) and place it under the `CosyVoice-main` directory.

------

### 3. Real3DPortrait Models

Please follow the instructions in `Real3DPortrait-main/README.md` to download:

- 3DMM BFM model
- Pretrained Real3DPortrait models

------

### 4. (Optional) Hubert Model (Network Issue Workaround)

If you encounter network issues while downloading the `hubert_model`, you may download it from the following Baidu Cloud link:

- Link: https://pan.baidu.com/s/1Yr8lUNpi12p9guDUlAygmg
- Extraction code: `cwzu`

After downloading, place the model under `Real3DPortrait-main/`.

## Usage

### 1. Text-driven Video Generation (GUI)

```
python demo.py
```

Then select in the interface: portrait image, reference audio, background image, narration text.

### 2. Local LLM-driven Video Generation

Replace `model_name` in the script with the path to your local LLM, then run:

```
python LLM_local_example.py --audio xx.mp3 --image xx.jpg --text "Your input text"
```

### 3. LLM API-driven Video Generation

Replace `api_key` and `base_url` in the script, then run:

```
python LLM_API_example.py \
  --audio path_to_reference_audio \
  --image path_to_face_image \
  --bg_img path_to_background_image \
  --text "Prompt sent to the LLM"
```

### 4. Continuous Video Generation via Terminal

After configuring `api_key` and `base_url`, run:

```
python LLM_API_example_continue.py --audio path_to_reference_audio --image path_to_face_image
```

------

## Real-time Conversation Service (`realtime/`)

Keeps CosyVoice2 + Real3DPortrait **resident in GPU memory** and exposes a real-time
"listen → think → speak → render" loop over FastAPI WebSocket:

```
mic/text → silero-VAD + faster-whisper (ASR) → LLM (streaming) → sentence split
         → CosyVoice2 (TTS) + Real3DPortrait (render) → per-sentence mp4 streamed back
```

Highlights:

- **Streaming, low latency**: the LLM streams tokens and we split sentences on the fly, so
  sentence #1 renders while sentence #2 is still being generated ("render while playing").
- **Full-duplex barge-in**: when the user starts speaking (VAD `speech_start`), the current
  answer is cancelled at three levels (LLM stream / render queue / the frame being rendered) —
  no need to wait for it to finish.
- **Multi-user concurrency**: every browser session has its own config / voice / memory /
  interrupt-round / ASR state, isolated by a session token `sid`.
  > ⚠️ On a single GPU, rendering is still a **single serial worker** (fair queue); true
  > parallelism needs multiple GPUs + multiple workers.

### Install & Run

```bash
pip install -r realtime/requirements.txt

# Recommended for users in CN: set HF mirror first
# PowerShell: $env:HF_ENDPOINT = "https://hf-mirror.com"
python -m realtime.download_models    # one-click weight download

# Enable LLM (OpenAI-compatible, any gateway)
#   $env:SYNCHRO_LLM_API_KEY  = "sk-xxx"
#   $env:SYNCHRO_LLM_BASE_URL = "https://api.openai.com/v1"
#   $env:SYNCHRO_LLM_MODEL    = "gpt-4o-mini"

uvicorn realtime.server:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000/`, submit the config (face image + reference audio), then chat via
text input or the microphone. See `realtime/README.md` for details.

### Performance Optimizations (this version)

- **Image-side preprocessing cache**: face segmentation / torso inpainting / 3DMM fitting are
  computed once per session and reused for every subsequent sentence (the biggest speedup in
  multi-turn chat). Tune entries via `SYNCHRO_R3D_IMG_CACHE`.
- **CUDA tuning**: `cudnn.benchmark` (fixed 512×512 frames) and TF32 (Ampere+) enabled.
- **Adjustable chunking**: smaller `SYNCHRO_SOFT_LIMIT` / `SYNCHRO_HARD_LIMIT` → earlier first clip.

### Common Environment Variables

| Variable | Default | Meaning |
|----------|---------|---------|
| `SYNCHRO_ENABLE_LLM` | `1` | If off, input text goes straight to TTS (no LLM) |
| `SYNCHRO_LLM_API_KEY` / `SYNCHRO_LLM_BASE_URL` / `SYNCHRO_LLM_MODEL` | - | OpenAI-compatible LLM config |
| `WHISPER_MODEL` | `small` | faster-whisper size (`tiny`/`base`/`small`...) |
| `WHISPER_DEVICE` | `auto` | Set `cpu` to move ASR off the GPU |
| `SYNCHRO_COSY_FP16` | `0` | Set `1` for half-precision CosyVoice2 (~1GB saved) |
| `SYNCHRO_R3D_IMG_CACHE` | `4` | Real3DPortrait image-preprocessing cache size |
| `SYNCHRO_SOFT_LIMIT` / `SYNCHRO_HARD_LIMIT` | `18` / `40` | Streaming sentence-split granularity |

---

## VRAM Requirements & Minimum Spec

The realtime service keeps CosyVoice2 + Real3DPortrait + Whisper + VAD resident at once:

| Component | VRAM |
|-----------|------|
| CosyVoice2-0.5B | fp32 ≈ 2~3 GB / fp16 ≈ 1~2 GB |
| Real3DPortrait (render) | ≈ 3~5 GB (heaviest) |
| faster-whisper `small` (GPU) | ≈ 1 GB |
| silero-VAD | < 0.1 GB |
| **Total** | **≈ 7~9 GB** |

- **Recommended**: 12 GB. **Minimum**: 8 GB.
- **6 GB (e.g. RTX 2060) can barely run it** with memory-saving switches:

```powershell
$env:WHISPER_DEVICE        = "cpu"
$env:WHISPER_MODEL         = "base"
$env:SYNCHRO_COSY_FP16     = "1"
$env:SYNCHRO_R3D_IMG_CACHE = "2"
uvicorn realtime.server:app --host 127.0.0.1 --port 8000
```

> If VRAM is still insufficient, use the offline GUI `python demo.py` (one clip at a time, lower peak).

---

## Voice Pack / Audio Generation

For voice pack generation and audio synthesis, please refer to:

👉 https://github.com/yvdu/CosyVoice2-voice-pack
