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

---

## Environment Setup

It is recommended to use **Conda** to create an isolated environment.

```bash
conda create -n SynchroAvatar -y python=3.9
conda activate SynchroAvatar
```

Install dependencies:

```
conda install -y -c conda-forge pynini==2.1.5
pip install -r requirements.txt
```

Install PyTorch with CUDA 12.1 support:

```
conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=12.1 -c pytorch -c nvidia
```

Install audio-related system dependencies:

```
sudo apt-get install sox libsox-dev
```

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

Please follow the instructions in:

```
Real3DPortrait-main/README.md
```

to download:

- 3DMM BFM model
- Pretrained Real3DPortrait models

------

### 4. (Optional) Hubert Model (Network Issue Workaround)

If you encounter network issues while downloading the `hubert_model`, you may download it from the following Baidu Cloud link:

- Link: https://pan.baidu.com/s/1Yr8lUNpi12p9guDUlAygmg
- Extraction code: `cwzu`

After downloading, place the model under:

```
Real3DPortrait-main/
```

## Usage

### 1. Text-driven Video Generation (GUI)

Run:

```
python demo.py
```

Then select in the interface:

- Portrait image
- Reference audio
- Background image
- Input narration text

------

### 2. Local LLM-driven Video Generation

Replace `model_name` in the script with the path to your local large language model, then run:

```
python LLM_local_example.py --audio xx.mp3 --image xx.jpg --text "Your input text"
```

### 3. LLM API-driven Video Generation

Replace the following variables in the script:

- `api_key` with your API key
- `base_url` with the corresponding LLM API endpoint

Run:

```
python LLM_API_example.py \
  --audio path_to_reference_audio \
  --image path_to_face_image \
  --bg_img path_to_background_image \
  --text "Prompt sent to the LLM"
```

------

### 4. Continuous Video Generation via Terminal

After configuring `api_key` and `base_url`, run:

```
python LLM_API_example_continue.py --audio path_to_reference_audio --image path_to_face_image
```

Once the model is loaded, you can continuously input text in the terminal to interact with the LLM and generate sequential avatar videos.

## Voice Pack / Audio Generation

For voice pack generation and audio synthesis, please refer to:

👉 https://github.com/yvdu/CosyVoice2-voice-pack