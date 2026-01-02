# SynchroAvatar

本项目是一个基于 **CosyVoice2 / Real3DPortrait** 的数字人视频合成框架。

在开源模型的基础上，支持以下功能：

## Features

1. 基于 **CosyVoice2** 的语音包生成  
2. 无需提供参考音频对应的文本内容  
3. 支持 **本地大语言模型 / API** 的回复内容播报  
4. 提供一个简单前端，支持选择：
   - 人物图片  
   - 参考音频  
   - 背景图片  
   - 指定播报文本  

---

## Environment Setup

```bash
conda create -n SynchroAvatar -y python=3.9
conda activate SynchroAvatar
conda install -y -c conda-forge pynini==2.1.5
pip install -r requirements.txt
conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=12.1 -c pytorch -c nvidia
#音频相关依赖
sudo apt-get install sox libsox-dev
```

## Installation

### 1. CosyVoice2 模型下载

在 `CosyVoice-main` 文件夹下运行：

```
mkdir -p pretrained_models
git clone https://www.modelscope.cn/iic/CosyVoice2-0.5B.git pretrained_models/CosyVoice2-0.5B
git clone https://www.modelscope.cn/iic/CosyVoice-ttsfrd.git pretrained_models/CosyVoice-ttsfrd
```

------

### 2. （可选）Whisper 模型

下载任意 Whisper 模型（如 `medium`），并放置在 `CosyVoice-main` 文件夹下。

------

### 3. Real3DPortrait 模型

参考：

```
Real3DPortrait-main/README.md
```

下载以下内容：

- 3DMM BFM 模型
- Real3DPortrait 预训练模型

------

### 4. （可选）Hubert 模型下载（网络问题备用）

如果在下载 `hubert_model` 时遇到网络问题，可使用以下百度网盘链接：

- 链接：https://pan.baidu.com/s/1Yr8lUNpi12p9guDUlAygmg
- 提取码：`cwzu`

下载后放置于：

```
Real3DPortrait-main/
```

## Usage

### 1. 指定文本视频生成（GUI）

运行：

```
python demo.py
```

在界面中选择：

- 人物图片
- 参考音频
- 背景图片
- 输入指定播报文本

------

### 2. 本地大模型文本视频生成

将脚本中的 `model_name` 替换为你自己的本地大模型路径，然后运行：

```
python LLM_local_example.py --audio xx.mp3 --image xx.jpg --text 指定文本
```

### 3. 大模型 API 文本视频生成

在脚本中替换：

- `api_key` 为你的 API Key
- `base_url` 为对应的大模型 API 地址

运行：

```
python LLM_API_example.py \
  --audio 参考音频路径 \
  --image 人脸图片路径 \
  --bg_img 背景图片路径 \
  --text 向大模型发送的文本
```

### 4. 终端视频生成

同样先配置：

- `api_key`
- `base_url`

运行：

```
python LLM_API_example_continue.py --audio 参考音频路径 --image 人脸图片路径
```

模型加载完成后，即可在终端中持续输入文本，与大模型交互并生成视频。

------

## Voice Pack / Audio Generation

语音包生成与音频生成相关内容，请参考：

👉 https://github.com/yvdu/CosyVoice2-voice-pack