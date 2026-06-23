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
5. 🆕 **实时全双工对话服务（`realtime/`）**：麦克风说话 → ASR → LLM → 数字人流式回话，
   支持**随时打断**与**多用户并发**（详见下方「实时对话服务」）。

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

> Windows 用户：系统级 `sox` 用 [SoX 官方包](https://sourceforge.net/projects/sox/) 安装即可；
> Python 依赖请改用 `pip install -r requirements_win.txt`（已剔除 Linux-only 包），
> 再单独安装匹配的 torch：
> `pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121`

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

## 实时全双工对话服务（`realtime/`）

把 CosyVoice2 + Real3DPortrait **常驻显存**，通过 FastAPI WebSocket 暴露一条
「听 → 想 → 说 → 演」的实时闭环：

```
麦克风/文本 → silero-VAD + faster-whisper(ASR) → LLM(流式) → 流式分句
            → CosyVoice2(TTS) + Real3DPortrait(渲染) → 逐句 mp4 流式回放
```

核心特性：

- **流式低延迟**：LLM 边出 token 边切句，第 1 句渲染时第 2 句还在生成，「边渲染边播放」。
- **全双工打断**：用户一开口（VAD `speech_start`）立即三级取消当前回答（LLM 流 / 渲染队列 / 正在渲染的那一帧），不必等它说完。
- **多用户并发**：每个浏览器会话独立的 配置 / 音色 / 记忆 / 打断轮 / ASR 状态，按会话令牌 `sid` 隔离。
  > ⚠️ 单卡 GPU 下渲染仍是**单 worker 串行**（公平排队）；真并行需多卡 + 多 worker。

### 安装与启动

```bash
pip install -r realtime/requirements.txt

# 国内建议先设 HF 镜像
# PowerShell: $env:HF_ENDPOINT = "https://hf-mirror.com"
python -m realtime.download_models          # 一键下载权重（CosyVoice2/BFM/R3D/Whisper/VAD...）

# 启用 LLM（OpenAI 兼容，可指向任意网关）
# PowerShell:
#   $env:SYNCHRO_LLM_API_KEY  = "sk-xxx"
#   $env:SYNCHRO_LLM_BASE_URL = "https://api.openai.com/v1"
#   $env:SYNCHRO_LLM_MODEL    = "gpt-4o-mini"

uvicorn realtime.server:app --host 127.0.0.1 --port 8000
```

启动后浏览器打开 `http://127.0.0.1:8000/`，先「提交配置」（人脸图 + 参考音频），
再用「文本输入」或「开始说话」与数字人对话。详见 `realtime/README.md`。

### 性能优化（本版新增）

- **图像侧预处理缓存**：人脸分割 / 躯干 inpaint / 3DMM 拟合在一次会话内只算一次，
  后续每句直接命中缓存（多轮对话最大的提速点）。可用 `SYNCHRO_R3D_IMG_CACHE` 调缓存条目数。
- **CUDA 加速**：自动开启 `cudnn.benchmark`（固定 512×512 帧）与 TF32（30/40 系有效）。
- **切句粒度可调**：`SYNCHRO_SOFT_LIMIT` / `SYNCHRO_HARD_LIMIT` 越小，首段视频越早播放。

### 常用环境变量

| 变量 | 默认 | 说明 |
|------|------|------|
| `SYNCHRO_ENABLE_LLM` | `1` | 关掉则输入文本直接 TTS（不走 LLM） |
| `SYNCHRO_LLM_API_KEY` / `SYNCHRO_LLM_BASE_URL` / `SYNCHRO_LLM_MODEL` | - | OpenAI 兼容 LLM 配置 |
| `WHISPER_MODEL` | `small` | faster-whisper 模型大小（`tiny`/`base`/`small`...） |
| `WHISPER_DEVICE` | `auto` | 设 `cpu` 可把 ASR 挪到 CPU，给渲染让出显存 |
| `SYNCHRO_COSY_FP16` | `0` | 设 `1` 让 CosyVoice2 半精度，省约 1GB 显存 |
| `SYNCHRO_R3D_IMG_CACHE` | `4` | Real3DPortrait 图像预处理缓存条目数 |
| `SYNCHRO_SOFT_LIMIT` / `SYNCHRO_HARD_LIMIT` | `18` / `40` | 流式切句粒度 |

---

## 显存需求与最低配置

实时服务会把 CosyVoice2 + Real3DPortrait + Whisper + VAD **同时常驻显存**，粗估占用：

| 组件 | 显存 |
|------|------|
| CosyVoice2-0.5B | fp32 ≈ 2~3 GB / fp16 ≈ 1~2 GB |
| Real3DPortrait（渲染） | ≈ 3~5 GB（最吃显存） |
| faster-whisper `small`（GPU） | ≈ 1 GB |
| silero-VAD | < 0.1 GB |
| **合计** | **≈ 7~9 GB** |

- **推荐**：12 GB 显存舒适运行；**最低**：8 GB 可跑。
- **6 GB（如 RTX 2060）勉强可跑**，需开启省显存开关：

```powershell
$env:WHISPER_DEVICE   = "cpu"     # Whisper 上 CPU
$env:WHISPER_MODEL    = "base"    # 或 tiny
$env:SYNCHRO_COSY_FP16 = "1"      # CosyVoice2 半精度
$env:SYNCHRO_R3D_IMG_CACHE = "2"
uvicorn realtime.server:app --host 127.0.0.1 --port 8000
```

> 显存实在不够时，改用离线 GUI 版 `python demo.py`（一次只处理一段，对显存峰值更友好）。

---

## Voice Pack / Audio Generation

语音包生成与音频生成相关内容，请参考：

👉 https://github.com/yvdu/CosyVoice2-voice-pack
