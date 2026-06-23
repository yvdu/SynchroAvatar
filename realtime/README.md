# SynchroAvatar 实时服务（P0 + P1 + P2 + P3 + P4）

把 CosyVoice2 + Real3DPortrait **常驻显存**，通过 FastAPI WebSocket 暴露：

- **P0 文本通道**：浏览器输入文本 → 服务端 TTS + 数字人 → 前端拉取 mp4 自动播放。
- **P1 语音通道**：浏览器麦克风 16k PCM 流式上行 → silero-VAD 切句 + faster-whisper 识别 → 自动派发到下游。
- **P2 LLM + 记忆 + 流式分句**：用户文本 → LLM `stream=True` 出 token → 按句末标点 / 长度切句 → 立刻送 TTS+渲染。滑动窗口 Memory 维持多轮上下文。
- **P3 句级流式渲染**：每句单独走 TTS+Real3DPortrait → 产出短 mp4 → WS 推 `video_chunk` → 前端按 `seq` 顺序串行播放。
- **P4 全双工打断**：用户开口（VAD `speech_start`）/ 发送新文本时，**三级打断**当前回答（LLM 流 + 渲染队列 + 正在渲染的那一帧）。前端立即停播 + 清队列。ASR 持续监听，不受影响。

---

## 1. 安装依赖

先按仓库根 `readme.md` 装好 CosyVoice2、Real3DPortrait 权重并跑通 `python demo.py`，然后：

```bash
pip install -r realtime/requirements.txt
```

### 一键下载预训练权重（推荐）

```bash
# 默认全跑（CosyVoice2 / BFM / Real3DPortrait ckpt / MediaPipe / HuBERT / Whisper / silero-vad）
python -m realtime.download_models

# 列出所有可选任务
python -m realtime.download_models --list

# 只下载某几个
python -m realtime.download_models --only cosyvoice,bfm,r3d_ckpt
```

国内强烈建议先设环境变量加速 HuggingFace：

```powershell
$env:HF_ENDPOINT = "https://hf-mirror.com"
```

Google Drive 在国内无法直连，**`bfm` / `r3d_ckpt` / `r3d_pretrained` 三个任务需要全局代理**，否则会失败（脚本会给出明确报错与处理建议）。或参考 `Real3DPortrait-main/README.md` 用百度网盘手动放好。

`faster-whisper` 在 Windows 上需 cuDNN/CUDA，CPU 也可跑（自动回退 int8）。  
`silero-vad` 包不可用时会自动从 `torch.hub` 加载 `snakers4/silero-vad`，**需要首次联网**。

---

## 2. 启动（仅内网）

仓库根目录：

```bash
# 启用 LLM：先配环境变量
$env:SYNCHRO_LLM_API_KEY = "sk-xxx"
$env:SYNCHRO_LLM_BASE_URL = "https://api.openai.com/v1"   # 或你的兼容网关
$env:SYNCHRO_LLM_MODEL    = "gpt-4o-mini"                  # 或 ernie-4.5-vl-28b-a3b / qwen2.5-7b 等

# 不启用 LLM（输入文本直接 TTS）：
$env:SYNCHRO_ENABLE_LLM = "0"

uvicorn realtime.server:app --host 0.0.0.0 --port 8000
# 仅本机：--host 127.0.0.1
```

**不要做任何端口映射 / 公网代理 / frp / ngrok**，就是"仅内网，不透传"。  
浏览器打开 `http://<内网IP>:8000/`。首次启动会加载 CosyVoice2 + Real3DPortrait + Whisper，**耗时较久**，等终端 `ready.` 再操作。

> ⚠️ 浏览器走 `http://非localhost` 时麦克风可能被拒（Secure Context 限制）。  
> 临时解法：`http://127.0.0.1:8000` 自测；或 Chrome 启动加  
> `--unsafely-treat-insecure-origin-as-secure=http://192.168.x.x:8000`；或内网套自签 https。

---

## 3. 使用流程

1. **会话配置**：上传人脸图、参考音频（必填），背景图、参考文本（选填），点"提交配置"。
2. **文本测试**：在"文本输入"里输入一句话 → LLM 会逐句出答复，每句一段 mp4 流式播放。
3. **语音测试**：点"开始说话"，授权麦克风 → 说一句话停顿 ≥0.6s → 服务端识别 → 走 LLM → 流式数字人。
4. **清空记忆**：右上"清空对话记忆"按钮，让 LLM 忘掉前面对话。

---

## 4. 接口

| 路径 | 方法 | 作用 |
|------|------|------|
| `/`                  | GET   | 静态页面 |
| `/api/config`        | POST  | 上传配置（multipart） |
| `/api/reset_memory`  | POST  | 清空 LLM 对话历史 |
| `/api/files/{name}`  | GET   | 下载生成的 mp4 |
| `/ws/text`           | WS    | 发 `{"text":"..."}`；接收事件流（见下） |
| `/ws/audio`          | WS    | 发二进制 16k mono Int16 PCM；接收事件流（含 ASR + 视频事件） |

**WS 服务端事件**（统一 JSON）：

| `type` | 字段 | 含义 |
|--------|------|------|
| `accepted`        | `text` | 服务端收到了用户输入 |
| `round_start`     | `round_id` | 新一轮回答开始，前端用其打开新回合（P4） |
| `interrupt`       | `round_id`, `reason` | 当前回答被打断（P4） |
| `speech_start`    | -      | ASR 检测到用户开口（`/ws/audio`） |
| `final`           | `text` | ASR 识别完一句 |
| `llm_sentence`    | `text`, `seq`, `round_id` | LLM 切出一句，准备送 TTS+渲染 |
| `video_chunk`     | `url`, `seq`, `is_last`, `round_id` | 一段 mp4 准备好，前端按 seq 顺序播 |
| `done`            | `round_id` | 本轮回答全部结束 |
| `error`           | `msg`, `seq?` | 出错 |

---

## 5. 关键设计点

### 句级流水线（P2 + P3 的核心）

```
用户文本 ─► LLM.stream() ─┐
                         │  按 [。！？!?\n] 硬切，配合 soft/hard 长度兜底
                         ▼
                  SentenceSplitter
                         │  逐句产出（生产者）
                         ▼
                  InferWorker.submit (single thread, GPU)
                         │  TTS 整句 wav + Real3DPortrait 整句 mp4
                         ▼
                  pipeline_run.drain
                         │  按 seq 顺序 ws.send_json({type:"video_chunk"})
                         ▼
                  浏览器 playQueue
                         │  expectedSeq 严格递增播放
                         ▼
                       <video>
```

- **LLM 与渲染重叠**：第 2 句 LLM 还在生成时，第 1 句已经进 worker 排队甚至开始渲染。
- **按 seq 严格有序播放**：`drain()` 等队首 `done` 才发送；前端 `expectedSeq` 严格递增；避免短句到达乱序导致回答错乱。
- **每句一个 mp4**：模型层最小入侵，不动 Real3DPortrait 源码。

### 记忆

- `realtime/llm.py:Memory` 是滑动窗口（默认 `max_turns=8`），每轮 = 1 user + 1 assistant，超过自动裁剪，保留 system。
- 不做向量检索/摘要——实时场景下额外延迟不值得。需要长记忆时替换 `Memory` 类即可。
- `system` prompt 默认强制"短句、无 markdown、无表情"，对 TTS 友好。

### 引擎层 `low_memory_usage=True`

- 走 Real3DPortrait 原本就支持的"逐帧 imageio writer.append_data"分支。
- 显存峰值更低 + 没有"先拼全段 tensor 再写文件"的额外内存往返延迟。
- **限制**：本句的全部帧还是要算完才能产出 mp4，无法做到"帧级流式推流"。要真正帧级流式需 WebRTC + fMP4 + 修改 forward_secc2video 的输出形式，那是 P5 级别工作。

---

## 6. P4 全双工打断：三级取消

### 协议层
| 事件 | 来源 | 含义 |
|------|------|------|
| `round_start` | 服务端发 | 新一轮回答开始，前端 `currentRoundId = round_id`、清 playQueue |
| `interrupt`   | 服务端发 | 当前回答已被打断（`reason="user_speech" / "new_input" / "disconnect"`） |
| `llm_sentence`/`video_chunk`/`done` | 服务端发 | 均带 `round_id`，前端用其隔离新旧回合 |

### 触发时机
- `/ws/audio` 收到 VAD `speech_start` **立即** `interrupt_round()`（不等 ASR 出文本，**这才是真全双工**）。
- `/ws/audio` ASR `final` 后兜底再 `interrupt_round()`，然后 `start_round(new_text)`。
- `/ws/text` 发新文本立即 `interrupt_round()` + `start_round()`。
- WS 断连 `interrupt_round(wait=True)`，确保后台线程彻底回收。

### 三级取消（`InterruptToken` 一票否决）
| 层 | 谁来 cancel | 怎么生效 |
|----|------------|----------|
| **L1 LLM 流**     | `token.cancelled` 检查 | LLM 生产线程 / async 消费者每次 yield/get 前 check，立即 return |
| **L2 渲染队列**   | `WORKER._loop` 取出 `req` 时 check | 跳过未开始的句子，标记 `error="cancelled"` |
| **L3 正在渲染**   | `engine.interruptible_render()` monkey-patch `tqdm.trange` | 渲染循环每帧 check，下一帧 raise `Interrupted` |

**关键设计**：不修改 Real3DPortrait 源码（monkey-patch `tqdm.trange`）。Real3DPortrait 渲染循环都是 `for i in tqdm.trange(num_frames, ...)`，包装后每帧多一次 `is_set()` 检查，开销可忽略。

### 前端
- 收到 `speech_start` 本地立即 `video.pause()`（不等服务端 interrupt 到达，**毫秒级体验**）。
- 收到 `interrupt` 清 playQueue + 字幕标"interrupted"。
- 老回合的 `video_chunk` 凭 `round_id != currentRoundId` 直接丢弃（防止网络在途的延迟到达）。

### AEC 与误触发
- 浏览器自带 `echoCancellation/noiseSuppression/autoGainControl` 已开，扬声器 → 麦克风的回声大部分被滤掉。
- 若仍有误触发（数字人自己的声音把自己打断），可加策略：
  - 数字人播放期间提高 VAD 阈值（`prob > 0.7`）；
  - 或要求用户使用耳机；
  - 或采用 WebRTC 服务端 AEC（重，留作后续）。

### 已知约束
- 老 task `interrupt_round(wait=False)` fire-and-forget 时仍会用 1 帧时间退出（被 Interrupted 抛出）。这一帧内**可能**新 req 已进 worker 队列，但 worker 会先把老 req 清空，最坏多等一帧。
- 多用户：当前全局单 `SESSION` + 单 `Memory`，多浏览器会串台。改为按 ws 连接的 `dict[sid]` 即可。

---

## 7. 调试

- 不连前端，单独测引擎：
  ```bash
  python -m realtime.engine --audio ref.wav --image face.png --bg_img bg.png --text "你好"
  ```
- 关 LLM 直 TTS：`$env:SYNCHRO_ENABLE_LLM = "0"`
- 切 Whisper 大小：`$env:WHISPER_MODEL = "tiny"`
- 观察 P4：服务端日志会打印 `interrupt round #N (user_speech)`；前端日志显示 `[interrupt #N]` + `[drop stale chunk #M]`。
