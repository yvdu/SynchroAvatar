# -*- coding: utf-8 -*-
"""
SynchroAvatar 实时服务（P0 + P1 + P2 + P3 + P4 + 多用户并发）

阶段
====
P0 文本 -> 视频
P1 麦克风 PCM -> silero-VAD + faster-whisper -> 文本
P2 用户文本 -> LLM(stream, 带 Memory) -> 流式分句 -> 多句 TTS+渲染
P3 引擎层启用 low_memory_usage=True，**每句产一个短 mp4 片段**，按 seq 顺序推给前端
P4 全双工打断（VAD speech_start -> 三级取消）
P5(本版) 多用户并发：每个浏览器会话独立 配置/音色/记忆/打断轮/ASR 状态，
   通过 sid（会话令牌）隔离。**GPU 渲染仍由单 worker 串行**（单卡只能串行），
   多用户表现为"各自独立会话 + 公平排队"。

启动::

    uvicorn realtime.server:app --host 0.0.0.0 --port 8000

环境变量
--------
SYNCHRO_ENABLE_LLM=1   是否启用 P2 LLM 链路（默认开），关掉则走 P0 直 TTS
SYNCHRO_LLM_API_KEY    LLM API key
SYNCHRO_LLM_BASE_URL   LLM base url，默认 https://api.openai.com/v1
SYNCHRO_LLM_MODEL      模型名，默认 gpt-4o-mini
WHISPER_MODEL          faster-whisper 模型名，默认 small
SYNCHRO_R3D_IMG_CACHE  Real3DPortrait 图像预处理缓存条目数（多用户建议 >= 并发人脸数）

会话与并发
----------
1) POST /api/config（multipart）上传 人脸图/参考音频(必填) + 背景图/参考文本/位置(选填)，
   返回 {"ok":true,"sid":"<token>"}。可在表单里带 sid 复用同一会话。
2) WS 连接需带 ?sid=<token>：/ws/text?sid=xxx、/ws/audio?sid=xxx。
3) 每个会话独立 Memory / 音色 / 配置 / 打断轮，互不串台。
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import queue
import shutil
import threading
import time
import uuid
from pathlib import Path
from typing import AsyncIterator, Dict, Optional

import numpy as np
from fastapi import FastAPI, UploadFile, File, Form, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from .engine import SynchroEngine, OUT_ROOT, REPO_ROOT
from .engine import InterruptToken, Interrupted

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("synchro.server")


# ---------------------------------------------------------------------------
# 全局：模型 + 渲染 worker（串行队列，避免 GPU 冲突）
# ---------------------------------------------------------------------------
WEB_DIR = Path(__file__).resolve().parent / "web"
UPLOAD_DIR = REPO_ROOT / "realtime" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


class Session:
    """单个浏览器会话的全部独立状态。

    多用户并发的核心：配置、音色张量、对话记忆都挂在这里，按 sid 隔离，
    彻底取代旧版的全局单 SESSION + 全局 _MEMORY。
    """

    def __init__(self, sid: str):
        self.sid = sid
        # 配置
        self.src_image: Optional[str] = None
        self.bg_image: str = ""
        self.position: str = "center"
        self.prompt_audio: Optional[str] = None
        self.prompt_text: str = ""
        # 该会话的参考音色张量（每会话独立，避免共享串音）
        self.prompt_speech_16k = None  # torch.Tensor
        self.voice_text: str = ""
        # 该会话的对话记忆（LLM 启用时）
        self.memory = None
        self.created = time.time()

    def configured(self) -> bool:
        return bool(self.src_image) and self.prompt_speech_16k is not None


# sid -> Session
SESSIONS: Dict[str, Session] = {}
_SESSIONS_LOCK = threading.Lock()


def _get_session(sid: Optional[str]) -> Optional[Session]:
    if not sid:
        return None
    with _SESSIONS_LOCK:
        return SESSIONS.get(sid)


# ---------------------------------------------------------------------------
# 渲染 worker：单线程串行（每个任务自带 会话参数 + InterruptToken）
# ---------------------------------------------------------------------------
class InferRequest:
    """一次"一句话 -> 一个 mp4"的任务，携带本会话的全部渲染参数。

    cancelled: 任务被丢弃（不再渲染、也不再下发）。
    若任务正在渲染中被打断，会在 engine 内 raise Interrupted，
    worker 捕获后设置 error="interrupted"。
    """

    def __init__(self, text: str, token: InterruptToken, session: Session):
        self.text = text
        self.token = token
        # 渲染期间会话配置可能被改，这里快照下来，保证一致性
        self.src_image = session.src_image
        self.bg_image = session.bg_image
        self.position = session.position
        self.prompt_speech_16k = session.prompt_speech_16k
        self.prompt_text = session.voice_text
        self.result: Optional[str] = None
        self.error: Optional[str] = None
        self.done = threading.Event()

    @property
    def cancelled(self) -> bool:
        return self.token.cancelled


class InferWorker:
    def __init__(self, engine: SynchroEngine):
        self.engine = engine
        self.q: "queue.Queue[InferRequest]" = queue.Queue()
        self.thread = threading.Thread(target=self._loop, name="infer-worker", daemon=True)

    def start(self):
        self.thread.start()

    def submit(self, text: str, token: InterruptToken, session: Session) -> InferRequest:
        req = InferRequest(text, token, session)
        self.q.put(req)
        return req

    def _loop(self):
        while True:
            req = self.q.get()
            try:
                # L2 打断：取出来但 token 已 cancelled，跳过渲染
                if req.token.cancelled:
                    req.error = "cancelled"
                    continue
                if not req.src_image:
                    raise RuntimeError("session not configured: missing src_image")
                if req.prompt_speech_16k is None:
                    raise RuntimeError("session not configured: missing prompt voice")
                out = self.engine.synthesize_sentence(
                    sentence=req.text,
                    src_image=req.src_image,
                    bg_image=req.bg_image,
                    position=req.position,
                    token=req.token,
                    prompt_speech_16k=req.prompt_speech_16k,
                    prompt_text=req.prompt_text,
                )
                req.result = out
            except Interrupted:
                # L3 打断：渲染中被强制结束（tqdm.trange 抛 Interrupted）
                log.info("infer interrupted: %r", req.text)
                req.error = "interrupted"
            except Exception as exc:  # noqa: BLE001
                log.exception("infer failed")
                req.error = str(exc)
            finally:
                req.done.set()


ENGINE: Optional[SynchroEngine] = None
WORKER: Optional[InferWorker] = None


# ---------------------------------------------------------------------------
# LLM 模块（按需加载）：客户端共享（无状态），Memory 按会话隔离
# ---------------------------------------------------------------------------
def _llm_enabled() -> bool:
    return os.environ.get("SYNCHRO_ENABLE_LLM", "1") == "1" and bool(
        os.environ.get("SYNCHRO_LLM_API_KEY")
    )


_LLM_CLIENT = None


def _get_llm_client():
    global _LLM_CLIENT
    if _LLM_CLIENT is None:
        from .llm import LLMClient

        _LLM_CLIENT = LLMClient()
    return _LLM_CLIENT


def _ensure_memory(session: Session):
    if session.memory is None:
        from .llm import Memory

        session.memory = Memory()
    return session.memory


# ---------------------------------------------------------------------------
# FastAPI
# ---------------------------------------------------------------------------
app = FastAPI(title="SynchroAvatar Realtime")


@app.on_event("startup")
def _startup():
    global ENGINE, WORKER
    log.info("loading models ...")
    # 低显存可设 SYNCHRO_COSY_FP16=1 让 CosyVoice2 用半精度，省约 1GB 显存
    cosy_fp16 = os.environ.get("SYNCHRO_COSY_FP16", "0") == "1"
    ENGINE = SynchroEngine(fp16=cosy_fp16).load()
    WORKER = InferWorker(ENGINE)
    WORKER.start()
    if _llm_enabled():
        try:
            _get_llm_client()
            log.info("LLM enabled (model=%s).", os.environ.get("SYNCHRO_LLM_MODEL", "gpt-4o-mini"))
        except Exception as exc:  # noqa: BLE001
            log.warning("LLM init failed, fallback to direct-TTS: %s", exc)
    else:
        log.info("LLM disabled, /ws/text and /ws/audio will use direct-TTS.")
    log.info("ready.")


@app.get("/", response_class=HTMLResponse)
def index():
    f = WEB_DIR / "index.html"
    if not f.exists():
        return HTMLResponse("<h1>SynchroAvatar</h1><p>web/index.html missing</p>")
    return HTMLResponse(f.read_text(encoding="utf-8"))


if WEB_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")


@app.get("/api/files/{name}")
def get_file(name: str):
    if "/" in name or "\\" in name or ".." in name:
        raise HTTPException(400, "invalid name")
    fp = OUT_ROOT / name
    if not fp.exists():
        raise HTTPException(404, "not found")
    return FileResponse(str(fp))


@app.post("/api/config")
async def api_config(
    src_image: UploadFile = File(...),
    prompt_audio: UploadFile = File(...),
    prompt_text: str = Form(""),
    bg_image: Optional[UploadFile] = File(None),
    position: str = Form("center"),
    sid: str = Form(""),
):
    assert ENGINE is not None

    def _save(uf: UploadFile, sub: str) -> str:
        ext = Path(uf.filename or "").suffix or ""
        dst = UPLOAD_DIR / f"{sub}_{uuid.uuid4().hex}{ext}"
        with dst.open("wb") as f:
            shutil.copyfileobj(uf.file, f)
        return str(dst)

    # 复用已有 sid 或新建会话
    session = _get_session(sid)
    if session is None:
        sid = uuid.uuid4().hex
        session = Session(sid)
        with _SESSIONS_LOCK:
            SESSIONS[sid] = session

    session.src_image = _save(src_image, "face")
    session.prompt_audio = _save(prompt_audio, "prompt")
    session.prompt_text = prompt_text or ""
    session.position = position or "center"
    session.bg_image = _save(bg_image, "bg") if bg_image is not None else ""

    # 每会话独立音色张量（不写共享引擎状态）
    speech, vtext = ENGINE.prepare_prompt_voice(session.prompt_audio, session.prompt_text)
    session.prompt_speech_16k = speech
    session.voice_text = vtext
    return {"ok": True, "sid": sid}


@app.post("/api/reset_memory")
def reset_memory(sid: str = Form("")):
    """清空指定会话的对话历史。"""
    session = _get_session(sid)
    if session is not None and session.memory is not None:
        session.memory.reset()
    return {"ok": True}


# ---------------------------------------------------------------------------
# 核心调度：用户文本 -> 句子流 -> 渲染流 -> WS 下发
# ---------------------------------------------------------------------------
async def _await_event(ev: threading.Event):
    """异步等待 threading.Event，不阻塞 event loop。"""
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, ev.wait)


async def _iter_sentences(
    session: Session,
    user_text: str,
    token: Optional[InterruptToken] = None,
) -> AsyncIterator[str]:
    """LLM 启用时：流式产句；否则把整段文本当作一句。

    token：可选打断令牌。生产者线程每拉到下一句前检查 cancelled；
    异步消费侧每次 q.get 后也检查，确保 L1 打断及时生效。
    """
    if not _llm_enabled():
        yield user_text
        return

    try:
        client = _get_llm_client()
        memory = _ensure_memory(session)
    except Exception as exc:  # noqa: BLE001
        log.warning("LLM unavailable, fallback: %s", exc)
        yield user_text
        return

    from .llm import stream_sentences

    q: "queue.Queue[Optional[str]]" = queue.Queue(maxsize=64)

    def _producer():
        try:
            for s in stream_sentences(client, memory, user_text):
                if token is not None and token.cancelled:
                    return  # L1 打断
                q.put(s)
        except Exception as exc:  # noqa: BLE001
            log.exception("LLM stream failed")
            q.put(f"[LLM error] {exc}")
        finally:
            q.put(None)  # sentinel

    threading.Thread(target=_producer, name="llm-stream", daemon=True).start()

    loop = asyncio.get_running_loop()
    while True:
        item = await loop.run_in_executor(None, q.get)
        if item is None:
            return
        if token is not None and token.cancelled:
            return
        yield item


class RoundState:
    """每个 WS 连接维护的"当前回答轮"状态。

    一轮 = 一次"用户输入 -> LLM/TTS/渲染 -> WS 推送"全过程。
    新一轮开始前必须打断上一轮，避免回答互相覆盖。
    """

    def __init__(self):
        self.token: Optional[InterruptToken] = None
        self.task: Optional[asyncio.Task] = None
        self.round_id: int = 0

    def is_active(self) -> bool:
        return self.task is not None and not self.task.done()


async def interrupt_round(
    ws: WebSocket,
    state: RoundState,
    reason: str = "user_speech",
    wait: bool = False,
):
    """三级打断当前回答轮，并通知前端立即停播 + 清队列。

    L1 token.cancel() → LLM 生产线程下次 check 时退出
    L2 token.cancel() → WORKER._loop 取出未开始的 req 时跳过（标记 error=cancelled）
    L3 token.cancel() → engine 中 monkey-patched tqdm.trange 在下一帧 raise Interrupted
    """
    if not state.is_active() or state.token is None:
        return
    log.info("interrupt round #%d (%s, wait=%s)", state.round_id, reason, wait)
    state.token.cancel()
    try:
        await ws.send_json(
            {"type": "interrupt", "round_id": state.round_id, "reason": reason}
        )
    except Exception:  # noqa: BLE001
        pass
    task = state.task
    if task is not None and not task.done():
        task.cancel()
        if wait:
            try:
                await task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass


async def pipeline_run(
    ws: WebSocket,
    session: Session,
    user_text: str,
    state: RoundState,
):
    """完整流水线：用户文本 -> LLM 流式分句 -> 串行 TTS+渲染 -> WS 推送 video_chunk。

    全程绑定 state.token。任意时刻 token.cancel() 都会让流水线优雅退出。
    """
    if WORKER is None or ENGINE is None:
        await ws.send_json({"type": "error", "msg": "engine not ready"})
        return
    if not session.configured():
        await ws.send_json({"type": "error", "msg": "session not configured"})
        return

    token = state.token
    assert token is not None
    seq = 0
    pending: "list[tuple[int, InferRequest]]" = []

    async def drain(force_all: bool = False):
        while pending:
            s, req = pending[0]
            if not force_all and not req.done.is_set():
                return
            await _await_event(req.done)
            pending.pop(0)
            # 被打断/取消的 chunk 静默丢弃，不再下发也不报 error
            if req.error in ("cancelled", "interrupted"):
                continue
            if req.error:
                await ws.send_json({"type": "error", "msg": req.error, "seq": s})
                continue
            if token.cancelled:
                continue
            mp4 = Path(req.result or "")
            if not mp4.exists():
                await ws.send_json({"type": "error", "msg": "no output", "seq": s})
                continue
            await ws.send_json(
                {
                    "type": "video_chunk",
                    "url": f"/api/files/{mp4.name}",
                    "seq": s,
                    "is_last": False,
                    "round_id": state.round_id,
                }
            )

    try:
        async for sentence in _iter_sentences(session, user_text, token=token):
            if token.cancelled:
                break
            sentence = sentence.strip()
            if not sentence:
                continue
            seq += 1
            await ws.send_json(
                {
                    "type": "llm_sentence",
                    "text": sentence,
                    "seq": seq,
                    "round_id": state.round_id,
                }
            )
            req = WORKER.submit(sentence, token, session)
            pending.append((seq, req))
            await drain(force_all=False)

        await drain(force_all=True)
        if not token.cancelled:
            await ws.send_json({"type": "done", "round_id": state.round_id})
    except asyncio.CancelledError:
        log.info("pipeline_run cancelled (round #%d)", state.round_id)
        raise
    finally:
        pending.clear()


def start_round(ws: WebSocket, session: Session, state: RoundState, user_text: str) -> asyncio.Task:
    """开新一轮：生成新 token + 起新 task。调用方应在此之前已 interrupt_round。"""
    state.round_id += 1
    state.token = InterruptToken()
    state.task = asyncio.create_task(
        _pipeline_with_round_start(ws, session, user_text, state),
        name=f"round-{state.round_id}",
    )
    return state.task


async def _pipeline_with_round_start(ws: WebSocket, session: Session, user_text: str, state: RoundState):
    """先发 round_start，再跑 pipeline_run。被 cancel 时让 CancelledError 自然冒泡。"""
    try:
        await ws.send_json({"type": "round_start", "round_id": state.round_id})
    except Exception:  # noqa: BLE001
        return
    await pipeline_run(ws, session, user_text, state)


# ---------------------------------------------------------------------------
# WS：/ws/text
# ---------------------------------------------------------------------------
@app.websocket("/ws/text")
async def ws_text(ws: WebSocket):
    await ws.accept()
    sid = ws.query_params.get("sid", "")
    session = _get_session(sid)
    if session is None:
        await ws.send_json({"type": "error", "msg": "invalid or missing sid; POST /api/config first"})
        await ws.close()
        return

    state = RoundState()
    try:
        while True:
            msg = await ws.receive_text()
            try:
                data = json.loads(msg)
            except json.JSONDecodeError:
                await ws.send_json({"type": "error", "msg": "invalid json"})
                continue
            text = (data or {}).get("text", "").strip()
            if not text:
                await ws.send_json({"type": "error", "msg": "empty text"})
                continue
            # P4：用户发新文本 = 显式打断上一轮
            await interrupt_round(ws, state, reason="new_input")
            await ws.send_json({"type": "accepted", "text": text})
            start_round(ws, session, state, text)
    except WebSocketDisconnect:
        await interrupt_round(ws, state, reason="disconnect", wait=True)
        return


# ---------------------------------------------------------------------------
# WS：/ws/audio
# ---------------------------------------------------------------------------
class AsrModels:
    """共享的 ASR 模型（silero-VAD + faster-whisper），全进程只加载一份。

    模型本身无会话状态，可被多个连接安全并发只读调用；
    每连接的滑动缓冲状态放在 AsrStream 里。
    """

    SR = 16000
    FRAME_MS = 20
    FRAME_SAMPLES = SR * FRAME_MS // 1000  # 320

    def __init__(self, whisper_model: str = "small", device: str = "auto"):
        import torch  # noqa: WPS433

        self.device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")

        log.info("loading silero-vad ...")
        try:
            from silero_vad import load_silero_vad  # type: ignore

            self.vad = load_silero_vad()
        except Exception:  # noqa: BLE001
            self.vad, _ = torch.hub.load(
                repo_or_dir="snakers4/silero-vad",
                model="silero_vad",
                trust_repo=True,
            )
        self.vad.to(self.device).eval()

        log.info("loading faster-whisper (%s) ...", whisper_model)
        from faster_whisper import WhisperModel  # type: ignore

        compute = "float16" if self.device == "cuda" else "int8"
        self.asr = WhisperModel(whisper_model, device=self.device, compute_type=compute)
        log.info("ASR ready.")

    def is_speech(self, frame_int16: np.ndarray) -> bool:
        import torch  # noqa: WPS433

        with torch.no_grad():
            x = torch.from_numpy(frame_int16.astype(np.float32) / 32768.0).to(self.device)
            prob = self.vad(x, self.SR).item()
        return prob > 0.5

    def transcribe(self, pcm_int16: np.ndarray) -> str:
        audio = pcm_int16.astype(np.float32) / 32768.0
        segments, _info = self.asr.transcribe(
            audio,
            language="zh",
            beam_size=1,
            vad_filter=False,
        )
        return "".join(s.text for s in segments).strip()


class AsrStream:
    """每个 /ws/audio 连接独立的 VAD 切句状态机（复用共享 AsrModels）。"""

    FRAME_MS = AsrModels.FRAME_MS
    FRAME_SAMPLES = AsrModels.FRAME_SAMPLES
    SR = AsrModels.SR
    SILENCE_TAIL_MS = 600
    MIN_SPEECH_MS = 250
    MAX_SEG_MS = 15000

    def __init__(self, models: AsrModels):
        self.m = models
        self._buf = np.zeros(0, dtype=np.int16)
        self._silence_ms = 0
        self._in_speech = False

    def feed(self, frame_int16: np.ndarray):
        events = []
        if len(frame_int16) != self.FRAME_SAMPLES:
            return events

        speech = self.m.is_speech(frame_int16)
        if speech:
            if not self._in_speech:
                self._in_speech = True
                events.append({"event": "speech_start"})
            self._buf = np.concatenate([self._buf, frame_int16])
            self._silence_ms = 0
        else:
            if self._in_speech:
                self._silence_ms += self.FRAME_MS
                self._buf = np.concatenate([self._buf, frame_int16])

        seg_ms = len(self._buf) * 1000 // self.SR
        end = False
        if self._in_speech and self._silence_ms >= self.SILENCE_TAIL_MS:
            end = True
        elif self._in_speech and seg_ms >= self.MAX_SEG_MS:
            end = True

        if end:
            if seg_ms >= self.MIN_SPEECH_MS:
                text = self.m.transcribe(self._buf)
                if text:
                    events.append({"event": "final", "text": text})
            self._buf = np.zeros(0, dtype=np.int16)
            self._silence_ms = 0
            self._in_speech = False
        return events


_ASR_MODELS: Optional[AsrModels] = None


def _get_asr_models() -> AsrModels:
    global _ASR_MODELS
    if _ASR_MODELS is None:
        model = os.environ.get("WHISPER_MODEL", "small")
        # 低显存(如 6GB)建议 WHISPER_DEVICE=cpu，把 Whisper 挪到 CPU 给渲染让出显存
        device = os.environ.get("WHISPER_DEVICE", "auto")
        _ASR_MODELS = AsrModels(whisper_model=model, device=device)
    return _ASR_MODELS


@app.websocket("/ws/audio")
async def ws_audio(ws: WebSocket):
    await ws.accept()
    sid = ws.query_params.get("sid", "")
    session = _get_session(sid)
    if session is None:
        await ws.send_json({"type": "error", "msg": "invalid or missing sid; POST /api/config first"})
        await ws.close()
        return

    try:
        models = _get_asr_models()
    except Exception as exc:  # noqa: BLE001
        await ws.send_json({"type": "error", "msg": f"ASR init failed: {exc}"})
        await ws.close()
        return

    asr = AsrStream(models)
    loop = asyncio.get_running_loop()
    state = RoundState()

    try:
        while True:
            raw = await ws.receive_bytes()
            arr = np.frombuffer(raw, dtype=np.int16)
            for off in range(0, len(arr), AsrStream.FRAME_SAMPLES):
                frame = arr[off : off + AsrStream.FRAME_SAMPLES]
                if len(frame) != AsrStream.FRAME_SAMPLES:
                    break
                events = await loop.run_in_executor(None, asr.feed, frame)
                for ev in events:
                    if ev["event"] == "speech_start":
                        await ws.send_json({"type": "speech_start"})
                        # P4 全双工核心：用户一开口就打断正在说话的数字人
                        if state.is_active():
                            await interrupt_round(ws, state, reason="user_speech")
                    elif ev["event"] == "final":
                        await ws.send_json({"type": "final", "text": ev["text"]})
                        if state.is_active():
                            await interrupt_round(ws, state, reason="new_input")
                        start_round(ws, session, state, ev["text"])
    except WebSocketDisconnect:
        await interrupt_round(ws, state, reason="disconnect", wait=True)
        return
