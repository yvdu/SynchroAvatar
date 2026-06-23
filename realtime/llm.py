# -*- coding: utf-8 -*-
"""
LLM 客户端 + 滑动窗口记忆 + 流式分句

设计要点
========
1. **Memory**：维护 [system, user, assistant, user, assistant, ...]，按"轮"裁剪，
   永远保留 system。用最简单的滑动窗口（max_turns），不做向量检索 / 摘要，
   避免引入额外延迟。需要更长记忆时可以替换实现。
2. **LLMClient**：兼容 OpenAI 接口（百度文心、Qwen API、OpenAI、本地 vLLM 都支持），
   开启 stream=True 逐 token 出文本。
3. **SentenceSplitter**：把 token 流按"中/英文句末标点 + 兜底长度"切句，
   一旦切到一句立刻 yield，给下游 TTS 用——这是把"首句延迟"打下来的关键。
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Iterable, Iterator, List, Optional

# 同原工程：去掉对 TTS 不友好的字符
_CLEAN_RE = re.compile(r"[*#@${}）（\[\]]")
# 句末标点（中英）
_SENT_END = set("。！？!?\n")
# 软切点（句中），用于兜底太长时
_SOFT_END = set("，,；;:：")


def clean_text(text: str) -> str:
    text = text.replace("\n", " ").replace("、", ",")
    text = _CLEAN_RE.sub("", text)
    return text.strip()


# ---------------------------------------------------------------------------
# Memory：滑动窗口
# ---------------------------------------------------------------------------
@dataclass
class Memory:
    system: str = (
        "你是一个亲切的语音助手。请用简洁、口语化的中文短句回答，"
        "每句不超过 30 字，禁止使用 Markdown、列表、表情符号、括号注释。"
    )
    max_turns: int = 8  # 保留最近 N 轮 user/assistant
    history: List[dict] = field(default_factory=list)

    def add_user(self, content: str) -> None:
        self.history.append({"role": "user", "content": content})
        self._trim()

    def add_assistant(self, content: str) -> None:
        self.history.append({"role": "assistant", "content": content})
        self._trim()

    def _trim(self) -> None:
        # 一轮 = 一条 user + 一条 assistant
        max_msgs = self.max_turns * 2
        if len(self.history) > max_msgs:
            self.history = self.history[-max_msgs:]

    def build(self) -> List[dict]:
        return [{"role": "system", "content": self.system}] + list(self.history)

    def reset(self) -> None:
        self.history.clear()


# ---------------------------------------------------------------------------
# LLM 客户端：OpenAI 兼容
# ---------------------------------------------------------------------------
class LLMClient:
    """OpenAI 兼容客户端，环境变量配置：

    - SYNCHRO_LLM_API_KEY
    - SYNCHRO_LLM_BASE_URL  （例如 https://api.openai.com/v1，或自己的网关地址）
    - SYNCHRO_LLM_MODEL     （例如 gpt-4o-mini / ernie-4.5-vl-28b-a3b / qwen2.5-7b-instruct）
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.6,
        max_tokens: int = 512,
    ):
        try:
            from openai import OpenAI  # type: ignore
        except ImportError as e:  # noqa: BLE001
            raise RuntimeError("pip install openai>=1.0 first") from e

        self.api_key = api_key or os.environ.get("SYNCHRO_LLM_API_KEY", "")
        self.base_url = base_url or os.environ.get(
            "SYNCHRO_LLM_BASE_URL", "https://api.openai.com/v1"
        )
        self.model = model or os.environ.get("SYNCHRO_LLM_MODEL", "gpt-4o-mini")
        self.temperature = temperature
        self.max_tokens = max_tokens
        if not self.api_key:
            raise RuntimeError("SYNCHRO_LLM_API_KEY not set")
        self._client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    def stream(self, messages: List[dict]) -> Iterator[str]:
        """逐 token 产出 delta 文本（不是整句）。"""
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            stream=True,
        )
        for chunk in resp:
            try:
                delta = chunk.choices[0].delta.content
            except (IndexError, AttributeError):
                delta = None
            if delta:
                yield delta


# ---------------------------------------------------------------------------
# 流式分句器
# ---------------------------------------------------------------------------
class SentenceSplitter:
    """把 token 流切成"可直接送 TTS 的短句"。

    规则：
      - 遇到 `。！？!?` 或换行：硬切。
      - 缓冲长度 >= soft_limit 且最近遇到 `，,；;:：`：在软切点切。
      - 缓冲长度 >= hard_limit：强切。
      - flush() 把剩余内容当作最后一句吐出。
    """

    def __init__(self, soft_limit: int = 18, hard_limit: int = 40):
        self.soft_limit = soft_limit
        self.hard_limit = hard_limit
        self._buf: List[str] = []

    def feed(self, delta: str) -> Iterator[str]:
        for ch in delta:
            self._buf.append(ch)
            if ch in _SENT_END:
                s = self._take()
                if s:
                    yield s
            elif len(self._buf) >= self.soft_limit and ch in _SOFT_END:
                s = self._take()
                if s:
                    yield s
            elif len(self._buf) >= self.hard_limit:
                s = self._take()
                if s:
                    yield s

    def flush(self) -> Optional[str]:
        s = self._take()
        return s or None

    def _take(self) -> str:
        raw = "".join(self._buf).strip()
        self._buf.clear()
        return clean_text(raw)


# ---------------------------------------------------------------------------
# 高层入口：text -> 流式 sentence
# ---------------------------------------------------------------------------
def stream_sentences(
    llm: LLMClient,
    memory: Memory,
    user_text: str,
    soft_limit: Optional[int] = None,
    hard_limit: Optional[int] = None,
) -> Iterator[str]:
    """喂入用户文本，流式产出"可送 TTS 的句子"，同时维护 memory。

    soft_limit / hard_limit 控制切句粒度：值越小，单句越短，
    第一段视频越早能渲染并播放（"边渲染边播放"更细），但句子更碎；
    可用环境变量 SYNCHRO_SOFT_LIMIT / SYNCHRO_HARD_LIMIT 覆盖默认值。
    """
    if soft_limit is None:
        soft_limit = int(os.environ.get("SYNCHRO_SOFT_LIMIT", "18"))
    if hard_limit is None:
        hard_limit = int(os.environ.get("SYNCHRO_HARD_LIMIT", "40"))
    memory.add_user(user_text)
    messages = memory.build()
    splitter = SentenceSplitter(soft_limit=soft_limit, hard_limit=hard_limit)
    full_answer: List[str] = []
    for delta in llm.stream(messages):
        full_answer.append(delta)
        for sent in splitter.feed(delta):
            yield sent
    tail = splitter.flush()
    if tail:
        yield tail
    memory.add_assistant("".join(full_answer).strip())
