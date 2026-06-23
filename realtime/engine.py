# -*- coding: utf-8 -*-
"""
SynchroAvatar 实时引擎（P0）

把 CosyVoice2 + Real3DPortrait 两个模型常驻显存，避免每次请求都拉起子进程。
本模块仅做"文本 -> wav -> mp4"的同步生成，由 server.py 负责通过 WebSocket 把视频块下发到前端。

设计要点：
1. 模型只在进程启动时加载一次。
2. 不再使用 os.system 调子进程，全部走类方法调用。
3. 原 Real3DPortrait/CosyVoice2 代码大量使用相对路径，所以用 chdir_ctx() 上下文管理器
   "进入->调用->退出"，避免污染主进程 cwd，也方便上层并发管理。
4. 推理本身是 GPU 密集任务，调用方应当串行化（server.py 用单 worker 队列）。
"""
from __future__ import annotations

import contextlib
import os
import re
import sys
import threading
import time
import uuid
import shutil
import subprocess
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
import torch
import torchaudio


# ---------------------------------------------------------------------------
# 打断机制（P4）
# ---------------------------------------------------------------------------
class Interrupted(Exception):
    """渲染/合成被外部打断（用户开口说新话）。"""


class InterruptToken:
    """三级打断令牌：

    - L1 LLM 流：调用方在 stream 循环里 check `cancelled` 提前 return。
    - L2 渲染队列：worker 取出任务时 check `cancelled` 直接丢弃。
    - L3 正在渲染那一句：通过 InterruptibleRange 上下文 monkey-patch tqdm.trange，
      渲染循环每帧检查一次，一旦置位就 raise Interrupted。

    线程安全：threading.Event 内部加锁，跨线程读写无问题。
    """

    __slots__ = ("_ev",)

    def __init__(self):
        self._ev = threading.Event()

    @property
    def cancelled(self) -> bool:
        return self._ev.is_set()

    def cancel(self) -> None:
        self._ev.set()

    def raise_if_cancelled(self) -> None:
        if self._ev.is_set():
            raise Interrupted("interrupted by user")


# 渲染线程当前持有的 token；monkey-patch 后的 tqdm.trange 包装器从这里读取。
# 必须用 thread-local，因为 worker 线程独立、不要影响其它线程里可能正在跑的 tqdm。
_TL = threading.local()


def _set_thread_token(token: Optional[InterruptToken]) -> None:
    _TL.token = token


def _get_thread_token() -> Optional[InterruptToken]:
    return getattr(_TL, "token", None)


@contextlib.contextmanager
def interruptible_render(token: Optional[InterruptToken]):
    """把"当前线程的 tqdm.trange"包装成"每次迭代检查 token 的 trange"。

    退出时恢复原始 tqdm.trange，避免影响其它代码。

    注意：monkey-patch 的是 **R3D 代码里 import 到的那个 trange**——
    Real3DPortrait 用的是 `import tqdm; tqdm.trange(...)`（即模块属性访问），
    替换 `tqdm.trange` 即可生效。如果将来该工程改成 `from tqdm import trange` 然后
    在自己的模块里直接用 trange，就需要 patch 那个模块的 trange 引用。
    """
    if token is None:
        yield
        return

    import tqdm  # noqa: WPS433

    original = tqdm.trange
    prev_token = _get_thread_token()
    _set_thread_token(token)

    def patched_trange(*args, **kwargs):
        it = original(*args, **kwargs)

        def gen():
            for x in it:
                # 每帧检查一次。trange 是个 tqdm 对象，本身不能被装饰；
                # 我们包成 generator，调用方写法 `for i in tqdm.trange(...)` 不受影响。
                t = _get_thread_token()
                if t is not None and t.cancelled:
                    # 关掉进度条避免控制台残留
                    try:
                        it.close()
                    except Exception:  # noqa: BLE001
                        pass
                    raise Interrupted("interrupted during render")
                yield x

        return gen()

    tqdm.trange = patched_trange
    try:
        yield
    finally:
        tqdm.trange = original
        _set_thread_token(prev_token)



# ---------------------------------------------------------------------------
# 路径常量（以 realtime/ 的父目录即仓库根作为基准）
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
COSY_DIR = REPO_ROOT / "CosyVoice-main"
R3D_DIR = REPO_ROOT / "Real3DPortrait-main"
COSY_MODEL_DIR = COSY_DIR / "pretrained_models" / "CosyVoice2-0.5B"

# 输出目录（不污染原工程目录）
OUT_ROOT = REPO_ROOT / "realtime" / "out"
OUT_ROOT.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# 工具：临时切换 cwd / sys.path
# ---------------------------------------------------------------------------
@contextlib.contextmanager
def chdir_ctx(path: Path):
    """临时切换工作目录；退出时恢复。"""
    old = os.getcwd()
    os.chdir(str(path))
    try:
        yield
    finally:
        os.chdir(old)


@contextlib.contextmanager
def syspath_ctx(*paths: Path):
    """临时在 sys.path 头部插入路径。"""
    inserted = []
    for p in paths:
        sp = str(p)
        if sp not in sys.path:
            sys.path.insert(0, sp)
            inserted.append(sp)
    try:
        yield
    finally:
        for sp in inserted:
            try:
                sys.path.remove(sp)
            except ValueError:
                pass


def _clean_text(text: str) -> str:
    """与原 LLM_API_example 中相同的清洗逻辑（去掉对 TTS 不友好的字符）。"""
    text = text.replace("\n", ",").replace("、", ",")
    text = re.sub(r"[*#@${}）（]", "", text)
    return text.strip()


# ---------------------------------------------------------------------------
# 引擎主体
# ---------------------------------------------------------------------------
class SynchroEngine:
    """常驻显存的数字人推理引擎。

    使用示例::

        engine = SynchroEngine()
        engine.load()
        engine.set_prompt_voice("prompt.wav", "你好，这是参考音频对应的文本")  # 可选
        mp4_path = engine.synthesize(
            text="你好，世界",
            src_image="path/to/face.png",
            bg_image="path/to/bg.png",
            position="center",
        )
    """

    # ---- 初始化 ----
    def __init__(
        self,
        cosy_model_dir: Optional[str] = None,
        a2m_ckpt: str = "checkpoints/240210_real3dportrait_orig/audio2secc_vae",
        torso_ckpt: str = "checkpoints/240210_real3dportrait_orig/secc2plane_torso_orig",
        head_ckpt: str = "",
        device: Optional[str] = None,
        load_jit: bool = False,
        load_trt: bool = False,
        fp16: bool = False,
    ):
        self.cosy_model_dir = cosy_model_dir or str(COSY_MODEL_DIR)
        self.a2m_ckpt = a2m_ckpt
        self.torso_ckpt = torso_ckpt
        self.head_ckpt = head_ckpt
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.load_jit = load_jit
        self.load_trt = load_trt
        self.fp16 = fp16

        self.cosyvoice = None  # CosyVoice2 实例
        self.r3d = None        # GeneFace2Infer 实例

        # 默认推理超参（沿用 real3d_infer_video.py main 的默认值）
        self.default_inp = {
            "a2m_ckpt": self.a2m_ckpt,
            "head_ckpt": self.head_ckpt,
            "torso_ckpt": self.torso_ckpt,
            "drv_pose_name": "data/raw/examples/May_5s_coeff_fit_mp.npy",
            "blink_mode": "period",
            "temperature": 0.2,
            "mouth_amp": 0.45,
            "out_mode": "final",
            "map_to_init_pose": "True",
            "head_torso_threshold": None,
            "seed": None,
            "min_face_area_percent": 0.2,
            "low_memory_usage": False,
            "position": "center",
        }

        # 当前的参考音色（CosyVoice2 zero-shot 需要 prompt 音频 + prompt 文本）
        self._prompt_speech_16k: Optional[torch.Tensor] = None
        self._prompt_text: str = ""

    # ---- 加载模型（只调用一次） ----
    def load(self):
        """加载 CosyVoice2 与 Real3DPortrait 权重，过程较慢。"""
        self._tune_cuda()
        if self.cosyvoice is None:
            self._load_cosyvoice()
        if self.r3d is None:
            self._load_real3d()
        return self

    @staticmethod
    def _tune_cuda():
        """开启对推理友好的 CUDA/cuDNN 加速开关。

        - cudnn.benchmark：每帧输入尺寸固定为 512x512，启用 autotuner 选最快卷积算法。
        - TF32：Ampere+(30/40 系)上对 matmul/conv 提速明显，对数字人画质几乎无感。
        这些都是进程级全局开关，只需设一次。
        """
        if not torch.cuda.is_available():
            return
        try:
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            print("[engine] cuda tuned: cudnn.benchmark=True, tf32=True")
        except Exception as exc:  # noqa: BLE001
            print(f"[engine] cuda tune skipped: {exc}")

    def _load_cosyvoice(self):
        print("[engine] loading CosyVoice2 ...")
        with syspath_ctx(COSY_DIR):
            from cosyvoice.cli.cosyvoice import CosyVoice2  # noqa: WPS433
            # CosyVoice2 的 yaml 内部用了相对路径资源，进入其目录加载更稳
            with chdir_ctx(COSY_DIR):
                self.cosyvoice = CosyVoice2(
                    self.cosy_model_dir,
                    load_jit=self.load_jit,
                    load_trt=self.load_trt,
                    fp16=self.fp16,
                )
        self.sample_rate = self.cosyvoice.sample_rate
        print(f"[engine] CosyVoice2 ready (sr={self.sample_rate}).")

    def _load_real3d(self):
        print("[engine] loading Real3DPortrait ...")
        with syspath_ctx(R3D_DIR), chdir_ctx(R3D_DIR):
            # 这里必须在 R3D_DIR 下导入，因为该文件内部用了相对 import
            from inference.real3d_infer_video import GeneFace2Infer  # noqa: WPS433
            inp = dict(self.default_inp)
            inp["src_image_name"] = "data/raw/examples/Macron.png"  # 占位
            self.r3d = GeneFace2Infer(
                audio2secc_dir=self.a2m_ckpt,
                head_model_dir=self.head_ckpt,
                torso_model_dir=self.torso_ckpt,
                device=self.device,
                inp=inp,
            )
            # 图像侧预处理 LRU 缓存条目数：多用户时建议 >= 并发不同人脸数
            self.r3d._static_cache_max = int(os.environ.get("SYNCHRO_R3D_IMG_CACHE", "4"))
        print("[engine] Real3DPortrait ready.")

    # ---- 设置参考音色（zero-shot 克隆） ----
    def set_prompt_voice(self, audio_path: str, prompt_text: str = ""):
        """设置参考音频，以后所有 TTS 都用这把声音。

        参数
        ----
        audio_path: 参考音频路径（任意采样率，会被重采样到 16k）
        prompt_text: 参考音频对应的文本；可留空（CosyVoice2 在 zero_shot 模式下需要它，
                     传空串实测可工作但效果略差）。
        """
        wav, sr = torchaudio.load(audio_path)
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        if sr != 16000:
            wav = torchaudio.transforms.Resample(sr, 16000)(wav)
        self._prompt_speech_16k = wav
        self._prompt_text = prompt_text or "这是参考音频。"
        print(f"[engine] prompt voice set: {audio_path} (sr->16k, shape={wav.shape})")

    @staticmethod
    def prepare_prompt_voice(audio_path: str, prompt_text: str = ""):
        """把参考音频读成 16k 单声道张量并返回 (tensor, text)，不写入共享状态。

        多用户场景下，每个会话各自持有自己的音色张量，推理时传给 synthesize，
        从而避免共享 self._prompt_speech_16k 造成的串音。
        """
        wav, sr = torchaudio.load(audio_path)
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        if sr != 16000:
            wav = torchaudio.transforms.Resample(sr, 16000)(wav)
        return wav, (prompt_text or "这是参考音频。")

    # ---- 文本 -> wav ----
    def tts(
        self,
        text: str,
        out_wav: Optional[str] = None,
        token: Optional[InterruptToken] = None,
        prompt_speech_16k: Optional[torch.Tensor] = None,
        prompt_text: Optional[str] = None,
    ) -> str:
        """文本合成语音，返回 wav 文件路径。

        prompt_speech_16k / prompt_text：可选的"本次调用使用的参考音色"。
        多用户场景下由调用方按会话传入，避免共享 self._prompt_speech_16k 串音；
        不传则回退到 set_prompt_voice() 设的全局音色（单会话/离线用法不受影响）。
        token：可选打断令牌。
        """
        assert self.cosyvoice is not None, "call load() first"
        speech = prompt_speech_16k if prompt_speech_16k is not None else self._prompt_speech_16k
        ptext = prompt_text if prompt_text is not None else self._prompt_text
        assert speech is not None, "call set_prompt_voice() first or pass prompt_speech_16k"

        text = _clean_text(text)
        if not text:
            raise ValueError("empty text")

        out_wav = out_wav or str(OUT_ROOT / f"tts_{uuid.uuid4().hex}.wav")
        with syspath_ctx(COSY_DIR), chdir_ctx(COSY_DIR):
            chunks = []
            for out in self.cosyvoice.inference_zero_shot(
                tts_text=text,
                prompt_text=ptext,
                prompt_speech_16k=speech,
                stream=False,
                speed=1.0,
            ):
                if token is not None:
                    token.raise_if_cancelled()
                chunks.append(out["tts_speech"])
            wav = torch.cat(chunks, dim=1).squeeze(0).cpu().numpy()
        sf.write(out_wav, wav, self.sample_rate)
        return out_wav


    # ---- wav + 图片 -> mp4 ----
    def synthesize_video(
        self,
        wav_path: str,
        src_image: str,
        bg_image: str = "",
        position: str = "center",
        out_mp4: Optional[str] = None,
        low_memory_usage: bool = True,
        token: Optional[InterruptToken] = None,
    ) -> str:
        """调用 Real3DPortrait 生成说话视频，返回 mp4 路径。

        low_memory_usage=True 走"逐帧写文件"分支。
        token：传入则启用渲染循环逐帧中断（monkey-patch tqdm.trange）。
        """
        assert self.r3d is not None, "call load() first"

        wav_abs = str(Path(wav_path).resolve())
        src_abs = str(Path(src_image).resolve())
        bg_abs = str(Path(bg_image).resolve()) if bg_image else ""

        inp = dict(self.default_inp)
        inp.update(
            drv_audio_name=wav_abs,
            src_image_name=src_abs,
            src_image_name0=src_abs,
            bg_image_name=bg_abs,
            position=position,
            out_name="",  # 让其内部按规则命名
            low_memory_usage=low_memory_usage,
        )

        if token is not None:
            token.raise_if_cancelled()

        with syspath_ctx(R3D_DIR), chdir_ctx(R3D_DIR), interruptible_render(token):
            out_fname = self.r3d.infer_once(inp)
            produced = Path(out_fname)
            if not produced.is_absolute():
                produced = R3D_DIR / produced

        out_mp4 = out_mp4 or str(OUT_ROOT / f"video_{uuid.uuid4().hex}.mp4")
        # 把内部产出物拷出来，避免被下一次推理覆盖
        if produced.exists():
            shutil.copyfile(str(produced), out_mp4)
        else:
            raise RuntimeError(f"Real3DPortrait did not produce output: {produced}")
        return out_mp4

    # ---- 端到端：text -> mp4 ----
    def synthesize(
        self,
        text: str,
        src_image: str,
        bg_image: str = "",
        position: str = "center",
        token: Optional[InterruptToken] = None,
        prompt_speech_16k: Optional[torch.Tensor] = None,
        prompt_text: Optional[str] = None,
    ) -> str:
        """端到端：文本 -> wav -> 视频，返回 mp4 路径。"""
        t0 = time.time()
        wav_path = self.tts(
            text, token=token, prompt_speech_16k=prompt_speech_16k, prompt_text=prompt_text
        )
        t1 = time.time()
        mp4_path = self.synthesize_video(
            wav_path, src_image, bg_image, position, token=token
        )
        t2 = time.time()
        print(f"[engine] tts {t1 - t0:.2f}s, video {t2 - t1:.2f}s, total {t2 - t0:.2f}s")
        return mp4_path

    # ---- 流式：一句一段 mp4 ----
    def synthesize_sentence(
        self,
        sentence: str,
        src_image: str,
        bg_image: str = "",
        position: str = "center",
        token: Optional[InterruptToken] = None,
        prompt_speech_16k: Optional[torch.Tensor] = None,
        prompt_text: Optional[str] = None,
    ) -> str:
        """流式管线中的最小单元：一句话 -> 一个短 mp4。支持 P4 打断。"""
        return self.synthesize(
            sentence,
            src_image,
            bg_image,
            position,
            token=token,
            prompt_speech_16k=prompt_speech_16k,
            prompt_text=prompt_text,
        )


# ---------------------------------------------------------------------------
# 调试入口
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", required=True, help="参考音色音频")
    parser.add_argument("--image", required=True, help="人脸图")
    parser.add_argument("--bg_img", default="", help="背景图")
    parser.add_argument("--text", required=True, help="要合成的文本")
    parser.add_argument("--prompt_text", default="", help="参考音频对应的文本（可空）")
    args = parser.parse_args()

    engine = SynchroEngine().load()
    engine.set_prompt_voice(args.audio, args.prompt_text)
    out = engine.synthesize(
        text=args.text,
        src_image=args.image,
        bg_image=args.bg_img,
    )
    print("output:", out)
