# -*- coding: utf-8 -*-
"""
SynchroAvatar 预训练权重一键下载脚本

覆盖：
  1. CosyVoice2-0.5B          (ModelScope，~1.6 GB) — TTS 主模型
  2. CosyVoice-ttsfrd         (ModelScope，~600 MB，可选)
  3. Real3DPortrait BFM       (Google Drive，8 个文件) — 3DMM
  4. Real3DPortrait ckpts     (Google Drive，~1 GB) — audio2secc + secc2plane_torso
  5. Real3DPortrait pretrained_ckpts (Google Drive，~50 MB) — mit_b0
  6. MediaPipe face_landmarker (官方 CDN)
  7. MediaPipe selfie segmenter (官方 CDN)
  8. HuBERT large ls960-ft    (HuggingFace) — 音频特征
  9. faster-whisper small     (HuggingFace) — ASR
 10. silero-vad               (校验 / 触发首次下载)

用法
----
# 全量下载（默认跳过已存在）
python -m realtime.download_models

# 只下载某几个
python -m realtime.download_models --only cosyvoice,bfm,r3d_ckpt

# 强制重下
python -m realtime.download_models --force

# 列出所有任务
python -m realtime.download_models --list

# 国内镜像（推荐）
# PowerShell:  $env:HF_ENDPOINT = "https://hf-mirror.com"
# bash:        export HF_ENDPOINT=https://hf-mirror.com
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
import zipfile
from pathlib import Path
from typing import Callable, Dict, List, Optional


REPO_ROOT = Path(__file__).resolve().parent.parent
COSY_DIR = REPO_ROOT / "CosyVoice-main"
R3D_DIR = REPO_ROOT / "Real3DPortrait-main"


def log(msg: str) -> None:
    print(f"[download] {msg}", flush=True)


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def file_ok(p: Path, min_bytes: int = 1024) -> bool:
    return p.exists() and p.is_file() and p.stat().st_size >= min_bytes


def dir_ok(p: Path) -> bool:
    return p.exists() and p.is_dir() and any(p.iterdir())


def pip_install(pkg: str) -> None:
    log(f"pip install {pkg}")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])


def lazy_import(modname: str, pip_name: Optional[str] = None):
    try:
        return __import__(modname)
    except ImportError:
        pip_install(pip_name or modname)
        return __import__(modname)


# ---------------------------------------------------------------------------
# 通用下载器
# ---------------------------------------------------------------------------
def gdown_file(url_or_id: str, dst: Path, *, force: bool, min_bytes: int = 1024) -> None:
    """从 Google Drive 拉单个文件。url_or_id 可以是 https://... 或纯 file id。"""
    if not force and file_ok(dst, min_bytes=min_bytes):
        log(f"skip (exists): {dst.name}")
        return
    gdown = lazy_import("gdown")
    ensure_dir(dst.parent)
    url = url_or_id if url_or_id.startswith("http") else f"https://drive.google.com/uc?id={url_or_id}"
    log(f"gdown -> {dst}")
    gdown.download(url=url, output=str(dst), quiet=False, fuzzy=True)
    if not file_ok(dst, min_bytes=min_bytes):
        raise RuntimeError(
            f"gdown 似乎失败，文件过小：{dst} "
            f"(size={dst.stat().st_size if dst.exists() else 0}). "
            f"Google Drive 在国内常被墙，请挂代理；或手动下载后放到对应目录。"
        )


def unzip(zip_path: Path, dst_dir: Path) -> None:
    log(f"unzip {zip_path.name} -> {dst_dir}")
    ensure_dir(dst_dir)
    with zipfile.ZipFile(str(zip_path)) as zf:
        zf.extractall(str(dst_dir))


def http_download(url: str, dst: Path, *, force: bool, min_bytes: int = 1024) -> None:
    if not force and file_ok(dst, min_bytes=min_bytes):
        log(f"skip (exists): {dst.name}")
        return
    requests = lazy_import("requests")
    ensure_dir(dst.parent)
    log(f"GET {url}")
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        tmp = dst.with_suffix(dst.suffix + ".part")
        total = int(r.headers.get("content-length", 0))
        got, last = 0, time.time()
        with tmp.open("wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 16):
                if chunk:
                    f.write(chunk)
                    got += len(chunk)
                    if total and time.time() - last > 1.0:
                        pct = got * 100 / total
                        print(
                            f"\r  {got / 1e6:.1f}/{total / 1e6:.1f} MB ({pct:.1f}%)",
                            end="",
                            flush=True,
                        )
                        last = time.time()
        print()
        tmp.replace(dst)


# ---------------------------------------------------------------------------
# 任务定义
# ---------------------------------------------------------------------------
# Google Drive 文件 id 全部来自 Real3DPortrait 官方 Colab：
# Real3DPortrait-main/inference/real3dportrait_demo.ipynb
BFM_FILES = {
    # 文件名: gdrive id
    "01_MorphableModel.mat":           "1SPM3IHsyNAaVMwqZZGV6QVaV7I2Hly0v",
    "BFM_exp_idx.mat":                 "1MSldX9UChKEb3AXLVTPzZQcsbGD4VmGF",
    "BFM_front_idx.mat":               "180ciTvm16peWrcpl4DOekT9eUQ-lJfMU",
    "BFM_model_front.mat":             "1KX9MyGueFB3M-X0Ss152x_johyTXHTfU",
    "Exp_Pca.bin":                     "19-NyZn_I0_mkF-F5GPyFMwQJ_-WecZIL",
    "facemodel_info.mat":              "11ouQ7Wr2I-JKStp2Fd1afedmWeuifhof",
    "index_mp468_from_mesh35709.npy":  "18ICIvQoKX-7feYWP61RbpppzDuYTptCq",
    "std_exp.txt":                     "1VktuY46m0v_n_d4nvOupauJkK4LF6mHE",
}

R3D_CKPT_ZIP_ID = "1gSUIw2AkkKnlLJnNfS2FCqtaVw9tw3QF"     # 240210_real3dportrait_orig.zip
R3D_PRETRAINED_ZIP_ID = "1gz8A6xestHp__GbZT5qozb43YaybRJhZ"  # pretrained_ckpts.zip

MP_FACE_LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/latest/face_landmarker.task"
)
MP_SEGMENTER_URL = (
    "https://storage.googleapis.com/mediapipe-models/image_segmenter/"
    "selfie_multiclass_256x256/float32/latest/selfie_multiclass_256x256.tflite"
)


def _ms_snapshot(repo_id: str, dst: Path) -> None:
    """ModelScope 的 snapshot_download 在不同版本里参数名不同，做一个兼容包装。"""
    from modelscope import snapshot_download  # type: ignore  # noqa: WPS433

    ensure_dir(dst.parent)
    try:
        # 新版本：直接落到指定 local_dir
        snapshot_download(repo_id, local_dir=str(dst))
        return
    except TypeError:
        pass
    # 老版本：只支持 cache_dir，模型会落到 cache_dir/<repo_id>/，再 mv 过去
    import shutil  # noqa: WPS433

    cache_dir = dst.parent / "_ms_cache"
    out = snapshot_download(repo_id, cache_dir=str(cache_dir))
    out_path = Path(out)
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(str(out_path), str(dst))


# --- CosyVoice2 ---
def task_cosyvoice(force: bool) -> None:
    dst = COSY_DIR / "pretrained_models" / "CosyVoice2-0.5B"
    if not force and (dst / "cosyvoice.yaml").exists():
        log(f"skip (exists): {dst}")
        return
    lazy_import("modelscope")
    log(f"ModelScope snapshot iic/CosyVoice2-0.5B -> {dst}")
    _ms_snapshot("iic/CosyVoice2-0.5B", dst)


def task_cosyvoice_ttsfrd(force: bool) -> None:
    dst = COSY_DIR / "pretrained_models" / "CosyVoice-ttsfrd"
    if not force and dir_ok(dst):
        log(f"skip (exists): {dst}")
        return
    lazy_import("modelscope")
    log(f"ModelScope snapshot iic/CosyVoice-ttsfrd -> {dst}")
    _ms_snapshot("iic/CosyVoice-ttsfrd", dst)


# --- Real3DPortrait 3DMM BFM ---
def task_bfm(force: bool) -> None:
    dst_dir = R3D_DIR / "deep_3drecon" / "BFM"
    ensure_dir(dst_dir)
    for name, gid in BFM_FILES.items():
        gdown_file(gid, dst_dir / name, force=force, min_bytes=1024)
    log(f"BFM ready at {dst_dir}")


# --- Real3DPortrait 主 ckpt（audio2secc + secc2plane_torso） ---
def task_r3d_ckpt(force: bool) -> None:
    dst_dir = R3D_DIR / "checkpoints"
    marker = dst_dir / "240210_real3dportrait_orig" / "audio2secc_vae" / "config.yaml"
    if not force and marker.exists():
        log(f"skip (exists): {marker.parent.parent}")
        return
    ensure_dir(dst_dir)
    zip_path = dst_dir / "240210_real3dportrait_orig.zip"
    gdown_file(R3D_CKPT_ZIP_ID, zip_path, force=force, min_bytes=10 * 1024 * 1024)
    unzip(zip_path, dst_dir)
    zip_path.unlink(missing_ok=True)
    if not marker.exists():
        raise RuntimeError(f"解压后未找到 {marker}，请人工检查 zip 内部结构")
    log(f"Real3DPortrait main ckpt ready at {marker.parent.parent}")


# --- Real3DPortrait pretrained_ckpts（mit_b0 等） ---
def task_r3d_pretrained(force: bool) -> None:
    dst_dir = R3D_DIR / "checkpoints"
    marker = dst_dir / "pretrained_ckpts" / "mit_b0.pth"
    if not force and marker.exists():
        log(f"skip (exists): {marker}")
        return
    ensure_dir(dst_dir)
    zip_path = dst_dir / "pretrained_ckpts.zip"
    gdown_file(R3D_PRETRAINED_ZIP_ID, zip_path, force=force, min_bytes=1024 * 1024)
    unzip(zip_path, dst_dir)
    zip_path.unlink(missing_ok=True)
    if not marker.exists():
        raise RuntimeError(f"解压后未找到 {marker}")
    log(f"pretrained_ckpts ready at {marker.parent}")


# --- MediaPipe ---
def task_mp_landmarker(force: bool) -> None:
    dst = R3D_DIR / "data_gen" / "utils" / "mp_feature_extractors" / "face_landmarker.task"
    http_download(MP_FACE_LANDMARKER_URL, dst, force=force, min_bytes=1024 * 100)


def task_mp_segmenter(force: bool) -> None:
    dst = (
        R3D_DIR
        / "data_gen"
        / "utils"
        / "mp_feature_extractors"
        / "selfie_multiclass_256x256.tflite"
    )
    http_download(MP_SEGMENTER_URL, dst, force=force, min_bytes=1024 * 100)


# --- HuBERT large（HuggingFace） ---
def task_hubert(force: bool) -> None:
    """放在 Real3DPortrait-main/tf_model（与 extract_hubert.py 中的相对路径一致）。

    如果不预下，运行时 transformers 会自动从 facebook/hubert-large-ls960-ft 拉到
    HF 缓存目录；预下放到工程内可以避免每次 transformers 缓存重建。
    """
    dst = R3D_DIR / "tf_model"
    if not force and (dst / "config.json").exists():
        log(f"skip (exists): {dst}")
        return
    lazy_import("huggingface_hub")
    from huggingface_hub import snapshot_download  # type: ignore  # noqa: WPS433

    ensure_dir(dst.parent)
    log(f"HF snapshot facebook/hubert-large-ls960-ft -> {dst}")
    try:
        snapshot_download(
            repo_id="facebook/hubert-large-ls960-ft",
            local_dir=str(dst),
            local_dir_use_symlinks=False,
        )
    except TypeError:
        # 新版 huggingface_hub 移除了 local_dir_use_symlinks
        snapshot_download(
            repo_id="facebook/hubert-large-ls960-ft",
            local_dir=str(dst),
        )


# --- faster-whisper（HuggingFace） ---
def task_whisper(force: bool) -> None:
    """faster-whisper 自带模型按需下载，但预下放在 HF 缓存里，启动服务更快。"""
    model_name = os.environ.get("WHISPER_MODEL", "small")
    repo_id = f"Systran/faster-whisper-{model_name}"
    lazy_import("huggingface_hub")
    from huggingface_hub import snapshot_download  # type: ignore  # noqa: WPS433

    log(f"HF snapshot {repo_id} (cache mode)")
    # 走 HF 缓存目录（faster-whisper 自己加载时会找到）
    snapshot_download(repo_id=repo_id)


# --- silero-vad ---
def task_silero(force: bool) -> None:
    """silero-vad 的 pip 包 silero_vad 自带权重；如果没装则 pip 装。

    fallback：torch.hub 也行，但需要联网；这里只触发首次加载以验证可用。
    """
    try:
        from silero_vad import load_silero_vad  # type: ignore
    except ImportError:
        pip_install("silero-vad")
        from silero_vad import load_silero_vad  # type: ignore  # noqa: WPS433
    log("loading silero-vad to verify ...")
    _ = load_silero_vad()
    log("silero-vad ok")


# ---------------------------------------------------------------------------
# 任务表 + 入口
# ---------------------------------------------------------------------------
TASKS: Dict[str, Callable[[bool], None]] = {
    "cosyvoice":       task_cosyvoice,
    "cosyvoice_frd":   task_cosyvoice_ttsfrd,
    "bfm":             task_bfm,
    "r3d_ckpt":        task_r3d_ckpt,
    "r3d_pretrained":  task_r3d_pretrained,
    "mp_landmarker":   task_mp_landmarker,
    "mp_segmenter":    task_mp_segmenter,
    "hubert":          task_hubert,
    "whisper":         task_whisper,
    "silero":          task_silero,
}

# 默认全跑（cosyvoice_frd 可选；HuBERT 国内拉慢，但运行时也会自动下，所以也默认跑）
DEFAULT_TASKS = [
    "cosyvoice",
    "bfm",
    "r3d_ckpt",
    "r3d_pretrained",
    "mp_landmarker",
    "mp_segmenter",
    "hubert",
    "whisper",
    "silero",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="SynchroAvatar 预训练权重下载")
    parser.add_argument(
        "--only",
        type=str,
        default="",
        help=f"逗号分隔的任务名，可选：{','.join(TASKS.keys())}；默认全跑（不含 cosyvoice_frd）",
    )
    parser.add_argument("--force", action="store_true", help="强制重下，即使已存在")
    parser.add_argument("--list", action="store_true", help="只列出任务表并退出")
    args = parser.parse_args()

    if args.list:
        for k in TASKS:
            print(f"  {k:18s}  {'(default)' if k in DEFAULT_TASKS else '(optional)'}")
        return

    if args.only.strip():
        names = [s.strip() for s in args.only.split(",") if s.strip()]
        for n in names:
            if n not in TASKS:
                log(f"unknown task: {n}, available: {list(TASKS)}")
                sys.exit(2)
    else:
        names = list(DEFAULT_TASKS)

    log(f"repo root: {REPO_ROOT}")
    log(f"tasks: {names}")
    if not COSY_DIR.exists() or not R3D_DIR.exists():
        log(
            f"⚠️  未检测到 CosyVoice-main 或 Real3DPortrait-main 目录在 {REPO_ROOT}。"
            "请确认你正在仓库根目录下运行。"
        )

    failed: List[str] = []
    for name in names:
        log(f"==== [{name}] start ====")
        t0 = time.time()
        try:
            TASKS[name](args.force)
            log(f"==== [{name}] done in {time.time() - t0:.1f}s ====")
        except KeyboardInterrupt:
            log("interrupted by user")
            sys.exit(130)
        except Exception as exc:  # noqa: BLE001
            log(f"==== [{name}] FAILED: {exc} ====")
            failed.append(name)

    log("=" * 60)
    if failed:
        log(f"完成，但有失败任务：{failed}")
        log("常见原因：")
        log("  · Google Drive 在国内被墙 → 给系统挂代理后重试，或参考 README 手动下载")
        log("  · HuggingFace 慢 → $env:HF_ENDPOINT='https://hf-mirror.com' 后重试")
        log("  · 网络中断 → 加 --force 不一定有用，直接删除残留 .part 文件再跑")
        sys.exit(1)
    else:
        log("全部完成 ✅")
        log("现在可以启动服务：")
        log("  uvicorn realtime.server:app --host 0.0.0.0 --port 8000")


if __name__ == "__main__":
    main()


