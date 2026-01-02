import os
import argparse
import sys
sys.path.append(os.path.abspath("CosyVoice-main"))
from voice_utils import run_test_voice2
from cosyvoice.cli.cosyvoice import CosyVoice2
import time
from moviepy.editor import VideoFileClip, AudioFileClip
def run_pipeline(audio_path: str, image_path: str, out_text: str, bg_img: str, position: str,cosyvoice = None):
    if cosyvoice == None:
        cosyvoice = CosyVoice2('./CosyVoice-main/pretrained_models/CosyVoice2-0.5B',
                               load_jit=False, load_trt=False, fp16=False)

    print("image_path:", image_path)
    print("audio_path:", audio_path)
    print("out_text:", out_text)

    # ========== CosyVoice ==========
    t1 = time.time()
    os.chdir('CosyVoice-main')
    os.makedirs('audio_out', exist_ok=True)

    run_test_voice2(voice_path = audio_path,
                    pt_name = "temp_chinese.pt",
                    text = out_text,
                    cosyvoice = cosyvoice)
    #os.system(f'python voice_utils.py --voice_path "{audio_path}" --pt_name temp_chinese.pt --text "{out_text}"')
    print("音频合成成功")
    t2 = time.time()
    print("CosyVoice", t2 - t1)
    # ========== Real3DPortrait ==========

    t1 = time.time()
    os.chdir('../Real3DPortrait-main')
    os.makedirs('video', exist_ok=True)
    os.system(f'python inference/real3d_infer_video.py --src_img "{image_path}" --drv_aud ../CosyVoice-main/temp/temp.wav --bg_img {bg_img} --position {position} --out_name 1')



    from datetime import datetime, timezone, timedelta

    t2 = time.time()
    tz = timezone(timedelta(hours=8))  # 东八区
    time_str = datetime.fromtimestamp(t2, tz).strftime("%Y%m%d%H%M%S")

    # 重命名输出视频
    out_path = f'video/{time_str}.mp4'
    if os.path.exists('infer_out/tmp/cropped_src_img_May_5s_coeff_fit_mp.mp4'):
        overlay_video_on_image('infer_out/tmp/cropped_src_img_May_5s_coeff_fit_mp.mp4', bg_img, position, out_path)

    video_path = out_path
    audio_path = "../CosyVoice-main/temp/temp.wav"
    output_path = f'video/{time_str}.mp4'

    # 读取视频和音频
    video_clip = VideoFileClip(video_path)
    audio_clip = AudioFileClip(audio_path)

    # 将音频设置为视频的音轨
    video_with_audio = video_clip.set_audio(audio_clip)

    # 保存为新视频（保留音质）
    video_with_audio.write_videofile(output_path, codec="libx264", audio_codec="aac")

    print(f"已完成: Real3DPortrait-main/{out_path}")

import cv2
import os

def overlay_video_on_image(video_path, image_path, position="center", output_path="output.mp4"):
    """
    将512x512的视频叠加到一张图片上，可指定放置位置。
    如果图片宽或高小于512，则跳过输出。

    参数:
        video_path: 输入视频路径 (512x512)
        image_path: 背景图片路径
        position: "left_top"、"right_top"、"left_bottom"、"right_bottom"、"center"
        output_path: 输出视频路径
    """
    # 读取背景图
    bg_img = cv2.imread(image_path)
    bg_img = cv2.cvtColor(bg_img, cv2.COLOR_BGR2RGB)
    if bg_img is None:
        print(f"无法读取图片: {image_path}")
        return

    h, w, _ = bg_img.shape
    target_size = 512

    # 如果背景图太小，直接跳过
    if h < target_size or w < target_size:
        print(f"跳过图片 {image_path}，尺寸太小: {w}x{h}")
        return

    # 打开视频
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    # 输出视频尺寸与背景图一致
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    # 计算视频放置位置
    if position == "left_top":
        x1, y1 = 0, 0
    elif position == "right_top":
        x1, y1 = w - target_size, 0
    elif position == "left_bottom":
        x1, y1 = 0, h - target_size
    elif position == "right_bottom":
        x1, y1 = w - target_size, h - target_size
    else:  # center
        x1 = (w - target_size) // 2
        y1 = (h - target_size) // 2

    x2, y2 = x1 + target_size, y1 + target_size

    # 逐帧叠加视频
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # frame 已经是512x512，可直接放置
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 创建背景副本
        bg_copy = bg_img.copy()

        # 覆盖区域
        bg_copy[y1:y2, x1:x2] = frame_rgb

        # 转回 BGR 以保存
        frame_bgr = cv2.cvtColor(bg_copy, cv2.COLOR_RGB2BGR)
        out.write(frame_bgr)

    cap.release()
    out.release()
    print(f"✅ 已保存视频到: {output_path}")



if __name__ == "__main__":
    # 命令行参数解析
    parser = argparse.ArgumentParser(description="Process input")
    parser.add_argument("--audio", type=str, required=True, help="Path to audio")
    parser.add_argument("--image", type=str, required=True, help="Path to image")
    parser.add_argument("--text", type=str, required=True, help="text,str")
    args = parser.parse_args()

    run_pipeline(audio_path=args.audio, image_path=args.image, out_text=args.text)
