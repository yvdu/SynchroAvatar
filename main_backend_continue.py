import os
import argparse
import sys
sys.path.append(os.path.abspath("CosyVoice-main"))
from test_voise2 import run_test_voice2
from cosyvoice.cli.cosyvoice import CosyVoice2
import time
from datetime import datetime
def run_pipeline(audio_path: str, image_path: str, out_text: str,cosyvoice = None,user_input = None):
    """
    主处理流程：根据音频 + 图像 + 文本生成视频
    """
    now = datetime.now()
    print("当前本地时间:", now)
    formatted = now.strftime("%Y%m%d_%H%M%S")
    if user_input == None:
        user_input = formatted
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
    #os.system(f'python test_voise2.py --voice_path "{audio_path}" --pt_name temp_chinese.pt --text "{out_text}"')
    print("音频合成成功")
    t2 = time.time()
    print("CosyVoice", t2 - t1)
    # ========== Real3DPortrait ==========

    t1 = time.time()
    os.chdir('../Real3DPortrait-main')
    os.makedirs('video', exist_ok=True)
    os.system(f'python inference/real3d_infer_video.py --src_img "{image_path}" --drv_aud ../CosyVoice-main/temp.wav --out_name 1')
    t2 = time.time()
    print("Real3DPortrait",t2 - t1)
    # 重命名输出视频
    out_path = f'video/{user_input}.mp4'
    if os.path.exists('infer_out/tmp/cropped_src_img_May_5s_coeff_fit_mp.mp4'):
        os.rename('infer_out/tmp/cropped_src_img_May_5s_coeff_fit_mp.mp4', out_path)
    print(f"已完成: Real3DPortrait-main/{out_path}")


if __name__ == "__main__":
    # 命令行参数解析
    parser = argparse.ArgumentParser(description="Process input")
    parser.add_argument("--audio", type=str, required=True, help="Path to audio")
    parser.add_argument("--image", type=str, required=True, help="Path to image")
    parser.add_argument("--text", type=str, required=True, help="text,str")
    args = parser.parse_args()

    run_pipeline(audio_path=args.audio, image_path=args.image, out_text=args.text)
