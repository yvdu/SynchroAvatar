# -*- coding: utf-8 -*-
from cosyvoice.cli.cosyvoice import CosyVoice2
from openai import OpenAI
import argparse
import os
from backend_continue import run_pipeline
import re
import sys
sys.path.append(os.path.abspath("CosyVoice-main"))

def ask_model(client, user_input):
    """调用大模型并返回文本"""
    chat_completion = client.chat.completions.create(
        model="ernie-4.5-vl-28b-a3b",
        messages=[
            {"role": "user", "content": user_input}
        ],
        extra_body={"penalty_score": 1, "enable_thinking": True},
        max_completion_tokens=8000,
        temperature=0.2,
        top_p=0.8,
    )
    content = chat_completion.choices[0].message.content
    # 文本清理
    content = content.replace("\n", ",")
    content = content.replace("、", ",")
    content = re.sub(r"[*#@${}）（]", "", content)
    return content

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", type=str,
                        help="输入音频文件路径")
    parser.add_argument("--image", type=str,
                        help="输入图像文件路径")
    args = parser.parse_args()

    # 初始化 API
    client = OpenAI(
        api_key="替换为自己的apikey",
        base_url="替换为自己的base_url",
    )

    # 初始化 CosyVoice
    cosyvoice = CosyVoice2(
        'CosyVoice-main/pretrained_models/CosyVoice2-0.5B',
        load_jit=False, load_trt=False, fp16=False
    )

    print("=== 开始对话，输入 exit 退出 ===")
    while True:
        user_input = input("你: ")
        if user_input.strip().lower() in ["exit", "quit", "q"]:
            break

        # 调用大模型
        content = ask_model(client, user_input)
        print("大模型:", content)

        # 调用音视频合成
        run_pipeline(
            audio_path = args.audio,
            image_path = args.image,
            out_text = content,
            cosyvoice = cosyvoice,
            user_input = user_input
        )

if __name__ == "__main__":
    main()
