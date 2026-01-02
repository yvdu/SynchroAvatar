# -*- coding: utf-8 -*-
# pip install openai
from cosyvoice.cli.cosyvoice import CosyVoice2
from openai import OpenAI
import argparse
import time
import os
from main_backend2 import run_pipeline
import re
import sys
sys.path.append(os.path.abspath("CosyVoice-main"))
client = OpenAI(
    api_key="替换为自己的apikey",
    base_url="替换为自己的base_url",
)
from cosyvoice.cli.cosyvoice import CosyVoice2

t1 = time.time()
parser = argparse.ArgumentParser()
parser.add_argument("--audio", type=str,
                    help="输入音频文件路径")
parser.add_argument("--image", type=str,
                    help="输入图像文件路径")
parser.add_argument("--bg_img", type=str,
                    help="背景图像文件路径")
parser.add_argument("--position", type=str, default="left_bottom",
                    help="人物图像位置")
parser.add_argument("--text", type=str, default=None,
                    help="输入文本。如果为空则调用模型生成")
args = parser.parse_args()


chat_completion = client.chat.completions.create(
    model="ernie-4.5-vl-28b-a3b",
    messages=[
    {
        "role": "user",
        "content": "请用一句话介绍黄河" #修改为自己想发送给大模型的文本
    }
],
    extra_body={
        "penalty_score": 1,
        "enable_thinking": True
    },
    max_completion_tokens=8000,
    temperature=0.2,
    top_p=0.8,
    frequency_penalty=0,
    presence_penalty=0
)


content = chat_completion.choices[0].message.content
content = content.replace("\n", ",")
content = content.replace("、", ",")
# 2. 删除特殊字符（这里以 * # @ $ 为例，可以按需扩展）
content = re.sub(r"[*#@${}）（]", "", content)
print(content)

cosyvoice = CosyVoice2('CosyVoice-main/pretrained_models/CosyVoice2-0.5B',
                           load_jit=False, load_trt=False, fp16=False)
run_pipeline(audio_path = args.audio,
             image_path = args.image,
             out_text = content,
             cosyvoice = cosyvoice,
             bg_img = args.bg_img,
             position = args.position)

t2 = time.time()
print(t2 - t1)