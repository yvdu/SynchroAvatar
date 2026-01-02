import os
import gc
import torch
import time
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_huggingface import HuggingFacePipeline
t1 = time.time()
# ========== 命令行参数解析 ==========
parser = argparse.ArgumentParser()
parser.add_argument("--audio", type=str,
                    help="输入音频文件路径")
parser.add_argument("--image", type=str,
                    help="输入图像文件路径")
parser.add_argument("--text", type=str, default=None,
                    help="输入文本。如果为空则调用模型生成")
args = parser.parse_args()

# ========== 环境设置 ==========
os.environ["http_proxy"] = "http://127.0.0.1:7890"
os.environ["https_proxy"] = "http://127.0.0.1:7890"
os.system('export HF_ENDPOINT=https://hf-mirror.com')

# ========== 加载本地 Qwen 模型 ==========
model_name = "./qwen2.5-7b" #替换为自己的模型路径

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)

# HuggingFace pipeline 封装
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=1024,
    do_sample=True,
    temperature=0.7
)
llm = HuggingFacePipeline(pipeline=pipe)

# ========== 获取文本内容 ==========
prompt = args.text
messages = [{"role": "user", "content": prompt}]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

generated_ids = model.generate(**model_inputs, max_new_tokens=1024)
output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
content = tokenizer.decode(output_ids, skip_special_tokens=True)
print("General Chat:", content)


# ========== 清理内存 ==========
del model, tokenizer, pipe, llm, model_inputs, generated_ids, output_ids
gc.collect()
torch.cuda.empty_cache()

# ========== 调用后端程序 ==========
os.system(f"python3 main_backend2.py --audio {args.audio} --image {args.image} --text \"{content}\"")
