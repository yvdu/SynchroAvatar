import numpy as np
from scipy.io.wavfile import write

# 定义参数
sample_rate = 16000  # 采样率，通常使用44100 Hz
duration = 0.1     # 持续时间，设置为0.001秒
num_samples = int(sample_rate * duration)

# 创建一个只有零（静默）的音频信号
silent_audio = np.zeros(num_samples, dtype=np.float32)

# 保存为WAV文件
audio_file = 'silent_audio_short.wav'
write(audio_file, sample_rate, silent_audio)

print(f"无声的WAV文件 '{audio_file}' 已生成。")