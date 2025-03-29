import os
import torch
import torchaudio
import numpy as np
from transformers import Wav2Vec2Processor, Wav2Vec2Model
import soundfile as sf
from pathlib import Pathx

# 设置基础路径
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 加载本地预训练模型
model_path = os.path.join(BASE_DIR, "models", "wav2vec2-base-960h")
processor = Wav2Vec2Processor.from_pretrained(model_path)
model = Wav2Vec2Model.from_pretrained(model_path)

def analyze_audio_lengths(audio_folder="src_competency/matched_audio"):
    """分析音频长度分布"""
    audio_lengths = []
    audio_files = [f for f in os.listdir(os.path.join(BASE_DIR, audio_folder)) 
                  if f.endswith(('.wav', '.mp3', '.flac'))]
    
    for audio_file in audio_files:
        audio_path = os.path.join(BASE_DIR, audio_folder, audio_file)
        audio_input, sr = sf.read(audio_path)
        duration = len(audio_input) / sr  # 计算时长（秒）
        audio_lengths.append((audio_file, duration))
    
    audio_lengths.sort(key=lambda x: x[1])  # 按时长排序
    
    print("\n音频长度分布：")
    for file, duration in audio_lengths:
        print(f"{file}: {duration:.2f}秒")
    
    median_length = np.median([d for _, d in audio_lengths])
    print(f"\n中位数长度: {median_length:.2f}秒")
    return audio_lengths, median_length

def extract_features(audio_path, window_size=16000*5, target_windows=60):
    """提取固定维度的特征"""
    audio_input, sampling_rate = sf.read(os.path.join(BASE_DIR, audio_path))
    
    if len(audio_input.shape) > 1:
        audio_input = audio_input[:, 0]
    
    audio_tensor = torch.FloatTensor(audio_input)
    
    if sampling_rate != 16000:
        resampler = torchaudio.transforms.Resample(sampling_rate, 16000)
        audio_tensor = resampler(audio_tensor)
    
    total_length = len(audio_tensor)
    
    # 计算步长以获得固定数量的窗口
    if total_length < window_size:
        # 短音频通过重复填充到目标长度
        repeats = int(np.ceil(window_size / total_length))
        audio_tensor = audio_tensor.repeat(repeats)[:window_size]
        windows = [audio_tensor] * target_windows
    else:
        # 长音频通过滑动窗口获取固定数量的片段
        total_stride = max(1, (total_length - window_size) // (target_windows - 1))
        windows = []
        for i in range(target_windows):
            start = min(i * total_stride, total_length - window_size)
            window = audio_tensor[start:start + window_size]
            windows.append(window)
    
    # 提取特征
    all_features = []
    for window in windows:
        inputs = processor(window.numpy(), sampling_rate=16000, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
            features = outputs.last_hidden_state
            # 使用更细粒度的特征表示
            window_features = torch.cat([
                torch.mean(features, dim=1),    # 平均池化
                torch.max(features, dim=1)[0],  # 最大池化
            ], dim=1)
            all_features.append(window_features)
    
    # 将所有窗口特征拼接并统一维度
    final_features = torch.stack(all_features, dim=0)
    return final_features

def process_and_save_features(audio_folder="src_competency/audio", save_dir="ml_test/features"):
    """处理并保存特征"""
    save_path = os.path.join(BASE_DIR, save_dir)
    os.makedirs(save_path, exist_ok=True)
    
    # 先分析音频长度
    audio_lengths, median_length = analyze_audio_lengths(audio_folder)
    
    features_info = {}
    for audio_file, duration in audio_lengths:
        print(f"\n处理文件: {audio_file} (时长: {duration:.2f}秒)")
        name = os.path.splitext(audio_file)[0]
        
        # 提取特征
        features = extract_features(os.path.join(audio_folder, audio_file))
        
        # 保存特征
        feature_path = os.path.join(save_path, f"{name}.npy")
        np.save(feature_path, features.numpy())
        
        # 记录信息
        features_info[name] = {
            'feature_path': feature_path,
            'feature_shape': features.shape,
            'duration': duration
        }
    
    # 保存特征信息
    np.save(os.path.join(save_path, 'features_info.npy'), features_info)
    print(f"\n特征提取完成！共处理了 {len(audio_lengths)} 个音频文件")
    print(f"特征保存在: {save_path}")
    return features_info

if __name__ == "__main__":
    features_info = process_and_save_features()
    print("\n特征维度示例:")
    for name, info in features_info.items():
        print(f"{name}: {info['feature_shape']} (音频时长: {info['duration']:.2f}秒)")
