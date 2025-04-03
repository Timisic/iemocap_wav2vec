import os
import torch
import torchaudio
import numpy as np
from transformers import Wav2Vec2Processor, Wav2Vec2Model
import soundfile as sf
from pathlib import Path
from sklearn.decomposition import PCA
from tqdm import tqdm
import time

# 设置基础路径
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 加载本地预训练模型
model_path = os.path.join(BASE_DIR, "models", "wav2vec2-large-xlsr-53-chinese-zh-cn")
processor = Wav2Vec2Processor.from_pretrained(model_path)
model = Wav2Vec2Model.from_pretrained(model_path)

def extract_features(audio_path, target_duration=70):
    """提取wav2vec2特征，固定处理70秒音频"""
    audio_input, sampling_rate = sf.read(os.path.join(BASE_DIR, audio_path))
    
    if len(audio_input.shape) > 1:
        audio_input = audio_input[:, 0]
    
    audio_tensor = torch.FloatTensor(audio_input)
    
    if sampling_rate != 16000:
        resampler = torchaudio.transforms.Resample(sampling_rate, 16000)
        audio_tensor = resampler(audio_tensor)
    
    # 计算70秒对应的采样点数
    target_length = 16000 * target_duration
    current_length = len(audio_tensor)
    
    if current_length < target_length:
        # 短音频通过重复填充到70秒
        repeats = int(np.ceil(target_length / current_length))
        audio_tensor = audio_tensor.repeat(repeats)[:target_length]
    elif current_length > target_length:
        # 长音频取中间70秒
        start = (current_length - target_length) // 2
        audio_tensor = audio_tensor[start:start + target_length]
    
    # 提取特征
    inputs = processor(audio_tensor.numpy(), sampling_rate=16000, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        features = outputs.last_hidden_state
        final_features = torch.cat([
            torch.mean(features, dim=1),
            torch.max(features, dim=1)[0],
        ], dim=1)
    
    return final_features.numpy()

def process_and_save_features(audio_folder="src_competency/audio_pure", save_dir="ml/features/features_raw_divided"):
    save_path = os.path.join(BASE_DIR, save_dir)
    os.makedirs(save_path, exist_ok=True)
    
    audio_files = [f for f in os.listdir(os.path.join(BASE_DIR, audio_folder)) 
                  if f.endswith(('.wav', '.mp3', '.flac'))]
    
    print(f"\n开始处理音频文件，共 {len(audio_files)} 个文件")
    
    features_info = {}
    total_time = 0
    
    for audio_file in tqdm(audio_files, desc="处理音频文件"):
        start_time = time.time()
        
        name = os.path.splitext(audio_file)[0]
        features = extract_features(os.path.join(audio_folder, audio_file))
        features_flat = features.reshape(1, -1)
        
        # 保存原始特征
        feature_path = os.path.join(save_path, f"{name}.npy")
        np.save(feature_path, features_flat)
        
        process_time = time.time() - start_time
        total_time += process_time
        
        features_info[name] = {
            'feature_path': feature_path,
            'feature_shape': features_flat.shape,
            'process_time': process_time
        }
        
        print(f"\n{name} - 处理时间: {process_time:.2f}秒")
        print(f"特征形状: {features.shape}")
    
    np.save(os.path.join(save_path, 'features_info.npy'), features_info)
    
    print(f"\n特征提取完成！")
    print(f"总处理时间: {total_time:.2f}秒")
    print(f"平均每个文件处理时间: {total_time/len(audio_files):.2f}秒")
    print(f"特征保存在: {save_path}")
    
    return features_info

if __name__ == "__main__":
    # 第一步：提取并保存原始特征
    features_info = process_and_save_features()
    print("\n原始特征维度示例:")
    for name, info in features_info.items():
        print(f"{name}: {info['feature_shape']}")
