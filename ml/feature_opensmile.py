import os
import numpy as np
from pathlib import Path
import opensmile
import time
from tqdm import tqdm
import soundfile as sf

# 设置基础路径
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def extract_features(audio_path, target_duration=70):
    """使用OpenSMILE提取特征，固定处理70秒音频"""
    # 初始化特征提取器 (使用ComParE_2016特征集)
    smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.ComParE_2016,
        feature_level=opensmile.FeatureLevel.Functionals,
    )
    
    # 读取音频
    audio_input, sampling_rate = sf.read(os.path.join(BASE_DIR, audio_path))
    
    if len(audio_input.shape) > 1:
        audio_input = audio_input[:, 0]
    
    # 处理音频长度
    target_length = sampling_rate * target_duration
    current_length = len(audio_input)
    
    if current_length < target_length:
        # 短音频通过重复填充到70秒
        repeats = int(np.ceil(target_length / current_length))
        audio_input = np.tile(audio_input, repeats)[:target_length]
    elif current_length > target_length:
        # 长音频取中间70秒
        start = (current_length - target_length) // 2
        audio_input = audio_input[start:start + target_length]
    
    # 提取特征
    features = smile.process_signal(
        audio_input,
        sampling_rate
    )
    
    return features.values

def process_and_save_features(audio_folder="src_competency/audio_pure", save_dir="ml/features/features_opensmile"):
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
        
        # 保存特征
        feature_path = os.path.join(save_path, f"{name}.npy")
        np.save(feature_path, features)
        
        process_time = time.time() - start_time
        total_time += process_time
        
        features_info[name] = {
            'feature_path': feature_path,
            'feature_shape': features.shape,
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
    features_info = process_and_save_features()
    print("\n特征维度示例:")
    for name, info in features_info.items():
        print(f"{name}: {info['feature_shape']}")