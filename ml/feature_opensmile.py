import os
import numpy as np
from pathlib import Path
import opensmile
import time
from tqdm import tqdm
import soundfile as sf
import librosa
import noisereduce as nr
from scipy import signal

# 设置基础路径
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def preprocess_audio(audio_input, sampling_rate):
    """音频预处理函数"""
    # 降噪
    audio_reduced_noise = nr.reduce_noise(
        y=audio_input,
        sr=sampling_rate,
        stationary=True,
        prop_decrease=0.75
    )
    
    # 预加重，增强高频部分
    pre_emphasis = 0.97
    audio_pre_emphasis = np.append(
        audio_reduced_noise[0],
        audio_reduced_noise[1:] - pre_emphasis * audio_reduced_noise[:-1]
    )
    
    # 应用带通滤波器，保留语音主要频率范围
    nyquist = sampling_rate // 2
    low = 80 / nyquist
    high = 4000 / nyquist
    b, a = signal.butter(4, [low, high], btype='band')
    audio_filtered = signal.filtfilt(b, a, audio_pre_emphasis)
    
    # 音量归一化
    audio_normalized = librosa.util.normalize(audio_filtered)
    
    return audio_normalized

def extract_features(audio_path, target_duration=60):
    """使用OpenSMILE提取特征，固定处理秒音频"""
    print(f"\n处理音频文件: {audio_path}")
    
    print("初始化OpenSMILE特征提取器...")
    smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.ComParE_2016,
        feature_level=opensmile.FeatureLevel.Functionals,
    )
    
    print("读取音频文件...")
    audio_input, sampling_rate = sf.read(os.path.join(BASE_DIR, audio_path))
    print(f"采样率: {sampling_rate} Hz")
    print(f"原始音频长度: {len(audio_input)} 采样点")
    
    if len(audio_input.shape) > 1:
        print("转换为单声道...")
        audio_input = audio_input[:, 0]
    
    print("执行音频预处理...")
    audio_input = preprocess_audio(audio_input, sampling_rate)
    
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

def process_and_save_features(audio_folder="src_competency/audio_pure", save_dir="ml/features/features_opensmile_divided"):
    save_path = os.path.join(BASE_DIR, save_dir)
    os.makedirs(save_path, exist_ok=True)
    
    audio_files = [f for f in os.listdir(os.path.join(BASE_DIR, audio_folder)) 
                  if f.endswith(('.wav', '.mp3', '.flac'))]
    
    print(f"\n开始处理音频文件，共 {len(audio_files)} 个文件")
    
    features_info = {}
    total_time = 0
    
    # 收集所有特征用于全局标准化
    all_features = []
    print("\n第一遍：提取所有特征...")
    for audio_file in tqdm(audio_files, desc="提取特征"):
        features = extract_features(os.path.join(audio_folder, audio_file))
        all_features.append(features)
    
    # 全局标准化
    print("\n执行全局标准化...")
    all_features = np.vstack(all_features)
    global_mean = np.mean(all_features, axis=0)
    global_std = np.std(all_features, axis=0)
    
    # 保存标准化参数
    np.save(os.path.join(save_path, 'global_mean.npy'), global_mean)
    np.save(os.path.join(save_path, 'global_std.npy'), global_std)
    print("已保存全局标准化参数")
    
    # 第二遍：保存标准化后的特征
    print("\n第二遍：保存标准化后的特征...")
    for i, audio_file in enumerate(tqdm(audio_files, desc="保存特征")):
        features_normalized = (all_features[i] - global_mean) / (global_std + 1e-8)
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
    # 配置路径
    print("\n=== 特征提取配置 ===")
    audio_folder = "src_competency/audio_pure"
    save_dir = "ml/features/features_opensmile_divided"
    
    print(f"\n当前配置:")
    print(f"音频文件夹: {os.path.join(BASE_DIR, audio_folder)}")
    print(f"特征保存目录: {os.path.join(BASE_DIR, save_dir)}")
    
    confirm = input("\n确认开始处理? (y/n): ").lower()
    if confirm != 'y':
        print("已取消处理")
        exit()
    
    features_info = process_and_save_features(audio_folder, save_dir)
    
    print("\n特征维度示例:")
    for name, info in features_info.items():
        print(f"{name}: {info['feature_shape']}")