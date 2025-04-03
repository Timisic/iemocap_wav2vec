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
    print(f"\n处理音频文件: {audio_path}")
    
    print("读取音频文件...")
    audio_input, sampling_rate = sf.read(os.path.join(BASE_DIR, audio_path))
    print(f"采样率: {sampling_rate} Hz")
    print(f"原始音频长度: {len(audio_input)} 采样点")
    
    if len(audio_input.shape) > 1:
        print("转换为单声道...")
        audio_input = audio_input[:, 0]
    
    print("转换为tensor...")
    audio_tensor = torch.FloatTensor(audio_input)
    
    if sampling_rate != 16000:
        print(f"重采样到 16kHz...")
        resampler = torchaudio.transforms.Resample(sampling_rate, 16000)
        audio_tensor = resampler(audio_tensor)
    
    # 处理音频长度
    target_length = 16000 * target_duration
    current_length = len(audio_tensor)
    print(f"目标长度: {target_length} 采样点")
    
    if current_length < target_length:
        print("音频较短，进行重复填充...")
        repeats = int(np.ceil(target_length / current_length))
        audio_tensor = audio_tensor.repeat(repeats)[:target_length]
    elif current_length > target_length:
        print("音频较长，截取中间部分...")
        start = (current_length - target_length) // 2
        audio_tensor = audio_tensor[start:start + target_length]
    
    print("使用wav2vec2提取特征...")
    inputs = processor(audio_tensor.numpy(), sampling_rate=16000, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        features = outputs.last_hidden_state
        final_features = torch.cat([
            torch.mean(features, dim=1),
            torch.max(features, dim=1)[0],
        ], dim=1)
    
    print(f"特征形状: {final_features.shape}")
    return final_features.numpy()

def process_and_save_features(audio_folder="src_competency/audio_pure", save_dir="ml/features/features_wav2vec2_divided"):
    save_path = os.path.join(BASE_DIR, save_dir)
    os.makedirs(save_path, exist_ok=True)
    print(f"\n创建保存目录: {save_path}")
    
    audio_files = [f for f in os.listdir(os.path.join(BASE_DIR, audio_folder)) 
                  if f.endswith(('.wav', '.mp3', '.flac'))]
    
    print(f"\n开始处理音频文件，共 {len(audio_files)} 个文件")
    
    # 收集所有特征用于全局标准化
    all_features = []
    print("\n第一遍：提取所有特征...")
    for audio_file in tqdm(audio_files, desc="提取特征"):
        features = extract_features(os.path.join(audio_folder, audio_file))
        features_flat = features.reshape(1, -1)
        all_features.append(features_flat)
    
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
    features_info = {}
    total_time = 0
    
    print("\n第二遍：保存标准化后的特征...")
    for i, audio_file in enumerate(tqdm(audio_files, desc="保存特征")):
        start_time = time.time()
        name = os.path.splitext(audio_file)[0]
        
        # 使用全局参数进行标准化
        features_normalized = (all_features[i] - global_mean) / (global_std + 1e-8)
        
        # 保存特征
        feature_path = os.path.join(save_path, f"{name}.npy")
        np.save(feature_path, features_normalized)
        
        process_time = time.time() - start_time
        total_time += process_time
        
        features_info[name] = {
            'feature_path': feature_path,
            'feature_shape': features_normalized.shape,
            'process_time': process_time
        }
    
    np.save(os.path.join(save_path, 'features_info.npy'), features_info)
    
    print(f"\n特征提取完成！")
    print(f"总处理时间: {total_time:.2f}秒")
    print(f"平均每个文件处理时间: {total_time/len(audio_files):.2f}秒")
    print(f"特征保存在: {save_path}")
    print(f"特征维度: {features_normalized.shape[1]}")
    
    return features_info

if __name__ == "__main__":
    print("\n=== Wav2Vec2特征提取配置 ===")

    audio_folder = "src_competency/audio_pure"
    save_dir = "ml/features/features_wav2vec2_divided"
    
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
