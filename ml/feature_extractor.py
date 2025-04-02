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

def extract_features(audio_path, window_size=16000*5, target_windows=60):
    """提取wav2vec2特征，不进行PCA降维"""
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
            window_features = torch.cat([
                torch.mean(features, dim=1),
                torch.max(features, dim=1)[0],
            ], dim=1)
            all_features.append(window_features)
    
    final_features = torch.stack(all_features, dim=0)
    return final_features.numpy()

def process_and_save_features(audio_folder="src_competency/audio", save_dir="ml/features/features_raw"):
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

def apply_pca(features_dir="ml/features/features_raw", save_dir="ml/features/features_pca_15", n_components=15, train_indices=None):
    features_path = os.path.join(BASE_DIR, features_dir)
    save_path = os.path.join(BASE_DIR, save_dir)
    os.makedirs(save_path, exist_ok=True)
    
    features_info = np.load(os.path.join(features_path, 'features_info.npy'), allow_pickle=True).item()
    
    all_features = []
    sample_names = []
    
    print("\n开始加载原始特征...")
    for name in tqdm(features_info.keys(), desc="加载特征"):
        features = np.load(features_info[name]['feature_path'])
        all_features.append(features)
        sample_names.append(name)
    
    # 将所有特征拼接
    all_features = np.vstack(all_features)
    print(f"合并后特征矩阵形状: {all_features.shape}")
    
    # PCA降维
    print("\n执行PCA降维...")
    pca = PCA(n_components=n_components)
    
    # 只使用训练集数据拟合PCA
    if train_indices is not None:
        train_features = all_features[train_indices]
        pca.fit(train_features)
    else:
        # 如果没有提供训练集索引，使用所有数据拟合
        pca.fit(all_features)
    
    # 使用训练好的PCA转换所有数据
    all_features_reduced = pca.transform(all_features)
    print(f"降维后特征矩阵形状: {all_features_reduced.shape}")
    
    # 保存降维后的特征
    new_features_info = {}
    for i, name in enumerate(sample_names):
        features = all_features_reduced[i].reshape(1, -1)
        feature_path = os.path.join(save_path, f"{name}.npy")
        np.save(feature_path, features)
        
        new_features_info[name] = {
            'feature_path': feature_path,
            'feature_shape': features.shape
        }
    
    # 保存PCA相关信息
    np.save(os.path.join(save_path, 'features_info.npy'), new_features_info)
    np.save(os.path.join(save_path, 'pca_components.npy'), pca.components_)
    np.save(os.path.join(save_path, 'pca_mean.npy'), pca.mean_)
    
    print(f"\nPCA降维完成！")
    print(f"降维后特征保存在: {save_path}")
    print(f"解释方差比: {pca.explained_variance_ratio_.sum():.4f}")
    
    return new_features_info

if __name__ == "__main__":
    # 第一步：提取并保存原始特征
    # features_info = process_and_save_features()
    # print("\n原始特征维度示例:")
    # for name, info in features_info.items():
    #     print(f"{name}: {info['feature_shape']}")
    
    # 第二步：对保存的特征进行PCA降维
    pca_features_info = apply_pca()
    print("\nPCA降维后特征维度示例:")
    for name, info in pca_features_info.items():
        print(f"{name}: {info['feature_shape']}")
