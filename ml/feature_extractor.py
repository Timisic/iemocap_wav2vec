import os
import torch
import torchaudio
import numpy as np
from transformers import Wav2Vec2Processor, Wav2Vec2Model
import soundfile as sf
from pathlib import Path
from sklearn.decomposition import PCA

# 设置基础路径
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 加载本地预训练模型
model_path = os.path.join(BASE_DIR, "models", "wav2vec2-base-960h")
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

def process_and_save_features(audio_folder="src_competency/audio", save_dir="ml/features_pca2"):
    """处理并保存特征"""
    save_path = os.path.join(BASE_DIR, save_dir)
    os.makedirs(save_path, exist_ok=True)
    
    # 直接获取音频文件列表
    audio_files = [f for f in os.listdir(os.path.join(BASE_DIR, audio_folder)) 
                  if f.endswith(('.wav', '.mp3', '.flac'))]
    
    print(f"\n开始处理音频文件，共 {len(audio_files)} 个文件")
    
    # 收集所有样本的特征
    all_samples_features = []
    sample_names = []
    
    for idx, audio_file in enumerate(audio_files, 1):
        print(f"\n处理第 {idx}/{len(audio_files)} 个文件: {audio_file}")
        name = os.path.splitext(audio_file)[0]
        
        print("提取 wav2vec2 特征...")
        features = extract_features(os.path.join(audio_folder, audio_file))
        print(f"特征形状: {features.shape}")
        
        features_flat = features.reshape(1, -1)
        print(f"展平后特征维度: {features_flat.shape}")
        
        all_samples_features.append(features_flat)
        sample_names.append(name)
    
    print("\n所有文件处理完成，开始PCA降维...")
    # 将所有样本的特征拼接
    all_features = np.vstack(all_samples_features)
    print(f"合并后特征矩阵形状: {all_features.shape}")
    
    # 对所有样本进行PCA降维到60维
    pca = PCA(n_components=60)
    print("执行PCA降维...")
    all_features_reduced = pca.fit_transform(all_features)
    print(f"降维后特征矩阵形状: {all_features_reduced.shape}")
    
    print("\n保存降维后的特征...")
    
    # 保存每个样本的降维后特征
    features_info = {}
    for i, name in enumerate(sample_names):
        features = all_features_reduced[i].reshape(1, -1)  # 保持二维数组形式
        
        # 保存特征
        feature_path = os.path.join(save_path, f"{name}.npy")
        np.save(feature_path, features)
        
        features_info[name] = {
            'feature_path': feature_path,
            'feature_shape': features.shape
        }
    
    # 保存特征信息和PCA模型
    np.save(os.path.join(save_path, 'features_info.npy'), features_info)
    np.save(os.path.join(save_path, 'pca_components.npy'), pca.components_)
    np.save(os.path.join(save_path, 'pca_mean.npy'), pca.mean_)
    
    print(f"\n特征提取完成！共处理了 {len(audio_files)} 个音频文件")
    print(f"特征保存在: {save_path}")
    print(f"PCA降维后的特征维度: {all_features_reduced.shape}")
    print(f"解释方差比: {pca.explained_variance_ratio_.sum():.4f}")
    
    return features_info

if __name__ == "__main__":
    features_info = process_and_save_features()
    print("\n特征维度示例:")
    for name, info in features_info.items():
        print(f"{name}: {info['feature_shape']}")
