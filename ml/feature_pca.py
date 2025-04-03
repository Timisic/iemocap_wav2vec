import os
import numpy as np
from pathlib import Path
from sklearn.decomposition import PCA
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler

def apply_pca(features_dir, save_dir, n_components=50, train_indices=None):
    """
    对特征进行PCA降维
    
    Args:
        features_dir (str): 原始特征目录
        save_dir (str): 降维后特征保存目录
        n_components (int): PCA降维后的维度
        train_indices (array-like): 训练集索引，用于拟合PCA
    """
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
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
    
    # 添加数据标准化
    print("\n执行数据标准化...")
    scaler = StandardScaler()
    all_features_scaled = scaler.fit_transform(all_features)
    
    # 移除零方差特征
    feature_variances = np.var(all_features_scaled, axis=0)
    non_zero_var_mask = feature_variances > 0
    all_features_scaled = all_features_scaled[:, non_zero_var_mask]
    print(f"移除零方差特征后的特征矩阵形状: {all_features_scaled.shape}")
    
    # PCA降维
    print("\n执行PCA降维...")
    # 添加方差解释率阈值
    cumsum_ratio = []
    pca_full = PCA().fit(all_features_scaled)
    for i, ratio in enumerate(np.cumsum(pca_full.explained_variance_ratio_)):
        cumsum_ratio.append(ratio)
        print(f"使用前 {i+1} 个主成分的累积解释方差比: {ratio:.4f}")
        if ratio > 0.95:  # 可以根据需要调整阈值
            print(f"使用 {i+1} 个主成分可以解释95%的方差")
            break
    
    # 继续使用指定的n_components进行降维
    pca = PCA(n_components=n_components)
    
    # 使用标准化后的数据进行PCA
    if train_indices is not None:
        train_features = all_features_scaled[train_indices]
        pca.fit(train_features)
    else:
        pca.fit(all_features_scaled)
    
    # 使用训练好的PCA转换所有数据
    all_features_reduced = pca.transform(all_features_scaled)
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
    
    print(f"\nPCA降维完成!")
    print(f"降维后特征保存在: {save_path}")
    print(f"解释方差比: {pca.explained_variance_ratio_.sum():.4f}")
    
    return new_features_info

if __name__ == "__main__":
    # 指定降维维度
    n_components = 40
    # model = "wav2vec2"
    model = "opensmile"
    
    features_dir = f"ml/features/features_{model}_divided"
    save_dir = f"ml/features/features_{model}_pca_{n_components}"
    pca_features_info = apply_pca(features_dir, save_dir, n_components=n_components)