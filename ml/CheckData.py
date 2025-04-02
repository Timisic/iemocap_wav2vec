import pandas as pd
import numpy as np
import os
from pathlib import Path

def check_data_matching():
    """检查标注数据和特征数据的匹配情况"""
    # 读取标注数据
    BASE_DIR = Path(__file__).parent.parent
    label_file = BASE_DIR / "src_competency" / "labels_2.xlsx"
    feature_dir = BASE_DIR / "ml" / "features" / "features_pca_15"
    
    print("=== 检查数据匹配情况 ===")
    
    # 读取标注文件
    df = pd.read_excel(label_file, sheet_name='Sheet1')
    print("\n1. 标注数据前10行:")
    print(df.head(10))
    
    # 检查特征文件
    print("\n2. 特征文件匹配情况:")
    matched_files = []
    for idx, row in df.iterrows():
        name = row['id']
        feature_path = feature_dir / f"{name}.npy"
        
        if feature_path.exists():
            matched_files.append({
                'id': name,
                'path': str(feature_path),
                'size': feature_path.stat().st_size,
                'feature_shape': np.load(feature_path).shape
            })
    
    print("\n匹配到的特征文件列表:")
    print("-" * 80)
    for file in matched_files:
        print(f"ID: {file['id']}")
        print(f"路径: {file['path']}")
        print(f"文件大小: {file['size']/1024:.2f} KB")
        print(f"特征维度: {file['feature_shape']}")
        print("-" * 80)
    
    # 统计整体匹配情况
    total_labels = len(df)
    matched_features = sum(1 for name in df['id'] if (feature_dir / f"{name}.npy").exists())
    
    print("\n3. 整体匹配统计:")
    print(f"- 标注数据总数: {total_labels}")
    print(f"- 匹配特征文件数: {matched_features}")
    print(f"- 匹配率: {matched_features/total_labels*100:.2f}%")

if __name__ == "__main__":
    check_data_matching()