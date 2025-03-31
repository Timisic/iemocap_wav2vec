import os
import warnings
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr
from autogluon.tabular import TabularPredictor

# 过滤警告信息
warnings.filterwarnings('ignore', category=UserWarning)

# 设置随机种子
np.random.seed(42)

# 设置基础路径
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(BASE_DIR, "autogluon_models")

# 加载已训练的预测器
predictor = TabularPredictor.load(model_path)

# 加载特征和标签
features_dir = os.path.join(BASE_DIR, "ml", "features_pca1")
label_file = os.path.join(BASE_DIR, "src_competency", "labels_2.xlsx")

# 读取标签数据
df = pd.read_excel(label_file, sheet_name='Sheet1')
labels = df['分析力得分'].values

# 读取特征数据
# 获取所有特征数据
features = []
valid_indices = []
for idx, row in df.iterrows():
    name = row['id']
    feature_path = os.path.join(features_dir, f"{name}.npy")
    if os.path.exists(feature_path):
        feature = np.load(feature_path)
        features.append(feature.reshape(-1))
        valid_indices.append(idx)

features = np.array(features)
labels = labels[valid_indices]

# 将特征数据转换为DataFrame格式
feature_names = [f'feature_{i}' for i in range(features.shape[1])]
test_data = pd.DataFrame(features, columns=feature_names)

# 指定要评估的模型
target_models = [
    'ExtraTrees_r197_BAG_L1',
    'NeuralNetTorch_r76_BAG_L1',
    'RandomForest_r16_BAG_L1',
    'ExtraTrees_r126_BAG_L1',
    'NeuralNetTorch_r19_BAG_L1',
    'XGBoost_r95_BAG_L1',
    'NeuralNetTorch_r89_BAG_L1',
    'XGBoost_BAG_L1'
]

# 保存分析结果
output_file = os.path.join(BASE_DIR, "ml", "model_analysis.txt")
with open(output_file, 'w', encoding='utf-8') as f:
    f.write("=== 模型评估结果 ===\n")
    
    for model_name in target_models:
        print(f"\n评估模型: {model_name}")
        f.write(f"\n\n模型: {model_name}\n")
        
        try:
            # 获取预测结果
            y_pred = predictor.predict(test_data, model=model_name)
            
            # 计算评估指标
            rmse = np.sqrt(mean_squared_error(labels, y_pred))
            mae = mean_absolute_error(labels, y_pred)
            r2 = r2_score(labels, y_pred)
            pearson_corr, _ = pearsonr(labels, y_pred)
            
            # 打印评估指标
            print(f"RMSE: {rmse:.4f}")
            print(f"MAE: {mae:.4f}")
            print(f"R²: {r2:.4f}")
            print(f"皮尔逊相关系数: {pearson_corr:.4f}")
            
            # 保存评估指标
            f.write(f"RMSE: {rmse:.4f}\n")
            f.write(f"MAE: {mae:.4f}\n")
            f.write(f"R²: {r2:.4f}\n")
            f.write(f"皮尔逊相关系数: {pearson_corr:.4f}\n")
            
            # 保存预测结果对比
            f.write("\n预测值与真实值对比:\n")
            for true, pred in zip(labels, y_pred):
                f.write(f"真实值: {true:.2f}, 预测值: {pred:.2f}\n")
                
        except Exception as e:
            print(f"评估模型 {model_name} 时出错: {str(e)}")
            f.write(f"评估失败: {str(e)}\n")

print(f"\n分析结果已保存到: {output_file}")