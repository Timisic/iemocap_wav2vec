import os
import warnings
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr
from autogluon.tabular import TabularPredictor
from tqdm import tqdm

# 过滤警告信息
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', module='sklearn.base')
warnings.filterwarnings('ignore', message='.*Unpickling.*')

# 设置随机种子
np.random.seed(42)

def load_and_process_data(base_dir, label_file):
    """加载并处理数据，与train.py保持一致"""
    print(f"\n正在读取标签文件: {label_file}")
    df = pd.read_excel(label_file, sheet_name='Sheet1')
    
    features = []
    labels = []
    
    target_columns = ['分析力得分', '开放创新得分', '成就导向得分', '决策力得分', 
                     '压力承受得分', '推进执行得分', '影响力得分', '激励他人得分']
    
    for idx, row in tqdm(df.iterrows(), desc="加载数据"):
        name = row['id']
        scores = row[target_columns].values
        
        feature_path = os.path.join(base_dir, f"{name}.npy")
        if os.path.exists(feature_path):
            feature = np.load(feature_path)
            feature = feature.reshape(-1)
            features.append(feature)
            labels.append(scores)
    
    features = np.array(features)
    labels = np.array(labels)
    
    print(f"\n数据集大小:")
    print(f"- 特征矩阵形状: {features.shape}")
    print(f"- 标签矩阵形状: {labels.shape}")
    
    return features, labels, target_columns

def evaluate_models(X, y, target_columns, results_dir):
    """评估所有模型的性能"""
    # 准备数据
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # 转换为DataFrame格式
    feature_names = [f'feature_{i}' for i in range(X.shape[1])]
    test_data = pd.DataFrame(X_test, columns=feature_names)
    
    results = {}
    for i, target in enumerate(target_columns):
        print(f"\n评估目标: {target}")
        
        # 加载对应的预测器
        predictor_path = os.path.join(BASE_DIR, "autogluon_models_multi", f"Predictor_{target}")
        predictor = TabularPredictor.load(predictor_path)
        
        # 获取所有可用模型
        all_models = predictor.model_names()
        model_metrics = []
        
        for model_name in tqdm(all_models, desc="评估模型"):
            try:
                # 使用当前模型进行预测
                y_pred = predictor.predict(test_data, model=model_name)
                
                # 计算评估指标
                rmse = np.sqrt(mean_squared_error(y_test[:, i], y_pred))
                mae = mean_absolute_error(y_test[:, i], y_pred)
                r2 = r2_score(y_test[:, i], y_pred)
                pearson_corr, _ = pearsonr(y_test[:, i], y_pred)
                
                model_metrics.append({
                    'model_name': model_name,
                    'rmse': rmse,
                    'mae': mae,
                    'r2': r2,
                    'pearson': pearson_corr
                })
            except Exception as e:
                print(f"警告: 评估模型 {model_name} 时出错: {str(e)}")
        
        # 按R²排序
        model_metrics.sort(key=lambda x: x['r2'], reverse=True)
        results[target] = model_metrics
        
        # 保存为CSV
        df_metrics = pd.DataFrame(model_metrics)
        csv_file = os.path.join(results_dir, f"{target}_metrics.csv")
        df_metrics.to_csv(csv_file, index=False)
        
        # 保存为TXT
        txt_file = os.path.join(results_dir, f"{target}_metrics.txt")
        with open(txt_file, 'w', encoding='utf-8') as f:
            f.write(f"{target} 模型评估结果\n")
            f.write("="*50 + "\n\n")
            
            for i, metrics in enumerate(model_metrics, 1):
                f.write(f"{i}. {metrics['model_name']}\n")
                f.write(f"   RMSE: {metrics['rmse']:.4f}\n")
                f.write(f"   MAE: {metrics['mae']:.4f}\n")
                f.write(f"   R²: {metrics['r2']:.4f}\n")
                f.write(f"   皮尔逊相关系数: {metrics['pearson']:.4f}\n\n")
    
    return results

if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    features_dir = os.path.join(BASE_DIR, "ml", "features_pca2")
    label_file = os.path.join(BASE_DIR, "src_competency", "labels_2.xlsx")
    results_dir = os.path.join(BASE_DIR, "ml", "model_evaluation")
    os.makedirs(results_dir, exist_ok=True)
    
    # 加载数据
    X, y, target_columns = load_and_process_data(features_dir, label_file)
    
    # 评估模型
    results = evaluate_models(X, y, target_columns, results_dir)
    
    # 打印总结
    print("\n评估完成！结果已保存到:", results_dir)
    for target in target_columns:
        best_model = results[target][0]
        print(f"\n{target} 最佳模型:")
        print(f"模型: {best_model['model_name']}")
        print(f"R²: {best_model['r2']:.4f}")
        print(f"RMSE: {best_model['rmse']:.4f}")
        print(f"MAE: {best_model['mae']:.4f}")
        print(f"皮尔逊相关系数: {best_model['pearson']:.4f}")