import numpy as np
import pandas as pd
import os
import warnings
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr

# 过滤警告信息
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

def get_model(model_name):
    """获取指定的模型"""
    models = {
        'rf': RandomForestRegressor(
            n_estimators=300,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=42,
            n_jobs=-1
        ),
        'xgb': XGBRegressor(
            n_estimators=1000,
            learning_rate=0.1,
            max_depth=6,
            random_state=42,
            n_jobs=-1
        )
    }
    return models.get(model_name.lower())

def train_and_evaluate_model(X, y, target_columns, is_train, model_names=['rf', 'xgb']):
    """训练和评估指定的模型"""
    if isinstance(model_names, str):
        model_names = [model_names]
    
    X_train = X[is_train]
    X_test = X[~is_train]
    y_train = y[is_train]
    y_test = y[~is_train]
    
    all_results = {}
    
    for model_name in model_names:
        print(f"\n使用 {model_name.upper()} 模型进行训练...")
        model = get_model(model_name)
        if model is None:
            print(f"未知模型: {model_name}")
            continue
            
        results = {}
        for i, target in enumerate(target_columns):
            print(f"\n训练 {target} 的模型...")
            
            # 训练模型
            model.fit(X_train, y_train[:, i])
            y_pred = model.predict(X_test)
            
            # 计算评估指标
            metrics = {
                'rmse': np.sqrt(mean_squared_error(y_test[:, i], y_pred)),
                'mae': mean_absolute_error(y_test[:, i], y_pred),
                'r2': r2_score(y_test[:, i], y_pred),
                'pearson': pearsonr(y_test[:, i], y_pred)[0]
            }
            
            # 特征重要性
            importance = (model.feature_importances_ if hasattr(model, 'feature_importances_')
                        else np.zeros(X.shape[1]))
            
            results[target] = {
                'model': model,
                'predictions': y_pred,
                'true_values': y_test[:, i],
                'metrics': metrics,
                'feature_importance': importance
            }
            
            # 打印评估结果
            print(f"\n{target} 评估结果:")
            for metric_name, value in metrics.items():
                print(f"{metric_name.upper()}: {value:.4f}")
        
        all_results[model_name] = results
        
        # 保存当前模型的结果
        save_results(all_results[model_name], model_name, target_columns)
    
    return all_results

def save_results(results, model_name, target_columns):
    """保存模型结果"""
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    results_dir = os.path.join(BASE_DIR, "ml", f"{model_name}_results")
    os.makedirs(results_dir, exist_ok=True)
    
    # 保存评估结果
    with open(os.path.join(results_dir, f"{model_name}_evaluation.txt"), 'w', encoding='utf-8') as f:
        f.write(f"{model_name.upper()} 模型评估结果\n")
        f.write("=" * 50 + "\n\n")
        
        for target in target_columns:
            metrics = results[target]['metrics']
            f.write(f"\n{target}:\n")
            f.write("-" * 30 + "\n")
            for metric_name, value in metrics.items():
                f.write(f"{metric_name.upper()}: {value:.4f}\n")
            
            # 预测结果统计
            predictions = pd.DataFrame({
                '真实值': results[target]['true_values'],
                '预测值': results[target]['predictions'],
                '误差': np.abs(results[target]['true_values'] - results[target]['predictions'])
            })
            f.write("\n预测结果统计:\n")
            f.write(predictions.describe().to_string())
            f.write("\n\n")
            
            # 保存特征重要性
            importance_df = pd.DataFrame({
                'feature': [f'feature_{i}' for i in range(len(results[target]['feature_importance']))],
                'importance': results[target]['feature_importance']
            }).sort_values('importance', ascending=False)
            
            importance_file = os.path.join(results_dir, f"{target}_feature_importance.csv")
            importance_df.to_csv(importance_file, index=False)

if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    base_dir = os.path.join(BASE_DIR, "ml", "features_pca_30")
    label_file = os.path.join(BASE_DIR, "src_competency", "labels_2.xlsx")
    
    # 加载数据
    X, y, target_columns, is_train = load_data(base_dir, label_file)
    
    # 可以选择单个模型或多个模型
    model_names = ['rf', 'xgb']  # 或者 model_names = 'rf' 只使用随机森林
    results = train_and_evaluate_model(X, y, target_columns, is_train, model_names)