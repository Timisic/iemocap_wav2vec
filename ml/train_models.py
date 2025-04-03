import numpy as np
import pandas as pd
import os
import warnings
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr
from sklearn.model_selection import KFold

# 过滤警告信息
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

def load_data(base_dir, label_file):
    """读取特征和标签数据"""
    print(f"\n=== 数据加载阶段 ===")
    print(f"读取标签文件: {label_file}")
    df = pd.read_excel(label_file, sheet_name='Sheet1')
    
    target_columns = ['分析力得分', '开放创新得分', '成就导向得分', '决策力得分', 
                     '压力承受得分', '推进执行得分', '影响力得分', '激励他人得分']
    
    # 标准化标签
    print("\n对标签进行标准化...")
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    df[target_columns] = scaler.fit_transform(df[target_columns])
    print("标签标准化完成")
    
    # 划分训练测试集
    print("\n按说话人ID划分训练测试集...")
    speaker_ids = [name.split('_')[0] for name in df['id']]
    unique_speakers = np.unique(speaker_ids)
    train_speakers, test_speakers = train_test_split(
        unique_speakers, test_size=0.2, random_state=42
    )
    print(f"训练集说话人数: {len(train_speakers)}")
    print(f"测试集说话人数: {len(test_speakers)}")
    
    print("\n=== 数据匹配情况 ===")
    print("标签ID\t\t特征文件\t\t各维度得分")
    print("-" * 80)
    
    features = []
    labels = []
    is_train = []
    
    # Then process the data
    for idx, row in df.iterrows():
        name = row['id']
        speaker_id = name.split('_')[0]
        scores = row[target_columns].values
        
        # 修改特征文件路径拼接方式
        feature_path = os.path.join(base_dir, f"{name}.npy")
        if os.path.exists(feature_path):
            feature = np.load(feature_path)
            feature = feature.reshape(-1)
            features.append(feature)
            labels.append(scores)
            is_train.append(speaker_id in train_speakers)
            
            # 打印匹配信息
            scores_str = " ".join([f"{score:.1f}" for score in scores])
            print(f"{name}\t{feature_path}\t{scores_str}")
    
    features = np.array(features)
    labels = np.array(labels)
    if task == 'classification':
        labels = labels.astype(int)  # 确保标签为整数类型
    is_train = np.array(is_train, dtype=bool)
    
    print(f"\n最终数据集:")
    print(f"- 特征矩阵形状: {features.shape}")
    print(f"- 标签矩阵形状: {labels.shape}")
    if task == 'classification':
        print("- 标签类别:", np.unique(labels))  # 打印所有唯一的类别
    print(f"- 训练集样本数: {sum(is_train)}")
    print(f"- 测试集样本数: {sum(~is_train)}")
    
    return features, labels, target_columns, is_train

def get_model(model_name):
    """获取指定的模型"""
    print(f"\n初始化 {model_name.upper()} 模型...")
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

def evaluate_predictions(y_true, y_pred):
    """评估预测结果"""
    return {
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'mae': mean_absolute_error(y_true, y_pred),
        'r2': r2_score(y_true, y_pred),
        'pearson': pearsonr(y_true, y_pred)[0]
    }

def train_and_evaluate_model(X, y, target_columns, is_train, model_names=['rf', 'xgb']):
    """使用五折交叉验证训练和评估模型"""
    print("\n=== 模型训练与评估阶段 ===")
    
    if isinstance(model_names, str):
        model_names = [model_names]
    
    # 划分数据集
    print("\n划分训练测试集...")
    X_train = X[is_train]
    X_test = X[~is_train]
    y_train = y[is_train]
    y_test = y[~is_train]
    print(f"训练集形状: {X_train.shape}")
    print(f"测试集形状: {X_test.shape}")
    
    # 创建5折交叉验证
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    all_results = {}
    
    for model_name in model_names:
        print(f"\n=== 使用 {model_name.upper()} 模型训练 ===")
        results = {}
        
        for i, target in enumerate(target_columns):
            print(f"\n训练 {target} 的模型...")
            
            fold_metrics = {'rmse': [], 'mae': [], 'r2': [], 'pearson': []}
            
            # 五折交叉验证
            for fold, (train_idx, val_idx) in enumerate(kf.split(X_train), 1):
                print(f"\n第 {fold} 折验证...")
                
                X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
                y_fold_train, y_fold_val = y_train[train_idx][:, i], y_train[val_idx][:, i]
                
                model = get_model(model_name)
                model.fit(X_fold_train, y_fold_train)
                y_fold_pred = model.predict(X_fold_val)
                
                # 计算评估指标
                metrics = evaluate_predictions(y_fold_val, y_fold_pred)
                for metric_name, value in metrics.items():
                    fold_metrics[metric_name].append(value)
            
            # 训练最终模型
            final_model = get_model(model_name)
            final_model.fit(X_train, y_train[:, i])
            y_pred = final_model.predict(X_test)
            
            test_metrics = evaluate_predictions(y_test[:, i], y_pred)
            
            # 计算交叉验证的平均指标
            cv_metrics = {
                metric: {
                    'mean': np.mean(values),
                    'std': np.std(values)
                }
                for metric, values in fold_metrics.items()
            }
            
            results[target] = {
                'model': final_model,
                'predictions': y_pred,
                'true_values': y_test[:, i],
                'test_metrics': test_metrics,
                'cv_metrics': cv_metrics,
                'feature_importance': final_model.feature_importances_ if hasattr(final_model, 'feature_importances_') else np.zeros(X.shape[1])
            }
            
            # 打印评估结果
            print(f"\n{target} 交叉验证结果:")
            for metric, stats in cv_metrics.items():
                print(f"{metric.upper()}: {stats['mean']:.4f} (±{stats['std']:.4f})")
            
            print(f"\n{target} 测试集结果:")
            for metric_name, value in test_metrics.items():
                print(f"{metric_name.upper()}: {value:.4f}")

def save_results(results, model_name, target_columns):
    """保存模型结果"""
    print("\n=== 保存模型结果 ===")
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    results_dir = os.path.join(BASE_DIR, "ml", "results", f"{model_name}")
    os.makedirs(results_dir, exist_ok=True)
    print(f"结果保存目录: {results_dir}")
    
    # 保存评估结果
    with open(os.path.join(results_dir, f"{model_name}_evaluation.txt"), 'w', encoding='utf-8') as f:
        f.write(f"{model_name.upper()} 模型评估结果\n")
        f.write("=" * 50 + "\n\n")
        
        for target in target_columns:
            f.write(f"\n{target}:\n")
            f.write("-" * 30 + "\n")
            
            # 添加交叉验证结果
            f.write("交叉验证结果:\n")
            for metric, stats in results[target]['cv_metrics'].items():
                f.write(f"{metric.upper()}: {stats['mean']:.4f} (±{stats['std']:.4f})\n")
            
            # 添加测试集结果
            f.write("\n测试集结果:\n")
            for metric_name, value in results[target]['test_metrics'].items():
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

if __name__ == "__main__":
    print("\n=== 开始模型训练流程 ===")
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # 配置路径
    base_dir = "ml/features/features_opensmile_pca_30"
    base_dir = os.path.join(BASE_DIR, base_dir)
    
    label_file = "src_competency/labels_2.xlsx"
    label_file = os.path.join(BASE_DIR, label_file)
    
    print(f"\n当前配置:")
    print(f"特征目录: {base_dir}")
    print(f"标签文件: {label_file}")
    
    confirm = input("\n确认开始训练? (y/n): ").lower()
    if confirm != 'y':
        print("已取消训练")
        exit()
    
    # 加载数据
    X, y, target_columns, is_train = load_data(base_dir, label_file)
    
    model_choice = 'rf' # 'rf' or 'xgb'
    
    # 训练模型
    results = train_and_evaluate_model(X, y, target_columns, is_train, model_choice)