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

# 添加新的导入
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

def load_data(base_dir, label_file, task='regression'):
    """读取特征和标签数据"""
    print(f"\n正在读取标签文件: {label_file}")
    df = pd.read_excel(label_file, sheet_name='Sheet1')
    
    # 根据任务类型选择不同的目标列名
    if task == 'regression':
        target_columns = ['分析力得分', '开放创新得分', '成就导向得分', '决策力得分', 
                         '压力承受得分', '推进执行得分', '影响力得分', '激励他人得分']
    else:
        target_columns = ['分析力等级', '开放创新等级', '成就导向等级', '决策力等级', 
                         '压力承受等级', '推进执行等级', '影响力等级', '激励他人等级']
        # 对类别标签进行编码
        level_mapping = {
            '优秀': 4,
            '良好': 3,
            '中等': 2,
            '欠佳': 1,
            '不足': 0
        }
        # 将所有等级列转换为数值
        for col in target_columns:
            df[col] = df[col].map(level_mapping).astype(int)  # 确保转换为整数类型
    
    # First, get all speaker IDs and create train/test split
    speaker_ids = [name.split('_')[0] for name in df['id']]
    unique_speakers = np.unique(speaker_ids)
    train_speakers, test_speakers = train_test_split(
        unique_speakers, test_size=0.2, random_state=42
    )
    
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

def get_model(model_name, task='regression'):
    """获取指定的模型"""
    models = {
        'regression': {
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
        },
        'classification': {
            'rf': RandomForestClassifier(
                n_estimators=300,
                max_depth=None,
                min_samples_split=2,
                min_samples_leaf=1,
                random_state=42,
                n_jobs=-1
            ),
            'xgb': XGBClassifier(
                n_estimators=1000,
                learning_rate=0.1,
                max_depth=6,
                random_state=42,
                n_jobs=-1
            )
        }
    }
    return models[task].get(model_name.lower())

def evaluate_predictions(y_true, y_pred, task='regression'):
    """评估预测结果"""
    if task == 'regression':
        return {
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred),
            'pearson': pearsonr(y_true, y_pred)[0]
        }
    else:
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'report': classification_report(y_true, y_pred, output_dict=True)
        }

def train_and_evaluate_model(X, y, target_columns, is_train, model_names=['rf', 'xgb'], task='regression'):
    """使用五折交叉验证训练和评估模型"""
    if isinstance(model_names, str):
        model_names = [model_names]
    
    # 使用is_train划分训练集和测试集
    X_train = X[is_train]
    X_test = X[~is_train]
    y_train = y[is_train]
    y_test = y[~is_train]
    
    # 创建5折交叉验证
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    all_results = {}
    
    for model_name in model_names:
        print(f"\n使用 {model_name.upper()} 模型进行训练...")
        results = {}
        
        for i, target in enumerate(target_columns):
            print(f"\n训练 {target} 的模型...")
            
            # 修改评估指标部分
            fold_metrics = {
                'regression': {'rmse': [], 'mae': [], 'r2': [], 'pearson': []},
                'classification': {'accuracy': [], 'f1': [], 'precision': [], 'recall': []}
            }[task]
            
            # 五折交叉验证
            for fold, (train_idx, val_idx) in enumerate(kf.split(X_train), 1):
                print(f"\n第 {fold} 折验证...")
                
                # 获取当前折的训练和验证数据
                X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
                y_fold_train, y_fold_val = y_train[train_idx][:, i], y_train[val_idx][:, i]
                
                # 训练模型
                model = get_model(model_name, task)  # 添加task参数
                model.fit(X_fold_train, y_fold_train)
                y_fold_pred = model.predict(X_fold_val)
                
                # 根据任务类型计算不同的评估指标
                if task == 'regression':
                    fold_metrics['rmse'].append(np.sqrt(mean_squared_error(y_fold_val, y_fold_pred)))
                    fold_metrics['mae'].append(mean_absolute_error(y_fold_val, y_fold_pred))
                    fold_metrics['r2'].append(r2_score(y_fold_val, y_fold_pred))
                    fold_metrics['pearson'].append(pearsonr(y_fold_val, y_fold_pred)[0])
                else:
                    fold_metrics['accuracy'].append(accuracy_score(y_fold_val, y_fold_pred))
                    report = classification_report(y_fold_val, y_fold_pred, output_dict=True, zero_division=0)
                    fold_metrics['f1'].append(report['weighted avg']['f1-score'])
                    fold_metrics['precision'].append(report['weighted avg']['precision'])
                    fold_metrics['recall'].append(report['weighted avg']['recall'])
            
            # 训练最终模型（使用全部训练数据）
            final_model = get_model(model_name, task)  # 添加task参数
            final_model.fit(X_train, y_train[:, i])
            y_pred = final_model.predict(X_test)
            
            # 修改评估方法
            test_metrics = evaluate_predictions(y_test[:, i], y_pred, task)
            
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
            if task == 'regression':
                for metric_name, value in test_metrics.items():
                    print(f"{metric_name.upper()}: {value:.4f}")
            else:
                print(f"ACCURACY: {test_metrics['accuracy']:.4f}")
                print("\n分类报告:")
                report = test_metrics['report']
                for label in sorted(set(y_test[:, i])):
                    metrics = report[str(label)]
                    print(f"\n类别 {label}:")
                    print(f"Precision: {metrics['precision']:.4f}")
                    print(f"Recall: {metrics['recall']:.4f}")
                    print(f"F1-score: {metrics['f1-score']:.4f}")
                print("\n加权平均:")
                weighted_avg = report['weighted avg']
                print(f"Precision: {weighted_avg['precision']:.4f}")
                print(f"Recall: {weighted_avg['recall']:.4f}")
                print(f"F1-score: {weighted_avg['f1-score']:.4f}")

def save_results(results, model_name, target_columns, task='regression'):
    """保存模型结果"""
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    results_dir = os.path.join(BASE_DIR, "ml","results", f"{model_name}")
    os.makedirs(results_dir, exist_ok=True)
    
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
            if task == 'regression':
                for metric_name, value in results[target]['test_metrics'].items():
                    f.write(f"{metric_name.upper()}: {value:.4f}\n")
            else:
                test_metrics = results[target]['test_metrics']
                f.write(f"ACCURACY: {test_metrics['accuracy']:.4f}\n")
                f.write("\n分类报告:\n")
                report = test_metrics['report']
                for label in sorted(set(results[target]['true_values'])):
                    metrics = report[str(label)]
                    f.write(f"\n类别 {label}:\n")
                    f.write(f"Precision: {metrics['precision']:.4f}\n")
                    f.write(f"Recall: {metrics['recall']:.4f}\n")
                    f.write(f"F1-score: {metrics['f1-score']:.4f}\n")
                f.write("\n加权平均:\n")
                weighted_avg = report['weighted avg']
                f.write(f"Precision: {weighted_avg['precision']:.4f}\n")
                f.write(f"Recall: {weighted_avg['recall']:.4f}\n")
                f.write(f"F1-score: {weighted_avg['f1-score']:.4f}\n")
            
            # 预测结果统计
            if task == 'regression':
                predictions = pd.DataFrame({
                    '真实值': results[target]['true_values'],
                    '预测值': results[target]['predictions'],
                    '误差': np.abs(results[target]['true_values'] - results[target]['predictions'])
                })
                f.write("\n预测结果统计:\n")
                f.write(predictions.describe().to_string())
                f.write("\n\n")
            else:
                # 对于分类任务，添加混淆矩阵
                cm = confusion_matrix(results[target]['true_values'], 
                                    results[target]['predictions'])
                f.write("\n混淆矩阵:\n")
                f.write(str(cm))
                f.write("\n\n")

if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # 指定任务类型
    task = 'regression'  # 'regression'或 'classification'
    
    # 根据任务类型选择特征目录
    base_dir = os.path.join(BASE_DIR, "ml", "features", "features_opensmile_pca_30")
    label_file = os.path.join(BASE_DIR, "src_competency", "labels_2.xlsx")
    
    # 加载数据
    X, y, target_columns, is_train = load_data(base_dir, label_file, task)
    
    # 可以选择单个模型或多个模型
    model_names = ['rf']  # 或者 model_names = 'rf' 只使用随机森林
    results = train_and_evaluate_model(X, y, target_columns, is_train, model_names, task)