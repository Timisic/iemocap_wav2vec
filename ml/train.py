import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy.stats import pearsonr
import os
import time
import warnings
from autogluon.tabular import TabularDataset, TabularPredictor
from autogluon.common.utils.utils import setup_outputdir
from autogluon.core.utils.loaders import load_pkl
from autogluon.core.utils.savers import save_pkl

# 过滤警告信息
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', module='sklearn.base')
warnings.filterwarnings('ignore', message='.*Unpickling.*')

# 读取特征和标签数据
def load_data(base_dir, label_file):
    # 过滤警告信息
    warnings.filterwarnings('ignore', category=UserWarning)
    warnings.filterwarnings('ignore', category=FutureWarning)
    warnings.filterwarnings('ignore', module='sklearn.base')
    warnings.filterwarnings('ignore', message='.*Unpickling.*')
    # 读取标签文件
    print(f"\n正在读取标签文件: {label_file}")
    df = pd.read_excel(label_file, sheet_name='Sheet1')
    print(f"标签数据形状: {df.shape}")
    print("列名:", df.columns.tolist())
    
    features = []
    labels = []
    
    # 定义所有需要预测的指标
    target_columns = ['分析力得分', '开放创新得分', '成就导向得分', '决策力得分', 
                     '压力承受得分', '推进执行得分', '影响力得分', '激励他人得分']
    
    # 遍历数据框
    for idx, row in df.iterrows():
        name = row['id']
        # 获取所有目标分数
        scores = row[target_columns].values
        
        feature_path = os.path.join(base_dir, f"{name}.npy")
        if os.path.exists(feature_path):
            feature = np.load(feature_path)
            feature = feature.reshape(-1)
            features.append(feature)
            labels.append(scores)
    
    features = np.array(features)
    labels = np.array(labels)
    print(f"\n最终数据集:")
    print(f"- 特征矩阵形状: {features.shape}")
    print(f"- 标签矩阵形状: {labels.shape}")
    
    return features, labels, target_columns

def train_model_with_autogluon(X, y, target_columns, time_limit=4800):
    """使用 AutoGluon 自动训练多目标回归模型"""
    if X.shape[0] == 0:
        raise ValueError("数据集为空！请检查数据加载过程。")
    
    # 准备数据
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # 转换为 DataFrame 格式
    feature_names = [f'feature_{i}' for i in range(X.shape[1])]
    train_data = pd.DataFrame(X_train, columns=feature_names)
    test_data = pd.DataFrame(X_test, columns=feature_names)
    
    # 添加所有目标变量
    for i, col in enumerate(target_columns):
        train_data[col] = y_train[:, i]
        test_data[col] = y_test[:, i]
    
    print("\n开始 AutoGluon 自动训练...")
    start_time = time.time()
    
    # 创建多标签预测器
    predictor = MultilabelPredictor(
        labels=target_columns,
        path='autogluon_models_multi',
        problem_types=['regression'] * len(target_columns),
        eval_metrics=['root_mean_squared_error'] * len(target_columns)
    )
    
    # 训练模型
    predictor.fit(
        train_data=train_data,
        time_limit=time_limit,
        num_gpus=1,
        ag_args_fit={'num_gpus':1},
        hyperparameters={
            'GBM': {'num_gpus': 1},
            'CAT': {'num_gpus': 1},
            'XGB': {'num_gpus': 1},
            'NN_TORCH': {'num_gpus': 1},
        },
        excluded_model_types=['KNN', 'FASTAI'],
        presets='best_quality'
    )
    
    # 预测并评估
    results = {}
    
    # 创建结果目录
    results_dir = os.path.join(BASE_DIR, "ml", "evaluation_results")
    os.makedirs(results_dir, exist_ok=True)
    
    for target in target_columns:
        # 获取该维度的预测器
        target_predictor = predictor.get_predictor(target)
        best_model_name = target_predictor.model_best
        
        # 获取所有可用模型
        all_models = target_predictor.model_names()
        print(f"\n{target} 可用模型: {all_models}")
        
        # 创建该维度的评估结果文件
        result_file = os.path.join(results_dir, f"{target}_evaluation.txt")
        with open(result_file, 'w', encoding='utf-8') as f:
            f.write(f"{target} 模型评估结果\n")
            f.write("="*50 + "\n\n")
            
            # 存储每个模型的性能指标
            model_metrics = {}
            
            # 对每个模型进行评估
            for model_name in all_models:
                f.write(f"\n模型: {model_name}\n")
                f.write("-"*30 + "\n")
                
                # 使用当前模型进行预测
                y_pred_model = target_predictor.predict(test_data, model=model_name)
                
                # 计算评估指标
                rmse = np.sqrt(mean_squared_error(test_data[target], y_pred_model))
                mae = mean_absolute_error(test_data[target], y_pred_model)
                r2 = r2_score(test_data[target], y_pred_model)
                pearson_corr, _ = pearsonr(test_data[target], y_pred_model)
                
                # 保存指标
                model_metrics[model_name] = {
                    'rmse': rmse,
                    'mae': mae,
                    'r2': r2,
                    'pearson': pearson_corr
                }
                
                # 写入评估指标
                f.write(f"RMSE: {rmse:.4f}\n")
                f.write(f"MAE: {mae:.4f}\n")
                f.write(f"R²: {r2:.4f}\n")
                f.write(f"皮尔逊相关系数: {pearson_corr:.4f}\n\n")
            
            # 标记最佳模型
            f.write("\n\n最佳模型性能比较\n")
            f.write("="*30 + "\n")
            f.write(f"AutoGluon选择的最佳模型: {best_model_name}\n\n")
            
            # 按R²排序所有模型
            sorted_models = sorted(model_metrics.items(), key=lambda x: x[1]['r2'], reverse=True)
            f.write("所有模型按R²排序:\n")
            for i, (model_name, metrics) in enumerate(sorted_models):
                f.write(f"{i+1}. {model_name}: R²={metrics['r2']:.4f}, RMSE={metrics['rmse']:.4f}, MAE={metrics['mae']:.4f}, Pearson={metrics['pearson']:.4f}\n")
            
            # 使用R²最佳的模型进行详细评估
            best_r2_model = sorted_models[0][0]
            f.write(f"\n\nR²最佳模型 ({best_r2_model}) 详细评估:\n")
            f.write("-"*30 + "\n")
            
            # 使用R²最佳模型进行预测
            y_pred_best = target_predictor.predict(test_data, model=best_r2_model)
            
            # 添加预测结果详情
            test_results = pd.DataFrame({
                '真实值': test_data[target],
                '预测值': y_pred_best,
                '误差': np.abs(test_data[target] - y_pred_best)
            })
            f.write(test_results.to_string() + "\n\n")
            f.write("统计摘要:\n")
            f.write(test_results.describe().to_string())
        
        # 使用R²最佳的模型作为该维度的最终结果
        best_r2_model = sorted(model_metrics.items(), key=lambda x: x[1]['r2'], reverse=True)[0][0]
        y_pred_best = target_predictor.predict(test_data, model=best_r2_model)
        
        print(f"\n{target} R²最佳模型: {best_r2_model}")
        print(f"RMSE: {model_metrics[best_r2_model]['rmse']:.4f}")
        print(f"MAE: {model_metrics[best_r2_model]['mae']:.4f}")
        print(f"R²: {model_metrics[best_r2_model]['r2']:.4f}")
        print(f"皮尔逊相关系数: {model_metrics[best_r2_model]['pearson']:.4f}")
        
        results[target] = {
            'test_data': test_data[target],
            'predictions': y_pred_best,
            'best_model': best_r2_model,
            'metrics': model_metrics[best_r2_model]
        }
    
    print(f"\n总训练时间: {(time.time() - start_time):.2f} 秒")
    
    return predictor, results

class MultilabelPredictor:
    """ Tabular Predictor for predicting multiple columns in table.
        Creates multiple TabularPredictor objects which you can also use individually.
        You can access the TabularPredictor for a particular label via: `multilabel_predictor.get_predictor(label_i)`
    """

    multi_predictor_file = 'multilabel_predictor.pkl'

    def __init__(self, labels, path=None, problem_types=None, eval_metrics=None, consider_labels_correlation=True, **kwargs):
        if len(labels) < 2:
            raise ValueError("MultilabelPredictor is only intended for predicting MULTIPLE labels (columns), use TabularPredictor for predicting one label (column).")
        if (problem_types is not None) and (len(problem_types) != len(labels)):
            raise ValueError("If provided, `problem_types` must have same length as `labels`")
        if (eval_metrics is not None) and (len(eval_metrics) != len(labels)):
            raise ValueError("If provided, `eval_metrics` must have same length as `labels`")
        self.path = setup_outputdir(path, warn_if_exist=False)
        self.labels = labels
        self.consider_labels_correlation = consider_labels_correlation
        self.predictors = {}  # key = label, value = TabularPredictor or str path to the TabularPredictor for this label
        if eval_metrics is None:
            self.eval_metrics = {}
        else:
            self.eval_metrics = {labels[i] : eval_metrics[i] for i in range(len(labels))}
        problem_type = None
        eval_metric = None
        for i in range(len(labels)):
            label = labels[i]
            path_i = os.path.join(self.path, "Predictor_" + str(label))
            if problem_types is not None:
                problem_type = problem_types[i]
            if eval_metrics is not None:
                eval_metric = eval_metrics[i]
            self.predictors[label] = TabularPredictor(label=label, problem_type=problem_type, eval_metric=eval_metric, path=path_i, **kwargs)

    def fit(self, train_data, tuning_data=None, **kwargs):
        """ Fits a separate TabularPredictor to predict each of the labels."""
        if isinstance(train_data, str):
            train_data = TabularDataset(train_data)
        if tuning_data is not None and isinstance(tuning_data, str):
            tuning_data = TabularDataset(tuning_data)
        train_data_og = train_data.copy()
        if tuning_data is not None:
            tuning_data_og = tuning_data.copy()
        else:
            tuning_data_og = None
        save_metrics = len(self.eval_metrics) == 0
        for i in range(len(self.labels)):
            label = self.labels[i]
            predictor = self.get_predictor(label)
            if not self.consider_labels_correlation:
                labels_to_drop = [l for l in self.labels if l != label]
            else:
                labels_to_drop = [self.labels[j] for j in range(i+1, len(self.labels))]
            train_data = train_data_og.drop(labels_to_drop, axis=1)
            if tuning_data is not None:
                tuning_data = tuning_data_og.drop(labels_to_drop, axis=1)
            print(f"训练标签 {label} 的模型...")
            predictor.fit(train_data=train_data, tuning_data=tuning_data, **kwargs)
            self.predictors[label] = predictor.path
            if save_metrics:
                self.eval_metrics[label] = predictor.eval_metric
        self.save()

    def predict(self, data, **kwargs):
        """ Returns DataFrame with label columns containing predictions for each label."""
        if isinstance(data, str):
            data = TabularDataset(data)
        data = data.copy()
        predictions = pd.DataFrame(index=data.index)
        for i in range(len(self.labels)):
            label = self.labels[i]
            predictor = self.get_predictor(label)
            predictions[label] = predictor.predict(data, **kwargs)
            if self.consider_labels_correlation and i < len(self.labels)-1:
                data[label] = predictions[label]
        return predictions

    def predict_proba(self, data, **kwargs):
        """ Returns dict where each key is a label and each value is the probability prediction for that label."""
        if isinstance(data, str):
            data = TabularDataset(data)
        data = data.copy()
        predictions = {}
        for i in range(len(self.labels)):
            label = self.labels[i]
            predictor = self.get_predictor(label)
            predictions[label] = predictor.predict_proba(data, **kwargs)
            if self.consider_labels_correlation and i < len(self.labels)-1:
                data[label] = predictor.predict(data, **kwargs)
        return predictions

    def evaluate(self, data, **kwargs):
        """ Returns dict where each key is a label and each value is the evaluation performance for that label."""
        if isinstance(data, str):
            data = TabularDataset(data)
        performance = {}
        for label in self.labels:
            predictor = self.get_predictor(label)
            performance[label] = predictor.evaluate(data, **kwargs)
        return performance

    def save(self):
        """ Save MultilabelPredictor to disk."""
        for label in self.labels:
            if not isinstance(self.predictors[label], str):
                self.predictors[label] = self.predictors[label].path
        save_pkl.save(path=os.path.join(self.path, self.multi_predictor_file), object=self)
        print(f"MultilabelPredictor saved to: {self.path}")

    @classmethod
    def load(cls, path):
        """ Load MultilabelPredictor from disk."""
        path = os.path.expanduser(path)
        if path[-1] == os.path.sep:
            path = path[:-1]
        return load_pkl.load(path=os.path.join(path, cls.multi_predictor_file))

    def get_predictor(self, label):
        """ Returns TabularPredictor which predicts only this label."""
        predictor = self.predictors[label]
        if isinstance(predictor, str):
            return TabularPredictor.load(path=predictor)
        else:
            return predictor

    def get_model_best(self, label):
        """ Returns name of the best model for a particular label."""
        return self.get_predictor(label).model_best

    def get_models_with_labels(self):
        """ Returns dict where keys are labels and values are the best model names for each label."""
        models = {}
        for label in self.labels:
            models[label] = self.get_model_best(label)
        return models

    def feature_importance(self, data=None):
        """ Returns dict where each key is a label and each value is the feature importance for that label."""
        feature_importance = {}
        for label in self.labels:
            predictor = self.get_predictor(label)
            feature_importance[label] = predictor.feature_importance(data)
        return feature_importance

if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    base_dir = os.path.join(BASE_DIR, "ml", "features_pca1")
    label_file = os.path.join(BASE_DIR, "src_competency", "labels_2.xlsx")
    
    # 加载数据
    X, y, target_columns = load_data(base_dir, label_file)
    
    # 训练模型
    predictors, results = train_model_with_autogluon(X, y, target_columns)
    
    # 显示所有目标的预测结果
    print("\n各能力维度预测结果:")
    for target in target_columns:
        print(f"\n{target}:")
        test_results = pd.DataFrame({
            '真实值': results[target]['test_data'],
            '预测值': results[target]['predictions'],
            '误差': np.abs(results[target]['test_data'] - results[target]['predictions'])
        })
        print(test_results.describe())
