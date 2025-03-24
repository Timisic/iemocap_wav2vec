import os
import time
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
import argparse

from config import Config
from data import IEMOCAPDataModule
from model import EmotionClassifier
from utils import EarlyStopping, MetricsTracker, compute_metrics, set_seed

class Trainer:
    def __init__(self, config=None, multi_task=False, train_id=None):
        """
        训练器
        
        Args:
            config: 配置对象
            multi_task: 是否使用多任务模型
            train_id: 训练ID（例如：'01'，'02'等）
        """
        if config is None:
            self.config = Config()
        else:
            self.config = config
            
        self.multi_task = multi_task
        self.train_id = train_id
        
        # 更新检查点和图片保存路径
        if train_id:
            self.checkpoint_dir = self.config.CHECKPOINT_DIR / f"train_{train_id}"
            self.figure_dir = self.config.FIGURE_DIR / f"train_{train_id}"
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
            self.figure_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.checkpoint_dir = self.config.CHECKPOINT_DIR
            self.figure_dir = self.config.FIGURE_DIR
            
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 设置随机种子
        set_seed(self.config.SEED)
        
        # 初始化数据模块
        self.data_module = IEMOCAPDataModule(self.config)
        self.data_module.prepare_data()
        
        # 初始化模型
        if multi_task:
            self.model = MultiTaskEmotionClassifier(self.config)
        else:
            self.model = EmotionClassifier(self.config)
            
        # 冻结特征提取器
        self.model.freeze_feature_extractor()
        
        # 移动模型到设备
        if self.config.USE_MULTI_GPU and torch.cuda.device_count() > 1:
            print(f"使用 {torch.cuda.device_count()} 个GPU进行训练")
            self.model = nn.DataParallel(self.model)
            
        self.model.to(self.device)
        
        # 初始化损失函数
        self.criterion = nn.CrossEntropyLoss()
        
        # 初始化优化器
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config.LEARNING_RATE,
            weight_decay=self.config.WEIGHT_DECAY
        )
        
        # 初始化学习率调度器
        warmup_scheduler = LinearLR(
            self.optimizer, 
            start_factor=0.1, 
            end_factor=1.0, 
            total_iters=int(self.config.NUM_EPOCHS * self.config.WARMUP_RATIO)
        )
        
        cosine_scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=self.config.NUM_EPOCHS - int(self.config.NUM_EPOCHS * self.config.WARMUP_RATIO),
            eta_min=1e-6
        )
        
        self.scheduler = SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[int(self.config.NUM_EPOCHS * self.config.WARMUP_RATIO)]
        )
        
        # 初始化早停策略
        self.early_stopping = EarlyStopping(
            patience=self.config.PATIENCE,
            min_delta=self.config.MIN_DELTA,
            mode='max'
        )
        
        # 初始化指标跟踪器（更新figure_dir）
        self.metrics_tracker = MetricsTracker(self.config, figure_dir=self.figure_dir)
        
        # 删除定期保存检查点的相关代码，只保存最佳模型
        self.best_model_path = None
        
    def train_epoch(self, epoch):
        """
        训练一个轮次
        
        Args:
            epoch: 当前轮次
            
        Returns:
            平均训练损失和指标
        """
        self.model.train()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        # 获取训练数据加载器
        train_loader = self.data_module.get_dataloader("train")
        
        # 进度条
        pbar = tqdm(train_loader, desc=f"轮次 {epoch+1}/{self.config.NUM_EPOCHS} [训练]")
        
        # 梯度累积计数器
        accumulation_steps = self.config.GRADIENT_ACCUMULATION_STEPS
        
        for i, batch in enumerate(pbar):
            # 将数据移动到设备
            input_values = batch["input_values"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)
            
            # 前向传播
            outputs = self.model(
                input_values, 
                attention_mask=attention_mask,
                labels=labels
            )
            
            # 获取损失
            loss = outputs['loss']
            loss = loss / accumulation_steps
            
            # 反向传播
            loss.backward()
            
            # 梯度累积
            if (i + 1) % accumulation_steps == 0 or (i + 1) == len(train_loader):
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.config.MAX_GRAD_NORM
                )
                
                # 更新参数
                self.optimizer.step()
                self.optimizer.zero_grad()
            
            # 更新进度条
            total_loss += loss.item() * accumulation_steps
            pbar.set_postfix({"loss": f"{total_loss / (i + 1):.4f}"})
            
            # 收集预测和标签
            all_preds.append(outputs['logits'].detach().cpu().numpy())
            all_labels.append(labels.detach().cpu().numpy())
            
        # 计算平均损失
        avg_loss = total_loss / len(train_loader)
        
        # 计算指标
        all_preds = np.vstack(all_preds)
        all_labels = np.concatenate(all_labels)
        metrics = compute_metrics(all_preds, all_labels)
        
        # 更新指标跟踪器
        self.metrics_tracker.update_train(avg_loss, metrics)
        
        return avg_loss, metrics
    
    def validate(self, epoch):
        """
        验证模型
        
        Args:
            epoch: 当前轮次
            
        Returns:
            平均验证损失和指标
        """
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        # 获取验证数据加载器
        val_loader = self.data_module.get_dataloader("val")
        
        # 进度条
        pbar = tqdm(val_loader, desc=f"轮次 {epoch+1}/{self.config.NUM_EPOCHS} [验证]")
        
        with torch.no_grad():
            for i, batch in enumerate(pbar):
                # 将数据移动到设备
                input_values = batch["input_values"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)
                
                # 前向传播
                if self.multi_task:
                    outputs = self.model(input_values, attention_mask)
                    logits = outputs["emotion_logits"]
                else:
                    logits = self.model(input_values, attention_mask)
                    
                # 计算损失
                loss = self.criterion(logits, labels)
                
                # 更新进度条
                total_loss += loss.item()
                pbar.set_postfix({"loss": f"{total_loss / (i + 1):.4f}"})
                
                # 收集预测和标签
                all_preds.append(logits.detach().cpu().numpy())
                all_labels.append(labels.detach().cpu().numpy())
                
        # 计算平均损失
        avg_loss = total_loss / len(val_loader)
        
        # 计算指标
        all_preds = np.vstack(all_preds)
        all_labels = np.concatenate(all_labels)
        metrics = compute_metrics(all_preds, all_labels)
        
        # 更新指标跟踪器
        self.metrics_tracker.update_val(avg_loss, metrics)
        
        return avg_loss, metrics
    
    def test(self):
        """
        测试模型
        
        Returns:
            测试指标
        """
        self.model.eval()
        all_preds = []
        all_labels = []
        
        # 获取测试数据加载器
        test_loader = self.data_module.get_dataloader("test")
        
        # 进度条
        pbar = tqdm(test_loader, desc="测试")
        
        with torch.no_grad():
            for batch in pbar:
                # 将数据移动到设备
                input_values = batch["input_values"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)
                
                # 前向传播
                if self.multi_task:
                    outputs = self.model(input_values, attention_mask)
                    logits = outputs["emotion_logits"]
                else:
                    logits = self.model(input_values, attention_mask)
                
                # 收集预测和标签
                all_preds.append(logits.detach().cpu().numpy())
                all_labels.append(labels.detach().cpu().numpy())
                
        # 计算指标
        all_preds = np.vstack(all_preds)
        all_labels = np.concatenate(all_labels)
        metrics = compute_metrics(all_preds, all_labels)
        
        # 绘制混淆矩阵
        pred_ids = np.argmax(all_preds, axis=1)
        self.metrics_tracker.plot_confusion_matrix(all_labels, pred_ids)
        
        return metrics
    
    def save_checkpoint(self, epoch, val_f1, is_best=False):
        """
        保存检查点（只保存最佳模型）
        """
        if not is_best:
            return None
            
        # 获取模型状态
        if isinstance(self.model, nn.DataParallel):
            model_state = self.model.module.state_dict()
        else:
            model_state = self.model.state_dict()
            
        # 构建检查点
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model_state,
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "val_f1": val_f1,
            "config": self.config.__dict__ if hasattr(self.config, "__dict__") else None
        }
        
        # 保存最佳模型
        model_type = "multi_task" if self.multi_task else "single_task"
        best_model_path = self.checkpoint_dir / f"{model_type}_best_model.pt"
        torch.save(checkpoint, best_model_path)
        self.best_model_path = best_model_path
            
        return best_model_path
    
    def load_checkpoint(self, checkpoint_path):
        """
        加载检查点
        
        Args:
            checkpoint_path: 检查点路径
        """
        # 加载检查点
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # 恢复模型状态
        if isinstance(self.model, nn.DataParallel):
            self.model.module.load_state_dict(checkpoint["model_state_dict"])
        else:
            self.model.load_state_dict(checkpoint["model_state_dict"])
            
        # 恢复优化器状态
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        # 恢复调度器状态
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        
        # 恢复训练状态
        self.current_epoch = checkpoint["epoch"] + 1
        self.best_val_f1 = checkpoint["val_f1"]
        
        print(f"已加载检查点: {checkpoint_path}")
        print(f"当前轮次: {self.current_epoch}, 最佳验证F1: {self.best_val_f1:.4f}")
    
    def train(self, resume_from=None):
        """
        训练模型
        
        Args:
            resume_from: 恢复训练的检查点路径
        """
        # 恢复训练
        if resume_from is not None:
            self.load_checkpoint(resume_from)
            
        # 记录开始时间
        start_time = time.time()
        
        # 训练循环
        for epoch in range(self.current_epoch, self.config.NUM_EPOCHS):
            # 训练一个轮次
            train_loss, train_metrics = self.train_epoch(epoch)
            
            # 验证
            val_loss, val_metrics = self.validate(epoch)
            
            # 更新学习率
            self.scheduler.step()
            
            # 打印指标
            print(f"轮次 {epoch+1}/{self.config.NUM_EPOCHS} - "
                  f"训练损失: {train_loss:.4f}, 训练准确率: {train_metrics['accuracy']:.4f}, 训练F1: {train_metrics['f1']:.4f} - "
                  f"验证损失: {val_loss:.4f}, 验证准确率: {val_metrics['accuracy']:.4f}, 验证F1: {val_metrics['f1']:.4f}")
            
            # 检查是否是最佳模型
            is_best = val_metrics['f1'] > self.best_val_f1
            if is_best:
                self.best_val_f1 = val_metrics['f1']
                
            # 保存检查点
            self.save_checkpoint(epoch, val_metrics['f1'], is_best)
            
            # 检查是否应该早停
            if self.config.EARLY_STOPPING and self.early_stopping(val_metrics['f1']):
                print(f"早停: 验证F1在{self.config.PATIENCE}轮内没有改善")
                break
                
        # 计算训练时间
        total_time = time.time() - start_time
        hours, remainder = divmod(total_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        print(f"训练完成! 总时间: {int(hours)}小时 {int(minutes)}分钟 {int(seconds)}秒")
        
        # 绘制训练曲线
        self.metrics_tracker.plot_all()
        
        # 加载最佳模型
        if self.best_model_path is not None:
            self.load_checkpoint(self.best_model_path)
            
        # 在测试集上评估
        test_metrics = self.test()
        
        # 打印测试指标
        print("测试指标:")
        for name, value in test_metrics.items():
            print(f"{name}: {value:.4f}")
            
        return test_metrics


if __name__ == "__main__":
    # 添加命令行参数解析
    parser = argparse.ArgumentParser(description='训练情感识别模型')
    parser.add_argument('--train_id', type=str, help='训练ID（例如：01，02等）')
    parser.add_argument('--multi_task', action='store_true', help='是否使用多任务模型')
    args = parser.parse_args()
    
    # 设置配置
    config = Config()
    
    # 创建训练器
    trainer = Trainer(config, multi_task=args.multi_task, train_id=args.train_id)
    
    # 检查是否有最佳模型检查点
    model_type = "multi_task" if args.multi_task else "single_task"
    best_model_path = trainer.checkpoint_dir / f"{model_type}_best_model.pt"
    
    if best_model_path.exists():
        print(f"找到最佳模型检查点: {best_model_path}")
        resume = input("是否恢复训练? (y/n): ").lower() == 'y'
        
        if resume:
            trainer.train(resume_from=best_model_path)
        else:
            trainer.train()
    else:
        trainer.train() 