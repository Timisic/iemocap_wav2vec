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
    def __init__(self, config=None, train_id=None):
        """
        训练器
        
        Args:
            config: 配置对象
            train_id: 训练ID（例如：'01'，'02'等）
        """
        if config is None:
            self.config = Config()
        else:
            self.config = config
            
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
        
        # 添加训练状态初始化
        self.current_epoch = 0
        self.best_val_f1 = 0.0
        
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
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.config.NUM_EPOCHS} [Train]")
        
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
                attention_mask=attention_mask
            )
            
            # 计算损失
            if isinstance(self.model, nn.DataParallel):
                if isinstance(outputs, dict):
                    logits = outputs['logits']
                else:
                    logits = outputs
                loss = self.criterion(logits, labels)
            else:
                if isinstance(outputs, dict):
                    logits = outputs['logits']
                    loss = outputs['loss']
                else:
                    logits = outputs
                    loss = self.criterion(logits, labels)
            
            # 梯度累积
            loss = loss / accumulation_steps
            loss.backward()
            
            if (i + 1) % accumulation_steps == 0 or (i + 1) == len(train_loader):
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.config.MAX_GRAD_NORM
                )
                self.optimizer.step()
                self.optimizer.zero_grad()
            
            # 更新进度条
            total_loss += loss.item() * accumulation_steps
            pbar.set_postfix({"loss": f"{total_loss / (i + 1):.4f}"})
            
            # 收集预测和标签
            all_preds.append(logits.detach().cpu().numpy())
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
        """
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        val_loader = self.data_module.get_dataloader("val")
        pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{self.config.NUM_EPOCHS} [Valid]")
        
        with torch.no_grad():
            for i, batch in enumerate(pbar):
                input_values = batch["input_values"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)
                
                # 前向传播
                outputs = self.model(
                    input_values, 
                    attention_mask=attention_mask
                )
                
                # 计算损失
                if isinstance(self.model, nn.DataParallel):
                    # 对于多GPU，在计算损失之前先收集所有GPU的输出
                    if isinstance(outputs, dict):
                        logits = outputs['logits']
                    else:
                        logits = outputs
                    loss = self.criterion(logits, labels)
                else:
                    if isinstance(outputs, dict):
                        logits = outputs['logits']
                        loss = outputs['loss']
                    else:
                        logits = outputs
                        loss = self.criterion(logits, labels)
                
                # 更新进度条和累积损失
                total_loss += loss.item()
                current_loss = total_loss / (i + 1)
                pbar.set_postfix({"loss": f"{current_loss:.4f}"})
                
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
        
        pbar = tqdm(test_loader, desc="Test")
        
        with torch.no_grad():
            for batch in pbar:
                # 将数据移动到设备
                input_values = batch["input_values"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)
                
                # 前向传播
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
        best_model_path = self.checkpoint_dir / "best_model.pt"
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
        
        print(f"Loaded checkpoint: {checkpoint_path}")
        print(f"Current epoch: {self.current_epoch}, Best validation F1: {self.best_val_f1:.4f}")
    
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
            train_loss, train_metrics = self.train_epoch(epoch)
            val_loss, val_metrics = self.validate(epoch)
            
            # 更新学习率，不传递epoch参数
            self.scheduler.step()
            
            # 打印指标
            print(f"Epoch {epoch+1}/{self.config.NUM_EPOCHS} - "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_metrics['accuracy']:.4f}, Train F1: {train_metrics['f1']:.4f} - "
                  f"Valid Loss: {val_loss:.4f}, Valid Acc: {val_metrics['accuracy']:.4f}, Valid F1: {val_metrics['f1']:.4f}")
            
            # 检查是否是最佳模型
            is_best = val_metrics['f1'] > self.best_val_f1
            if is_best:
                self.best_val_f1 = val_metrics['f1']
                
            # 保存检查点
            self.save_checkpoint(epoch, val_metrics['f1'], is_best)
            
            # 检查是否应该早停
            if self.config.EARLY_STOPPING and self.early_stopping(val_metrics['f1']):
                print(f"Early stopping: No improvement in validation F1 for {self.config.PATIENCE} epochs")
                break
                
        # 计算训练时间
        total_time = time.time() - start_time
        hours, remainder = divmod(total_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        print(f"Training completed! Total time: {int(hours)}h {int(minutes)}m {int(seconds)}s")
        
        # 绘制训练曲线
        self.metrics_tracker.plot_all()
        
        # 加载最佳模型
        if self.best_model_path is not None:
            self.load_checkpoint(self.best_model_path)
            
        # 在测试集上评估
        test_metrics = self.test()
        
        print("Test Metrics:")
        for name, value in test_metrics.items():
            print(f"{name}: {value:.4f}")
            
        return test_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Emotion Recognition Model')
    parser.add_argument('--train_id', type=str, help='Training ID (e.g., 01, 02)')
    args = parser.parse_args()
    
    # 设置配置
    config = Config()
    
    # 创建训练器
    trainer = Trainer(config, train_id=args.train_id)
    
    # 检查是否有最佳模型检查点
    best_model_path = trainer.checkpoint_dir / "best_model.pt"
    
    if best_model_path.exists():
        print(f"Found best model checkpoint: {best_model_path}")
        resume = input("Resume training? (y/n): ").lower() == 'y'
        
        if resume:
            trainer.train(resume_from=best_model_path)
        else:
            trainer.train()
    else:
        trainer.train() 