import os
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from config import Config
from preprocess import AudioPreprocessor


class IEMOCAPDataset(Dataset):
    def __init__(self, data_root, file_list, labels, preprocessor, audio_bytes):
        """
        IEMOCAP数据集
        
        Args:
            data_root: 数据根目录
            file_list: 文件路径列表
            labels: 标签列表
            preprocessor: 音频预处理器
            audio_bytes: 音频二进制数据列表
        """
        self.data_root = data_root
        self.file_list = file_list
        self.labels = labels
        self.preprocessor = preprocessor
        self.audio_bytes = audio_bytes

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        audio_data = self.audio_bytes[idx]
        label = self.labels[idx]

        # 预处理音频
        inputs = self.preprocessor.preprocess_bytes(audio_data)
        
        # 获取input_values并确保它是一维的
        input_values = inputs.input_values.squeeze()
        
        # 创建相应的attention_mask
        attention_mask = torch.ones_like(input_values, dtype=torch.long)

        return {
            "input_values": input_values,
            "attention_mask": attention_mask,
            "labels": torch.tensor(label, dtype=torch.long)
        }


class IEMOCAPDataModule:
    def __init__(self, config=None):
        """
        IEMOCAP数据模块
        
        Args:
            config: 配置对象
        """
        if config is None:
            self.config = Config()
        else:
            self.config = config

        self.data_root = self.config.IEMOCAP_PATH
        self.preprocessor = AudioPreprocessor(self.config.MODEL_PATH)
        
        # 修改情绪标签映射，使用数字标签
        self.label_map = {i: label for i, label in enumerate(self.config.EMOTION_LABELS)}
        self.emotion_map = {label: i for i, label in enumerate(self.config.EMOTION_LABELS)}

        # 数据集划分
        self.train_files = []
        self.train_labels = []
        self.val_files = []
        self.val_labels = []
        self.test_files = []
        self.test_labels = []

    def prepare_data(self):
        """准备数据，读取parquet文件并创建数据集划分"""
        all_files = []
        all_labels = []
        all_audio_bytes = []  # 新增：存储音频二进制数据

        # 读取parquet文件
        parquet_files = list(Path(self.data_root).glob("session*-*.parquet"))
        print(f"找到的parquet文件数量: {len(parquet_files)}")
        
        for session_file in parquet_files:
            print(f"正在处理文件: {session_file}")
            df = pd.read_parquet(session_file)
            print(f"原始数据行数: {len(df)}")
            print(f"数据框列名: {df.columns.tolist()}")
            
            print(f"可用的情绪标签: {df['label'].unique()}")
            
            all_files.extend(df["audio.path"].tolist())
            all_labels.extend(df["label"].tolist())
            all_audio_bytes.extend(df["audio.bytes"].tolist())

        print(f"总样本数: {len(all_files)}")
        if len(all_files) == 0:
            raise ValueError(
                f"没有找到任何符合条件的数据。\n"
                f"数据目录: {self.data_root}\n"
                f"目标情绪标签: {self.config.EMOTION_LABELS}\n"
                f"标签映射: {self.label_map}"
            )

        # 划分数据集
        train_indices, test_indices = train_test_split(
            range(len(all_files)), test_size=0.2, random_state=self.config.SEED,
            stratify=all_labels
        )

        train_indices, val_indices = train_test_split(
            train_indices, test_size=0.2, random_state=self.config.SEED,
            stratify=[all_labels[i] for i in train_indices]
        )

        # 使用索引来分割所有数据
        self.train_files = [all_files[i] for i in train_indices]
        self.train_labels = [all_labels[i] for i in train_indices]
        self.train_audio_bytes = [all_audio_bytes[i] for i in train_indices]

        self.val_files = [all_files[i] for i in val_indices]
        self.val_labels = [all_labels[i] for i in val_indices]
        self.val_audio_bytes = [all_audio_bytes[i] for i in val_indices]

        self.test_files = [all_files[i] for i in test_indices]
        self.test_labels = [all_labels[i] for i in test_indices]
        self.test_audio_bytes = [all_audio_bytes[i] for i in test_indices]

        print(f"训练集: {len(self.train_files)}个样本")
        print(f"验证集: {len(self.val_files)}个样本")
        print(f"测试集: {len(self.test_files)}个样本")

        # 检查类别分布
        train_dist = np.bincount(self.train_labels, minlength=len(self.config.EMOTION_LABELS))
        val_dist = np.bincount(self.val_labels, minlength=len(self.config.EMOTION_LABELS))
        test_dist = np.bincount(self.test_labels, minlength=len(self.config.EMOTION_LABELS))

        print("类别分布:")
        for i, label in enumerate(self.config.EMOTION_LABELS):
            print(f"{label}: 训练集 {train_dist[i]}, 验证集 {val_dist[i]}, 测试集 {test_dist[i]}")

    def get_dataloader(self, split="train", batch_size=None, shuffle=None):
        """
        获取数据加载器
        
        Args:
            split: 数据集划分，可选值为"train", "val", "test"
            batch_size: 批次大小，如果为None则使用配置中的值
            shuffle: 是否打乱数据，如果为None则根据split决定
            
        Returns:
            数据加载器
        """
        if batch_size is None:
            batch_size = self.config.BATCH_SIZE

        if shuffle is None:
            shuffle = (split == "train")

        if split == "train":
            dataset = IEMOCAPDataset(
                self.data_root, self.train_files, self.train_labels,
                self.preprocessor, self.train_audio_bytes
            )
        elif split == "val":
            dataset = IEMOCAPDataset(
                self.data_root, self.val_files, self.val_labels,
                self.preprocessor, self.val_audio_bytes
            )
        elif split == "test":
            dataset = IEMOCAPDataset(
                self.data_root, self.test_files, self.test_labels,
                self.preprocessor, self.test_audio_bytes
            )
        else:
            raise ValueError(f"未知的数据集划分: {split}")

        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=4,
            pin_memory=True
        )


if __name__ == "__main__":
    # 测试数据模块
    config = Config()
    data_module = IEMOCAPDataModule(config)

    # 检查数据目录是否存在
    if Path(config.IEMOCAP_PATH).exists():
        # 准备数据
        data_module.prepare_data()

        # 获取数据加载器
        train_loader = data_module.get_dataloader("train", batch_size=2)

        # 测试数据加载
        for batch in train_loader:
            print(f"输入形状: {batch['input_values'].shape}")
            print(f"注意力掩码形状: {batch['attention_mask'].shape}")
            print(f"标签: {batch['labels']}")
            break
    else:
        print(f"数据目录不存在: {config.IEMOCAP_PATH}")
        print("数据模块初始化成功，但无法测试具体功能")
