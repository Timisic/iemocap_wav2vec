import os
import torch
import torchaudio
import numpy as np
from pathlib import Path
from tqdm import tqdm
from transformers import Wav2Vec2Config, Wav2Vec2ForPreTraining
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
import argparse
import logging
from datetime import datetime
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AudioDataset(Dataset):
    def __init__(self, audio_paths, max_duration=10, sample_rate=16000):
        """
        音频数据集
        
        Args:
            audio_paths: 音频文件路径列表
            max_duration: 最大音频长度（秒）
            sample_rate: 采样率
        """
        self.audio_paths = audio_paths
        self.max_duration = max_duration
        self.sample_rate = sample_rate
        self.max_length = max_duration * sample_rate
        
    def __len__(self):
        return len(self.audio_paths)
    
    def __getitem__(self, idx):
        # 加载音频
        waveform, sr = torchaudio.load(self.audio_paths[idx])
        
        # 重采样
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)
        
        # 转换为单声道
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # 裁剪或填充
        if waveform.shape[1] > self.max_length:
            waveform = waveform[:, :self.max_length]
        else:
            padding = torch.zeros(1, self.max_length - waveform.shape[1])
            waveform = torch.cat([waveform, padding], dim=1)
        
        return {
            "input_values": waveform.squeeze(),
            "attention_mask": torch.ones(self.max_length)
        }

class Wav2Vec2Pretrainer:
    def __init__(self, output_dir, config=None, local_rank=-1):
        """
        初始化预训练器
        
        Args:
            output_dir: 输出目录
            config: 模型配置
            local_rank: 当前GPU编号
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.local_rank = local_rank
        
        # 设置设备
        if torch.cuda.is_available():
            if local_rank != -1:
                self.device = torch.device(f'cuda:{local_rank}')
                torch.cuda.set_device(local_rank)
            else:  # 单GPU或DataParallel
                self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
            
        logger.info(f"使用设备: {self.device}")
        
        # 初始化模型
        if config is None:
            self.model = Wav2Vec2ForPreTraining.from_pretrained("wav2vec2-base-960h")
        else:
            self.model = Wav2Vec2ForPreTraining(config)
        
        # 设置分布式训练
        if local_rank != -1:
            self.model = DDP(self.model, device_ids=[local_rank])
        elif torch.cuda.device_count() > 1:
            logger.info(f"使用 {torch.cuda.device_count()} 个GPU进行训练")
            self.model = torch.nn.DataParallel(self.model)
            
        self.model.to(self.device)
        
        # 使用混合精度训练
        self.scaler = torch.cuda.amp.GradScaler()
        
        # 初始化优化器
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=1e-4,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=0.01
        )

    def train(self, dataloader, num_epochs, save_steps=100):
        """训练方法保持不变，但添加混合精度训练"""
        self.model.train()
        
        scheduler = OneCycleLR(
            self.optimizer,
            max_lr=1e-4,
            epochs=num_epochs,
            steps_per_epoch=len(dataloader),
            pct_start=0.1
        )
        
        for epoch in range(num_epochs):
            if isinstance(dataloader.sampler, DistributedSampler):
                dataloader.sampler.set_epoch(epoch)
                
            epoch_loss = 0
            progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
            
            for step, batch in enumerate(progress_bar):
                input_values = batch["input_values"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                
                # 使用混合精度训练
                with torch.cuda.amp.autocast():
                    outputs = self.model(
                        input_values=input_values,
                        attention_mask=attention_mask
                    )
                    loss = outputs.loss
                
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
                scheduler.step()
                self.optimizer.zero_grad()
                
                epoch_loss += loss.item()
                progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
                
                if (step + 1) % save_steps == 0 and self.local_rank in [-1, 0]:
                    self.save_checkpoint(epoch, step)
            
            avg_loss = epoch_loss / len(dataloader)
            if self.local_rank in [-1, 0]:
                logger.info(f"Epoch {epoch+1}/{num_epochs} - Average Loss: {avg_loss:.4f}")
                self.save_checkpoint(epoch, len(dataloader))

    def save_checkpoint(self, epoch, step):
        """保存检查点"""
        checkpoint_path = self.output_dir / f"checkpoint_epoch{epoch+1}_step{step+1}.pt"
        torch.save({
            'epoch': epoch,
            'step': step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, checkpoint_path)
        logger.info(f"保存检查点到: {checkpoint_path}")

def get_audio_files(audio_dir):
    """获取目录下所有音频文件"""
    audio_files = []
    for ext in ['.wav', '.mp3', '.flac']:
        audio_files.extend(list(Path(audio_dir).glob(f"**/*{ext}")))
    if not audio_files:
        raise ValueError(f"在 {audio_dir} 中没有找到音频文件")
    return audio_files

def main():
    parser = argparse.ArgumentParser(description='Wav2Vec2自监督预训练')
    parser.add_argument('--audio_dir', type=str, default='./data/audio', help='音频文件目录')
    parser.add_argument('--output_dir', type=str, default='./outputs', help='输出目录')
    parser.add_argument('--num_epochs', type=int, default=50, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=4, help='每个GPU的批次大小')
    parser.add_argument('--max_duration', type=int, default=30, help='最大音频长度（秒）')
    parser.add_argument('--local_rank', type=int, default=-1, help='分布式训练的GPU编号')
    args = parser.parse_args()
    
    # 初始化分布式训练
    if args.local_rank != -1:
        torch.cuda.set_device(args.local_rank)
        dist.init_process_group(backend='nccl')
        
    # 创建数据加载器
    dataset = AudioDataset(
        get_audio_files(args.audio_dir),
        max_duration=args.max_duration
    )
    
    if args.local_rank != -1:
        sampler = DistributedSampler(dataset)
        batch_size = args.batch_size
    else:
        sampler = None
        batch_size = args.batch_size * max(1, torch.cuda.device_count())
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=4,
        pin_memory=True
    )
    
    # 初始化训练器
    trainer = Wav2Vec2Pretrainer(args.output_dir, local_rank=args.local_rank)
    
    # 开始训练
    if args.local_rank in [-1, 0]:
        logger.info("开始训练...")
    trainer.train(dataloader, args.num_epochs)
    if args.local_rank in [-1, 0]:
        logger.info("训练完成！")

if __name__ == "__main__":
    main() 