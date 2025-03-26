import os
import torch
import torchaudio
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse
import logging

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AudioSplitter:
    def __init__(self, segment_duration=30, overlap_duration=1, sample_rate=16000):
        """
        音频切分器
        
        Args:
            segment_duration: 切分片段长度（秒）
            overlap_duration: 重叠长度（秒）
            sample_rate: 采样率
        """
        self.segment_duration = segment_duration
        self.overlap_duration = overlap_duration
        self.sample_rate = sample_rate
        self.segment_length = segment_duration * sample_rate
        self.overlap_length = overlap_duration * sample_rate
        
    def split_audio(self, input_path, output_dir):
        """
        切分单个音频文件
        
        Args:
            input_path: 输入音频文件路径
            output_dir: 输出目录
        """
        # 加载音频
        waveform, sr = torchaudio.load(input_path)
        
        # 重采样
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)
        
        # 转换为单声道
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # 计算片段数
        total_length = waveform.shape[1]
        step_size = self.segment_length - self.overlap_length
        num_segments = max(1, int(np.ceil((total_length - self.overlap_length) / step_size)))
        
        # 创建输出目录
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 切分并保存
        segments_saved = 0
        for i in range(num_segments):
            start = i * step_size
            end = min(start + self.segment_length, total_length)
            
            # 如果最后一段太短，就不保存
            if end - start < self.segment_length * 0.5:
                break
                
            segment = waveform[:, start:end]
            
            # 如果需要，填充最后一段
            if segment.shape[1] < self.segment_length:
                padding = torch.zeros(1, self.segment_length - segment.shape[1])
                segment = torch.cat([segment, padding], dim=1)
            
            # 保存片段
            output_path = output_dir / f"{Path(input_path).stem}_segment_{i+1:03d}.wav"
            torchaudio.save(output_path, segment, self.sample_rate)
            segments_saved += 1
            
        return segments_saved

    def process_directory(self, input_dir, output_dir):
        """
        处理整个目录的音频文件
        
        Args:
            input_dir: 输入目录
            output_dir: 输出目录
        """
        input_dir = Path(input_dir)
        total_segments = 0
        
        # 获取所有音频文件
        audio_files = []
        for ext in ['.wav', '.mp3', '.flac']:
            audio_files.extend(list(input_dir.glob(f"**/*{ext}")))
        
        if not audio_files:
            raise ValueError(f"在 {input_dir} 中没有找到音频文件")
        
        logger.info(f"找到 {len(audio_files)} 个音频文件")
        
        # 处理每个文件
        for audio_file in tqdm(audio_files, desc="处理音频文件"):
            try:
                segments = self.split_audio(
                    audio_file,
                    Path(output_dir) / audio_file.stem
                )
                total_segments += segments
            except Exception as e:
                logger.error(f"处理文件 {audio_file} 时出错: {str(e)}")
                continue
        
        logger.info(f"处理完成！共生成 {total_segments} 个音频片段")
        return total_segments

def main():
    # 记得修改！！ ================================
    input_dir = "/Users/zhaoyu/Desktop/audio_data/audio_data_1"
    output_dir = "/Users/zhaoyu/Desktop/audio_data/audio_data_1_split"

    parser = argparse.ArgumentParser(description='音频文件切分工具')
    parser.add_argument('--input_dir', type=str, default=input_dir, help='输入音频目录')
    parser.add_argument('--output_dir', type=str, default=output_dir, help='输出目录')
    parser.add_argument('--segment_duration', type=int, default=30, help='切分长度（秒）')
    parser.add_argument('--overlap_duration', type=float, default=1.0, help='重叠长度（秒）')
    parser.add_argument('--sample_rate', type=int, default=16000, help='采样率')
    args = parser.parse_args()
    
    splitter = AudioSplitter(
        segment_duration=args.segment_duration,
        overlap_duration=args.overlap_duration,
        sample_rate=args.sample_rate
    )
    
    splitter.process_directory(args.input_dir, args.output_dir)

if __name__ == "__main__":
    main()