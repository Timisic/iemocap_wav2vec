import json
import torch
import librosa
import numpy as np
from pathlib import Path
from transformers import Wav2Vec2Processor
from config import Config
import tempfile


class AudioPreprocessor:
    def __init__(self, processor_path=None):
        """
        初始化音频预处理器
        
        Args:
            processor_path: Wav2Vec2Processor的路径，如果为None则使用配置中的默认路径
        """
        if processor_path is None:
            processor_path = Config.MODEL_PATH

        self.processor = Wav2Vec2Processor.from_pretrained(processor_path)
        self.sampling_rate = Config.SAMPLING_RATE
        self.max_duration = Config.MAX_DURATION_IN_SECONDS

        # 加载处理器配置
        processor_config_path = Path(processor_path) / "preprocessor_config.json"
        if processor_config_path.exists():
            with open(processor_config_path, "r") as f:
                self.processor_config = json.load(f)
            print(f"已加载预处理器配置: {processor_config_path}")
        else:
            self.processor_config = None
            print(f"未找到预处理器配置文件: {processor_config_path}")

    def load_audio(self, file_path):
        """
        加载音频文件并重采样
        
        Args:
            file_path: 音频文件路径
            
        Returns:
            重采样后的音频数据
        """
        audio, sr = librosa.load(file_path, sr=self.sampling_rate)
        return audio

    def preprocess_audio(self, audio):
        """
        预处理音频数据
        
        Args:
            audio: 音频数据
            
        Returns:
            预处理后的音频特征
        """
        # 裁剪或填充音频到最大长度
        max_length = self.max_duration * self.sampling_rate
        if len(audio) > max_length:
            audio = audio[:max_length]
        else:
            # 填充短音频
            padding = np.zeros(max_length - len(audio))
            audio = np.concatenate([audio, padding])

        # 使用Wav2Vec2Processor处理音频
        inputs = self.processor(
            audio,
            sampling_rate=self.sampling_rate,
            return_tensors="pt",
            padding="max_length",
            max_length=max_length
        )

        return inputs

    def preprocess_file(self, file_path):
        """
        预处理单个音频文件
        
        Args:
            file_path: 音频文件路径
            
        Returns:
            预处理后的音频特征
        """
        audio = self.load_audio(file_path)
        return self.preprocess_audio(audio)

    def preprocess_batch(self, file_paths):
        """
        批量预处理音频文件
        
        Args:
            file_paths: 音频文件路径列表
            
        Returns:
            批量预处理后的音频特征
        """
        audios = [self.load_audio(path) for path in file_paths]

        # 获取最大长度
        max_length = min(
            max(len(audio) for audio in audios),
            self.max_duration * self.sampling_rate
        )

        # 裁剪或填充音频
        processed_audios = []
        for audio in audios:
            if len(audio) > max_length:
                processed_audios.append(audio[:max_length])
            else:
                padding = np.zeros(max_length - len(audio))
                processed_audios.append(np.concatenate([audio, padding]))

        # 使用Wav2Vec2Processor处理音频
        inputs = self.processor(
            processed_audios,
            sampling_rate=self.sampling_rate,
            return_tensors="pt",
            padding="max_length",
            max_length=max_length
        )

        return inputs

    def preprocess_bytes(self, audio_bytes):
        """
        从二进制数据预处理音频
        
        Args:
            audio_bytes: 音频的二进制数据
        Returns:
            包含input_values的字典
        """
        # 将二进制数据写入临时文件
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=True) as temp_file:
            temp_file.write(audio_bytes)
            temp_file.flush()
            # 加载音频
            audio = self.load_audio(temp_file.name)
            
            # 裁剪或填充音频到最大长度
            max_length = self.max_duration * self.sampling_rate
            if len(audio) > max_length:
                audio = audio[:max_length]
            else:
                # 填充短音频
                padding = np.zeros(max_length - len(audio))
                audio = np.concatenate([audio, padding])
            
            # 预处理
            return self.processor(
                audio,
                sampling_rate=self.sampling_rate,
                return_tensors="pt",
                padding=True
            )


if __name__ == "__main__":
    # 测试预处理器
    config = Config()
    preprocessor = AudioPreprocessor(config.MODEL_PATH)

    test_file = Path("/Users/hong/Downloads/3.wav")
    if test_file.exists():
        print(f"测试预处理单个文件: {test_file}")
        inputs = preprocessor.preprocess_file(test_file)
        print(f"预处理后的输入形状: {inputs.input_values.shape}")
    else:
        print(f"测试文件不存在: {test_file}")
        print("预处理器初始化成功，但无法测试具体功能")
