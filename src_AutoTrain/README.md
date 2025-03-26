# Wav2Vec2 自监督预训练

这个脚本用于对 Wav2Vec2 模型进行自监督预训练，不需要标注数据。

## 环境要求

-   Python 3.7+
-   PyTorch
-   torchaudio
-   transformers
-   tqdm

## 使用方法

1. 准备音频数据：

    - 将所有音频文件（支持.wav、.mp3、.flac 格式）放在一个目录中
    - 音频文件会被自动重采样到 16kHz
    - 默认最大音频长度为 10 秒

2. 运行训练脚本：

```bash
python auto_train.py \
    --audio_dir /path/to/your/audio/files \
    --output_dir /path/to/save/model \
    --num_epochs 50 \
    --batch_size 4 \
    --max_duration 10
```

参数说明：

-   `--audio_dir`: 音频文件目录（必需）
-   `--output_dir`: 模型保存目录（必需）
-   `--num_epochs`: 训练轮数（默认：50）
-   `--batch_size`: 批次大小（默认：4）
-   `--max_duration`: 最大音频长度，单位秒（默认：10）

## 注意事项

1. 数据量建议：

    - 建议至少几百小时的音频数据
    - 数据质量比数量更重要

2. 硬件要求：

    - 建议使用 GPU 进行训练
    - 内存建议至少 16GB
    - 显存建议至少 8GB

3. 训练过程：

    - 会自动保存检查点
    - 每个 epoch 结束后保存一次
    - 每 100 步保存一次中间检查点

4. 输出文件：
    - 检查点文件格式：`checkpoint_epoch{N}_step{M}.pt`
    - 包含模型状态和优化器状态

## 使用预训练模型

训练完成后，您可以使用保存的模型进行特征提取或下游任务：

```python
from transformers import Wav2Vec2Model

# 加载预训练模型
model = Wav2Vec2Model.from_pretrained("/path/to/save/model")
```

## 常见问题

1. 内存不足：

    - 减小 batch_size
    - 减小 max_duration
    - 使用梯度累积

2. 训练不稳定：

    - 调整学习率
    - 检查数据质量
    - 增加 warmup 步数

3. 效果不理想：
    - 增加训练数据量
    - 调整模型配置
    - 增加训练轮数
