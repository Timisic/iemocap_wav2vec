import os
from pathlib import Path


class Config:
    PROJECT_ROOT = Path(os.path.dirname(os.path.abspath(__file__)))

    # 数据路径
    DATA_ROOT = PROJECT_ROOT / "data"
    IEMOCAP_PATH = DATA_ROOT / "IEMOCAP_Audio"

    # 模型路径
    MODEL_PATH = PROJECT_ROOT / "models" / "wav2vec2-base-960h"

    # 输出路径
    OUTPUT_DIR = PROJECT_ROOT / "outputs"
    LOG_DIR = OUTPUT_DIR / "logs"
    CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
    FIGURE_DIR = OUTPUT_DIR / "figures"

    # 创建必要的目录
    for dir_path in [DATA_ROOT, OUTPUT_DIR, LOG_DIR, CHECKPOINT_DIR, FIGURE_DIR]:
        dir_path.mkdir(parents=True, exist_ok=True)

    # 训练参数
    BATCH_SIZE = 32
    GRADIENT_ACCUMULATION_STEPS = 2
    LEARNING_RATE = 5e-6
    WEIGHT_DECAY = 0.01
    NUM_EPOCHS = 13
    WARMUP_RATIO = 0.1
    MAX_GRAD_NORM = 1.0

    # 音频处理参数
    MAX_DURATION_IN_SECONDS = 10
    SAMPLING_RATE = 16000

    # 早停参数
    EARLY_STOPPING = True
    PATIENCE = 5
    MIN_DELTA = 0.001

    # 多GPU训练
    USE_MULTI_GPU = True

    # 情绪标签
    EMOTION_LABELS = ["ang", "hap", "sad", "neu"]
    NUM_LABELS = len(EMOTION_LABELS)

    # 随机种子
    SEED = 42


if __name__ == "__main__":
    config = Config()
    print(f"项目根目录: {config.PROJECT_ROOT}")
    print(f"数据目录: {config.DATA_ROOT}")
    print(f"模型目录: {config.MODEL_PATH}")
    print(f"输出目录: {config.OUTPUT_DIR}")
    print(f"情绪标签: {config.EMOTION_LABELS}")
    print(f"批次大小: {config.BATCH_SIZE}")
    print(f"学习率: {config.LEARNING_RATE}")
