Title: Competency Prediction Model for Job Applicants Based on Audio Data

大的方法：迁移学习、多任务联合学习、wav2vec2.0模型微调

原始数据：IEMOCAP_Audio数据集，四个情绪分类
目的：输入语音，输出情绪分类，分类任务
方法：使用wav2vec2.0（facebook/wav2vec2-base-960h）模型进行微调，进行多任务学习

请帮我写微调wav2vec2.0的代码，要求：

1、模型结构与wav2vec2.0相同，复制其网络中的参数，放在model.py
2、读取数据集，并且转为wav2vec2.0的格式，放在data.py
3、使用wav2vec2.0自带的preprocessor_config.json对音频进行处理，放在preprocess.py
4、写一个配置文件config.py，包含训练中的各种参数，需要的各种路径（我需要跨系统，读取文件路径，使用相对路径）
5、写train.py，内含各种验证的指标，配置早停策略（我可以选择用或不用）
6、使用matplotlib绘制过程中的loss，评估指标等，保存图片，确保命名规范。放在utils.py，让train.py调用
7、核心的py文件都需要if \_name\_ == main的测试，确保该程序可以使用
8、生成需要的requirements.txt，我有双卡可以进行并行训练，总显存在48G

我的wav2vec2.0模型文件会放在models文件夹中