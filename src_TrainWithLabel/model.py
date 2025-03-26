import torch
import torch.nn as nn
from transformers import Wav2Vec2Model, Wav2Vec2Config
from config import Config

class EmotionClassifier(nn.Module):
    def __init__(self, config=None, num_labels=None, model_path=None):
        """
        基于Wav2Vec2的情感分类模型
        
        Args:
            config: 配置对象
            num_labels: 标签数量，如果为None则使用配置中的值
            model_path: 预训练模型路径，如果为None则使用配置中的值
        """
        super().__init__()
        
        if config is None:
            config = Config()
            
        if num_labels is None:
            num_labels = config.NUM_LABELS
            
        if model_path is None:
            model_path = config.MODEL_PATH
        
        # 加载预训练的Wav2Vec2模型
        self.wav2vec2 = Wav2Vec2Model.from_pretrained(model_path)
        
        # 获取隐藏层大小
        hidden_size = self.wav2vec2.config.hidden_size
        
        # 增强的分类头，添加多层正则化
        self.classifier = nn.Sequential(
            # 第一层
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(0.3),
            
            # 添加第二层，增加网络深度和正则化
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.GELU(),
            nn.Dropout(0.2),
            
            # 输出层
            nn.Linear(hidden_size // 2, num_labels)
        )
        
        # 添加Label Smoothing的损失函数
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        
        # 添加权重正则化
        self.l2_lambda = 0.01  # L2正则化系数
        
    def forward(self, input_values, attention_mask=None, labels=None):
        """
        前向传播
        
        Args:
            input_values: 输入特征
            attention_mask: 注意力掩码
            labels: 标签（用于计算损失）
            
        Returns:
            输出字典，包含logits和损失（如果提供了标签）
        """
        # 获取Wav2Vec2的输出
        outputs = self.wav2vec2(
            input_values,
            attention_mask=attention_mask,
            output_attentions=False,
            output_hidden_states=True,  # 获取所有隐藏层状态用于特征融合
            return_dict=True
        )
        
        # 获取最后一层的隐藏状态
        hidden_states = outputs.last_hidden_state
        
        # 特征融合：结合最后几层的隐藏状态
        last_hidden_states = outputs.hidden_states[-4:]  # 获取最后4层
        hidden_states = torch.stack(last_hidden_states, dim=1).mean(dim=1)
        
        # 对隐藏状态进行平均池化
        if attention_mask is not None:
            attention_mask = self._get_feature_vector_attention_mask(
                hidden_states.shape[1],
                attention_mask
            )
            mask = attention_mask.unsqueeze(-1).expand(hidden_states.size())
            hidden_states = torch.sum(hidden_states * mask, dim=1) / torch.sum(mask, dim=1)
        else:
            hidden_states = torch.mean(hidden_states, dim=1)
        
        # 分类
        logits = self.classifier(hidden_states)
        
        # 计算损失
        loss = None
        if labels is not None:
            # 交叉熵损失
            ce_loss = self.criterion(logits, labels)
            
            # L2正则化损失
            l2_loss = 0
            for name, param in self.named_parameters():
                if 'weight' in name:  # 只对权重应用L2正则化
                    l2_loss += torch.norm(param, p=2)
            
            # 总损失
            loss = ce_loss + self.l2_lambda * l2_loss
        
        return {
            'loss': loss,
            'logits': logits
        } if loss is not None else logits
    
    def _get_feature_vector_attention_mask(self, feature_vector_length, attention_mask):
        """
        将输入的注意力掩码调整为特征向量的长度
        
        Args:
            feature_vector_length: 特征向量长度
            attention_mask: 原始注意力掩码
            
        Returns:
            调整后的注意力掩码
        """
        # Wav2Vec2 默认下采样率为320
        output_lengths = self._get_feat_extract_output_lengths(attention_mask.sum(-1))
        batch_size = attention_mask.shape[0]
        
        attention_mask = torch.zeros(
            (batch_size, feature_vector_length),
            dtype=attention_mask.dtype,
            device=attention_mask.device,
        )
        
        # 设置有效位置为1
        for i in range(batch_size):
            # 将张量转换为整数
            length = int(output_lengths[i].item())
            attention_mask[i, :length] = 1
        
        return attention_mask

    def _get_feat_extract_output_lengths(self, input_lengths):
        """
        计算特征提取后的输出长度
        """
        def _conv_out_length(input_length, kernel_size, stride):
            return torch.div(input_length - kernel_size, stride, rounding_mode='floor') + 1

        for kernel_size, stride in zip(
            self.wav2vec2.config.conv_kernel, self.wav2vec2.config.conv_stride
        ):
            input_lengths = _conv_out_length(input_lengths, kernel_size, stride)

        return input_lengths
    
    def freeze_feature_extractor(self):
        """冻结特征提取器参数"""
        for param in self.wav2vec2.feature_extractor.parameters():
            param.requires_grad = False
    
    def unfreeze_feature_extractor(self):
        """解冻特征提取器参数"""
        for param in self.wav2vec2.feature_extractor.parameters():
            param.requires_grad = True


if __name__ == "__main__":
    # 测试模型
    config = Config()
    
    # 创建单任务模型
    model = EmotionClassifier(config)
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters())}")
    
    # 测试前向传播
    batch_size = 2
    seq_length = 16000 * 5  # 5秒音频
    input_values = torch.randn(batch_size, seq_length)
    attention_mask = torch.ones(batch_size, seq_length)
    
    # 前向传播
    logits = model(input_values, attention_mask)
    print(f"输出logits形状: {logits.shape}") 