import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# 1. 位置编码 (Positional Encoding)
class PositionalEncoding(nn.Module):
    """标准 Transformer 位置编码"""
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        if d_model % 2 != 0:
            raise ValueError("d_model 必须是偶数")
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """输入 x: (batch, seq_len, d_model), 输出增加 PE 后的 x"""
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

# 2. Transformer 特征提取器 (共享)
class TransformerFeatureExtractor(nn.Module):
    """使用 Transformer Encoder 提取序列特征"""
    def __init__(self, input_dim: int, model_dim: int, num_heads: int, num_layers: int, dropout: float = 0.1, max_len: int = 500):
        super().__init__()
        self.model_dim = model_dim
        self.embedding = nn.Linear(input_dim, model_dim) # 输入映射
        self.pos_encoder = PositionalEncoding(model_dim, dropout, max_len) # 位置编码
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim, nhead=num_heads, dropout=dropout,
            dim_feedforward=model_dim * 4, activation='relu', batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers) # Encoder 核心

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """输入 x: (batch, seq_len, input_dim) -> 输出 features: (batch, seq_len, model_dim)"""
        x = self.embedding(x) * math.sqrt(self.model_dim) # 映射 & 缩放
        x = self.pos_encoder(x) # 添加 PE
        encoded_features = self.transformer_encoder(x) # 通过 Encoder
        return encoded_features


# 3. Transformer 相似度模型 - Added LayerNorm before MLP
class TransformerSimilarityModel(nn.Module):
    """基于共享 Transformer 和特征比较的相似度模型"""
    def __init__(self, input_dim: int, model_dim: int, num_heads: int, num_layers: int, dropout: float = 0.1, pooling_method: str = 'mean', max_len: int = 500):
        super().__init__()
        if pooling_method not in ['mean', 'max']: # 添加 max 池化支持
             raise ValueError("pooling_method 支持 'mean' 或 'max'")
        self.pooling_method = pooling_method
        # 共享特征提取器 (Siamese)
        self.feature_extractor = TransformerFeatureExtractor(
            input_dim, model_dim, num_heads, num_layers, dropout, max_len
        )
        # ****** 新增: 在 MLP 输入前添加 LayerNorm ******
        self.mlp_input_norm = nn.LayerNorm(model_dim * 4)
        # **************************************************

        # MLP 分类器，输入为 4*model_dim
        self.mlp_classifier = nn.Sequential(
            # 注意: 第一个 Linear 之前已经过 Norm 处理
            nn.Linear(model_dim * 4, model_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim, 1),# 输出 logits
        )

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """输入 x1, x2: (batch, seq_len, input_dim) -> 输出 logits: (batch, 1)"""
        # 1. 提取特征 (共享编码器) -> (batch, seq_len, model_dim)
        enc1 = self.feature_extractor(x1)
        enc2 = self.feature_extractor(x2)

        # 2. 时间池化 -> (batch, model_dim)
        if self.pooling_method == 'mean':
            feat1 = torch.mean(enc1, dim=1)
            feat2 = torch.mean(enc2, dim=1)
        elif self.pooling_method == 'max': # 添加 max 池化实现
            feat1 = torch.max(enc1, dim=1)[0]
            feat2 = torch.max(enc2, dim=1)[0]


        # 3. 特征比较
        diff = torch.abs(feat1 - feat2)
        prod = feat1 * feat2

        # 4. 拼接特征 -> (batch, model_dim * 4)
        combined_features = torch.cat([feat1, feat2, diff, prod], dim=1)

        # ****** 应用新增的 LayerNorm ******
        combined_features = self.mlp_input_norm(combined_features)
        # ***********************************

        # 5. MLP 分类
        output = self.mlp_classifier(combined_features) # (batch, 1)
        return output