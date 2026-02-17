from email import iterators
from Transformer.models.embed import DataEmbedding
from torch import nn
import torch
import itertools

from Transformer.models.attn import AttentionLayer, FullAttention
from Transformer.models.layer import MultiHeadAttention

import pdb


class policy_transformer_stock_atten2(nn.Module): # attention(long, short), attention(hybrid, relational) 
    def __init__(self, d_model=128, n_heads=4, dropout=0.1, lr=0.0001, output_attention=False, device=None, additional_dim=20):
        super().__init__()
        # 第一级级联：用于关系特征与上下文特征的交叉融合
        self.attention = AttentionLayer(FullAttention(False, attention_dropout=dropout,
                                      output_attention=output_attention), d_model, n_heads)
        # 第二级级联：用于融合后的全局精炼
        self.attention2 = AttentionLayer(FullAttention(False, attention_dropout=dropout,
                                      output_attention=output_attention), d_model, n_heads)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # 1. Feature Adapters (渐进式融合适配器)
        # MAE 关系特征适配
        self.relational_adapter = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU()
        )
        # 环境上下文特征 (Tech + Date) 适配
        self.context_adapter = nn.Sequential(
            nn.Linear(additional_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU()
        )

        # 2. Output Projection
        self.out_dim = d_model 
        self.projection = nn.Sequential(
            nn.Linear(d_model, self.out_dim),
            nn.GELU(),
            nn.LayerNorm(self.out_dim)
        )

        # 检测GPU可用性并决定使用GPU还是CPU
        if device is None:
            if torch.cuda.is_available():
                device = 'cuda:0'
            else:
                device = 'cpu'

        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=1e-4)
        self.device = device
        
    def forward(self, relational_feature, temporal_feature_short, temporal_feature_long, additional_feature, mask=None):
        # relational_feature: [B, N, 128] (From MAE)
        # additional_feature: [B, N, additional_dim] (Tech + Date)
        
        # 1. 个性化特征适配
        rel_feat = self.relational_adapter(relational_feature) # [B, N, 128]
        ctx_feat = self.context_adapter(additional_feature)   # [B, N, 128]

        # 2. 第一级级联：Cross-Attention (用关系特征去索引上下文信息)
        # Query: Relational, Key/Value: Context
        fused_1, _ = self.attention(
            rel_feat, ctx_feat, ctx_feat,
            attn_mask=mask
        )
        # 残差连接与归一化
        fused_1 = self.norm(rel_feat + self.dropout(fused_1))
        
        # 3. 第二级级联：Self-Attention (全局特征精炼)
        # 在融合了上下文后，进行全市场的股票间信息交互
        fused_2, _ = self.attention2(
            fused_1, fused_1, fused_1,
            attn_mask=mask
        )
        fused_2 = self.norm2(fused_1 + self.dropout(fused_2))

        # 4. 输出投影
        output_hybrid = self.projection(fused_2) # [B, N, 128]
        
        # 保持与原有系统兼容的 Late Fusion (拼接原始上下文)
        combined_feature = torch.cat((output_hybrid, additional_feature), dim=-1) 
        return combined_feature


