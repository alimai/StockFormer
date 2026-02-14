from email import iterators
from Transformer.models.embed import DataEmbedding
from torch import nn
import torch
import itertools

from Transformer.models.attn import AttentionLayer, FullAttention
from Transformer.models.layer import MultiHeadAttention

import pdb


class policy_transformer_stock_atten2(nn.Module): # attention(long, short), attention(hybrid, relational) 
    def __init__(self, d_model=128, n_heads=4, dropout=0.1, lr=0.0001, output_attention=False, device=None):
        super().__init__()
        self.attention = AttentionLayer(FullAttention(False, attention_dropout=dropout,
                                      output_attention=output_attention), d_model, n_heads)
        self.attention2 = AttentionLayer(FullAttention(False, attention_dropout=dropout,
                                      output_attention=output_attention), d_model, n_heads)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

        # 1. Input Adapter: 由于 upstream state_transformer 是冻结的，
        # 这里的 input_projection 允许 policy_transformer 学习如何将固定的 embeddings 
        # 映射到适合当前任务的注意力空间。这对于打破“结果差不多”的僵局至关重要。
        self.input_projection = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU()
        )

        # 2. Remove Bottleneck: 原来的 32 维投影太小，可能导致信息丢失。
        # 将输出维度提升回 d_model (128)，保留更多特征信息。
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
        # relational_feature shape [B, N, D]
        # additional_feature shape [B, N, x]
        # return feature shape [B, N, D+x]

        # temporal_hybrid_feature, attn = self.attention(
        #     temporal_feature_long, temporal_feature_short, temporal_feature_short,
        #     attn_mask=mask
        # )
        # temporal_feature_long = temporal_feature_long + self.dropout(temporal_hybrid_feature)
        # temporal_feature = self.norm(temporal_feature_long)

        # temporal_relational_hybrid_feature, attn = self.attention2(
        #     temporal_feature, relational_feature, relational_feature,
        #     attn_mask=mask
        # )
        # temporal_feature = temporal_feature + self.dropout(temporal_relational_hybrid_feature)
        # hybrid_feature = self.norm(temporal_feature)

        # combined_feature = torch.cat((hybrid_feature, additional_feature), dim=-1) # [B, N, D+x]
        # return combined_feature

        # note: use without MAE update 
        # 0. Input Adaptation (Crucial for frozen upstream)
        relational_feature = self.input_projection(relational_feature)

        # 1. 处理关系特征 (Relational Hybrid)
        relational_hybrid_feature, attn = self.attention(
            relational_feature, relational_feature, relational_feature,
            attn_mask=mask
        )
        temporal_feature = relational_feature + self.dropout(relational_hybrid_feature)
        hybrid_feature = self.norm(temporal_feature)

        # 2. 提取日期特征 (根据 env 定义，最后 12 维是日期信息)        
        monthday_feature = additional_feature[:, :, :7]# 提取月份信息 (前 7 列)
        weekday_feature = additional_feature[:, :, -5:]# 提取星期信息 (最后 5 列)
        
        # 3. 组合逻辑与降维
        # 先对 hybrid feature 进行投影 (保持 d_model 维度)
        output_hybrid = self.projection(hybrid_feature) # [B, N, 128]
        
        # 拼接所有特征：Hybrid(128) + Month(7) + Week(5) = 140 dim
        # 这样 Actor/Critic 可以同时利用 复杂的股票关系特征 和 简单的日历特征
        combined_feature = torch.cat((output_hybrid, monthday_feature, weekday_feature), dim=-1) 
        
        return combined_feature






