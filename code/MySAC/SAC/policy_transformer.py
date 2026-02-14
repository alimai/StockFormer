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

        # 投影层：将 (d_model + 5) 维特征压缩到 32 维，减轻后续 MLP 压力
        self.out_dim = 32
        self.projection = nn.Sequential(
            nn.Linear(d_model + 5, self.out_dim),
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
        # temporal_feature_short=temporal_feature_long shape [B, N, D]
        # holding shape [B, N，x] or None
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
        combined_feature = torch.cat((hybrid_feature, monthday_feature, weekday_feature), dim=-1) # [B, N, 128+7+5]
        
        # 通过投影层降维到 32 维
        output_feature = self.projection(combined_feature) # [B, N, 32]
        
        return output_feature






