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
        self.attention = AttentionLayer(FullAttention(False, attention_dropout=dropout,
                                      output_attention=output_attention), d_model, n_heads)
        self.attention2 = AttentionLayer(FullAttention(False, attention_dropout=dropout,
                                      output_attention=output_attention), d_model, n_heads)
        self.attention3 = AttentionLayer(FullAttention(False, attention_dropout=dropout,
                                      output_attention=output_attention), d_model, n_heads)
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        # self.norm = nn.LayerNorm(d_model)

        # 1. Input Adapter (Early Fusion): 
        # 输入维度: d_model (128) + additional_dim (Tech + Date)
        self.input_projection = nn.Sequential(
            nn.Linear(d_model + additional_dim, d_model),
            nn.GELU(),
            #nn.LayerNorm(d_model)
        )

        # 2. Output Projection: 
        self.out_dim = d_model 
        self.projection = nn.Sequential(
            nn.Linear(d_model, self.out_dim),
            nn.GELU(),
            #nn.LayerNorm(self.out_dim)
        )
        self.projection2 = nn.Sequential(
            nn.Linear(d_model * 2, additional_dim),
            nn.GELU(),
            nn.LayerNorm(additional_dim)
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
        
        # Early Fusion & Adaptation
        # 拼接关系特征与附加上下文（Tech + Date）并投影回 d_model
        relational_fused_input = torch.cat([relational_feature, additional_feature], dim=-1) #temporal_feature_short#relational_feature
        relational_fused_adapted = self.input_projection(relational_fused_input) # [B, N, 128]

        # 处理时序特征 (Short)
        temporal_feature_1, attn = self.attention(
            temporal_feature_short, relational_fused_adapted, relational_fused_adapted,
            attn_mask=mask
        )
        # Residual Connection & Independent LayerNorm
        tmp_feature_1 = temporal_feature_short + self.dropout(temporal_feature_1)
        #temporal_hybrid_feature_1 = self.norm1(tmp_feature_1)

        # 处理时序特征 (Long)
        temporal_feature_2, attn = self.attention2(
            temporal_feature_long, relational_fused_adapted, relational_fused_adapted,
            attn_mask=mask
        )
        # Residual Connection & Independent LayerNorm
        tmp_feature_2 = temporal_feature_long + self.dropout(temporal_feature_2)
        #temporal_hybrid_feature_2 = self.norm2(tmp_feature_2)

        # 处理关系特征 (Refinement)
        relational_feature_1, attn = self.attention3(
            relational_feature, relational_fused_adapted, relational_fused_adapted,
            attn_mask=mask
        )
        # Residual Connection & Independent LayerNorm
        tmp_feature_3 = relational_feature + self.dropout(relational_feature_1)
        #relational_hybrid_feature = self.norm3(tmp_feature_3)
        relational_output_adapted = self.projection(tmp_feature_3) # [B, N, 128]

        # Early Fusion & Adaptation
        # 拼接股票特征与附加上下文（Tech + Date）并投影回 d_model
        temporal_fused_output = torch.cat([tmp_feature_1, tmp_feature_2], dim=-1) #temporal_feature_short/long
        temporal_output_adapted = self.projection2(temporal_fused_output) # [B, N, 128]
        
        # Late Fusion (Skip Connection with Clean Context)
        # 再次拼接: Processed Context (128)与附加上下文（Tech + Date）
        combined_feature = torch.cat((relational_output_adapted, temporal_output_adapted, additional_feature), dim=-1)  # [B, N, 128+128+additional_dim]
        return combined_feature


