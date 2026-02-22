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
            nn.LayerNorm(d_model),
            nn.GELU()
        )

        # 时序特征融合后投影回 d_model
        self.projection_temporal = nn.Sequential(
            nn.Linear(d_model * 2 + additional_dim, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model)
        )

        # Gated Fusion Mechanism
        self.fusion_gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid()
        )

        # 2. Output Projection: 
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
        
        # Early Fusion & Adaptation
        # 拼接关系特征与附加上下文（Tech + Date）并投影回 d_model
        relational_fused_input = torch.cat([relational_feature, additional_feature], dim=-1) #temporal_feature_short#relational_feature
        relational_input_adapted = self.input_projection(relational_fused_input) # [B, N, 128]

        # Temporal Feature Fusion
        # 拼接长短期时序特征与附加上下文，并投影回 d_model
        temporal_fused_input = torch.cat([temporal_feature_short, temporal_feature_long, additional_feature], dim=-1)
        temporal_input_adapted = self.projection_temporal(temporal_fused_input) # [B, N, 128]

        # Gated Fusion: Dynamically combine Relational and Temporal features
        # Calculate gate values based on both inputs
        gate_input = torch.cat([relational_input_adapted, temporal_input_adapted], dim=-1)
        gate = self.fusion_gate(gate_input)
        
        # Weighted sum
        fused_input = gate * relational_input_adapted + (1 - gate) * temporal_input_adapted

        # 处理关系特征 (Refinement)
        tmp_feature_3, attn = self.attention3(
            fused_input, fused_input, fused_input,
            attn_mask=mask
        )
        # Residual Connection & Independent LayerNorm
        relational_feature_attn = fused_input + self.dropout(tmp_feature_3)
        relational_hybrid_feature = self.norm3(relational_feature_attn)

        # Output Processing
        fused_output_adapted = self.projection(relational_hybrid_feature) # [B, N, 128]
        
        # Late Fusion (Skip Connection with Clean Context)
        # 再次拼接: Processed Context (128)与附加上下文（Tech + Date）
        combined_feature = torch.cat((fused_output_adapted, additional_feature), dim=-1)  # [B, N, 128+additional_dim]
        return combined_feature


