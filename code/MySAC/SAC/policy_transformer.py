from email import iterators
from Transformer.models.embed import DataEmbedding
from torch import nn
import torch
import itertools

from Transformer.models.attn import AttentionLayer, FullAttention
from Transformer.models.layer import MultiHeadAttention

#import pdb


class policy_transformer_stock_atten2(nn.Module): # attention(long, short), attention(hybrid, relational) 
    def __init__(self, d_model=128, n_heads=4, dropout=0.1, lr=0.0001, output_attention=False, device=None, additional_dim=20):
        super().__init__()
        self.attention1 = AttentionLayer(FullAttention(False, attention_dropout=dropout,
                                      output_attention=output_attention), d_model, n_heads)
        self.attention2 = AttentionLayer(FullAttention(False, attention_dropout=dropout,
                                      output_attention=output_attention), d_model, n_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        # 时序特征融合后投影回 d_model
        self.projection_temporal = nn.Sequential(
            nn.Linear(d_model * 2 + additional_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU()
        )

        # 关系特征与附加上下文融合: 
        # 输入维度: d_model (128) + additional_dim (Tech + Date)
        self.projection_relational = nn.Sequential(
            nn.Linear(d_model + additional_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU()
        )

        # # Gated Fusion Mechanism
        # self.fusion_gate = nn.Sequential(
        #     nn.Linear(d_model * 2, d_model),
        #     nn.Sigmoid()
        # )

        # 2. Output Projection: 
        self.temporal_out_dim = 1#additional_dim# // 2 #20//2=10
        self.Linear = nn.Linear(d_model, self.temporal_out_dim)
        self.projection_output = nn.Sequential(
            nn.Linear(d_model + self.temporal_out_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU()
        )

        # 检测GPU可用性并决定使用GPU还是CPU
        if device is None:
            if torch.cuda.is_available():
                device = 'cuda:0'
            else:
                device = 'cpu'
        self.device = device

        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=1e-4)
        
    def forward(self, relational_feature, temporal_feature_short, temporal_feature_long, additional_feature, mask=None):
        # relational_feature: [B, N, 128] (From MAE)
        # additional_feature: [B, N, additional_dim] (Tech + Date)
        
        # 1) 处理时序特征 (Refinement)
        temporal_fused_input = torch.cat([temporal_feature_short, temporal_feature_long, additional_feature], dim=-1)
        temporal_input_adapted = self.projection_temporal(temporal_fused_input) # [B, N, 128]

        tmp_feature_1, attn = self.attention1(
            temporal_input_adapted, temporal_input_adapted, temporal_input_adapted,
            attn_mask=mask
        )
        temporal_feature_attn = temporal_input_adapted + self.dropout(tmp_feature_1)
        temporal_hybrid_feature = self.norm1(temporal_feature_attn)


        # 2) 处理关系特征 (Refinement)
        relational_fused_input = torch.cat([relational_feature, additional_feature], dim=-1) #temporal_feature_short#relational_feature
        relational_input_adapted = self.projection_relational(relational_fused_input) # [B, N, 128]

        tmp_feature_2, attn = self.attention2(
            relational_input_adapted, relational_input_adapted, relational_input_adapted,
            attn_mask=mask
        )
        relational_feature_attn = relational_input_adapted + self.dropout(tmp_feature_2)
        relational_hybrid_feature = self.norm2(relational_feature_attn)


        # 3) Output Processing
        temporal_output = self.Linear(temporal_hybrid_feature) # [B, N, 1]
        fused_output = torch.cat((relational_hybrid_feature, temporal_output), dim=-1)
        # fused_output = relational_hybrid_feature + temporal_hybrid_feature
        fused_output_adapted = self.projection_output(fused_output) # [B, N, 128]
        
        # Late Fusion (Skip Connection with Clean Context)
        # 再次拼接: Processed Context (128)与附加上下文（Tech + Date）
        combined_feature = torch.cat((fused_output_adapted, additional_feature), dim=-1)  # [B, N, 128+additional_dim]
        return combined_feature


