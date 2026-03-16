import torch
import random
from torch import nn
from utils import config

from Transformer.models.embed import DataEmbedding
from Transformer.models.attn import AttentionLayer, FullAttention
from Transformer.models.layer import MultiHeadAttention

#import pdb


class policy_transformer_stock_atten2(nn.Module): # attention(long, short), attention(hybrid, relational) 
    def __init__(self, hidden_out=128, n_heads=4, dropout=0.1, lr=0.0001, output_attention=False, device=None, additional_dim=20):
        super().__init__()
        
        # 检测GPU可用性并决定使用GPU还是CPU
        if device is None:
            if torch.cuda.is_available():
                device = 'cuda:0'
            else:
                device = 'cpu'
        self.device = device
        self.hidden_out = hidden_out
        self.atten_dim = hidden_out // config.scale_ratio # attention内部维度，通常设置为hidden_out的整数倍
        self.add_dim = additional_dim#-1 #去掉holding部分
        
        self.attention1 = AttentionLayer(FullAttention(False, attention_dropout=dropout,
                                      output_attention=output_attention), self.atten_dim, n_heads)
        self.attention2 = AttentionLayer(FullAttention(False, attention_dropout=dropout,
                                      output_attention=output_attention), self.atten_dim, n_heads)
        self.norm0 = nn.LayerNorm(self.atten_dim)
        self.norm1 = nn.LayerNorm(self.atten_dim)
        self.norm2 = nn.LayerNorm(self.atten_dim)
        self.dropout = nn.Dropout(dropout)

        if config.fix_seed % 2 == 0:
            self.dropout1 = nn.Dropout(dropout*1.5)
            self.dropout2 = nn.Dropout(dropout*0.1)
        elif config.fix_seed % 2 == 1:
            self.dropout1 = nn.Dropout(dropout*0.1)
            self.dropout2 = nn.Dropout(dropout*1.5)
        else:
            self.dropout1 = nn.Dropout(dropout)
            self.dropout2 = nn.Dropout(dropout)

        # 特征融合后投影回 hidden_out
        self.projection_input = nn.Sequential(
            nn.Linear(hidden_out + self.add_dim, self.atten_dim),
            # nn.GELU(),#nn.Sigmoid()#
            self.norm0
        )
        self.projection_input2 = nn.Sequential(
            nn.Linear(hidden_out, self.atten_dim),
            # nn.GELU(),#nn.Sigmoid()
            self.norm0
        )
        self.projection_input3 = nn.Sequential(
            nn.Linear(hidden_out, self.atten_dim),
            # nn.GELU(),#nn.Sigmoid()#
            self.norm0
        )

        self.optimizer = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=1e-2)
        
    def forward(self, relational_feature, temporal_feature_short, temporal_feature_long, additional_feature, mask=None):
        # relational_feature: [B, N, 128] (From MAE)
        # additional_feature: [B, N, add_dim] (Tech + Date)
              
        # 处理输入特征 (Refinement)
        add_feature = additional_feature#[:, :, :-1] #去掉holding部分
        if config.struct_type % 2 == 1:
            base_feature = temporal_feature_long
            minor_feature = relational_feature
        else:
            base_feature = relational_feature
            minor_feature = temporal_feature_long

        base_feature_fused = torch.cat([base_feature, add_feature], dim=-1) #temporal_feature_short#relational_feature
        base_feature_input = self.dropout1(self.projection_input(base_feature_fused))
        if self.atten_dim != self.hidden_out:
            temporal_input_short = self.projection_input2(temporal_feature_short)
            minor_feature_input = self.projection_input3(minor_feature)
        else:
            temporal_input_short = temporal_feature_short
            minor_feature_input = minor_feature

        # Attention parts
        tmp_feature_1, attn = self.attention1(
            base_feature_input, temporal_input_short, temporal_input_short,
            attn_mask=mask
        )
        if config.struct_type <= 2:
            tmp_feature_2, attn = self.attention2(
                base_feature_input, minor_feature_input, minor_feature_input,
                attn_mask=mask
            )
            hybrid_feature = self.dropout2(base_feature_input) + self.dropout(tmp_feature_1) + self.dropout(tmp_feature_2)
        else:            
            temp_atten_adapt = self.norm1(base_feature_input + self.dropout(tmp_feature_1))
            tmp_feature_2, attn = self.attention2(
                temp_atten_adapt, minor_feature_input, minor_feature_input,
                attn_mask=mask
            )
            hybrid_feature = self.dropout2(temp_atten_adapt) + self.dropout(tmp_feature_2)
        hybrid_feature_adapted = self.norm2(hybrid_feature)

        return hybrid_feature_adapted      

    def forward_orig(self, relational_feature, temporal_feature_short, temporal_feature_long, holding, mask=None):
        # relational_feature shape [B, N, D]
        # temporal_feature_short=temporal_feature_long shape [B, N, D]
        # holding shape [B, N] or None
        # return feature shape [B, N, D+1]

        temporal_hybrid_feature, attn = self.attention(
            temporal_feature_long, temporal_feature_short, temporal_feature_short,
            attn_mask=mask
        )
        temporal_feature_long = temporal_feature_long + self.dropout(temporal_hybrid_feature)
        temporal_feature = self.norm(temporal_feature_long)

        temporal_relational_hybrid_feature, attn = self.attention2(
            temporal_feature, relational_feature, relational_feature,
            attn_mask=mask
        )

        temporal_feature = temporal_feature + self.dropout(temporal_relational_hybrid_feature)
        hybrid_feature = self.norm(temporal_feature)

        combined_feature = torch.cat((hybrid_feature, holding), dim=-1) # [B, N, D+1]

        return combined_feature