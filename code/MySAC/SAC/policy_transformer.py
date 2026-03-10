import torch
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
        add_dim = additional_dim-1#additional_dim if config.struct_base_flag else additional_dim-1
        
        atten_dim = hidden_out // 4
        self.attention1 = AttentionLayer(FullAttention(False, attention_dropout=dropout,
                                      output_attention=output_attention), atten_dim, n_heads)
        self.attention2 = AttentionLayer(FullAttention(False, attention_dropout=dropout,
                                      output_attention=output_attention), atten_dim, n_heads)
        self.norm1 = nn.LayerNorm(atten_dim)
        self.norm2 = nn.LayerNorm(atten_dim)
        self.dropout = nn.Dropout(dropout)

        # 特征融合后投影回 hidden_out
        self.projection_input = nn.Sequential(
            nn.Linear(hidden_out + add_dim, atten_dim),
            nn.LayerNorm(atten_dim),
            nn.GELU()#nn.Sigmoid()#
        )
        self.projection_input2 = nn.Sequential(
            nn.Linear(hidden_out, atten_dim),
            nn.LayerNorm(atten_dim),
            nn.GELU()#nn.Sigmoid()
        )
        self.projection_input3 = nn.Sequential(
            nn.Linear(hidden_out, atten_dim),
            nn.LayerNorm(atten_dim),
            nn.GELU()#nn.Sigmoid()#
        )
        
        tmp_input_dim = atten_dim * 2
        tmp_out_dim = atten_dim * 2 - add_dim # 32 * 2 - 20 = 44
        self.projection_output = nn.Sequential(
            nn.Linear(tmp_input_dim, tmp_out_dim),
            nn.LayerNorm(tmp_out_dim),
            nn.GELU()#nn.Sigmoid()#
        )

        self.optimizer = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=1e-2)
        
    def forward(self, relational_feature, temporal_feature_short, temporal_feature_long, additional_feature, mask=None):
        # relational_feature: [B, N, 128] (From MAE)
        # additional_feature: [B, N, add_dim] (Tech + Date)
        
        if config.struct_base_flag:
            update_type = 1
        else:
            update_type = 2
            
        # 处理输入特征 (Refinement)
        add_feature = additional_feature[:, :, :-1] #去掉holding部分
        relation_fused_input = torch.cat([relational_feature, add_feature], dim=-1) #temporal_feature_short#relational_feature
        if update_type==1:
            temporal_input_long = self.dropout(self.projection_input2(temporal_feature_long))
            temporal_input_short = self.dropout(self.projection_input3(temporal_feature_short))
            relation_fused_input = self.projection_input(relation_fused_input)
        else:
            temporal_input_long = self.projection_input2(temporal_feature_long)
            temporal_input_short = self.projection_input3(temporal_feature_short)
            relation_fused_input = self.dropout(self.projection_input(relation_fused_input))

        tmp_feature_1, attn = self.attention1(
            temporal_input_long, relation_fused_input, relation_fused_input,
            attn_mask=mask
        )
        temporal_attn_long = temporal_input_long + self.dropout(tmp_feature_1)
        temporal_adapt_long = self.norm1(temporal_attn_long)

        tmp_feature_2, attn = self.attention2(
            temporal_input_short, relation_fused_input, relation_fused_input,
            attn_mask=mask
        )
        temporal_attn_short = temporal_input_short + self.dropout(tmp_feature_2)
        temporal_adapt_short = self.norm2(temporal_attn_short)


        # 2) Output Processing
        hybrid_feature = torch.cat([temporal_adapt_long, temporal_adapt_short], dim=-1) 
        fused_output_adapted = self.projection_output(hybrid_feature) # [B, N, 128]
        #fused_output_adapted = temporal_adapt_long + temporal_adapt_short
        # 再次拼接: Processed Context (128)与附加上下文（Tech + Date）
        if update_type==2:
            combined_feature = self.dropout(torch.cat((fused_output_adapted, add_feature), dim=-1))  # [B, N, 128+add_dim]
        else:
            combined_feature = torch.cat((fused_output_adapted, self.dropout(add_feature)), dim=-1)  # [B, N, 128+add_dim]

        return combined_feature

    def forward_front(self, relational_feature, temporal_feature_short, temporal_feature_long, additional_feature, mask=None):
        # relational_feature: [B, N, 128] (From MAE)
        # additional_feature: [B, N, add_dim] (Tech + Date)
        
        # 1) 处理输入特征 (Refinement)
        temporal_fused_input = torch.cat([temporal_feature_long, additional_feature], dim=-1) #temporal_feature_short#relational_feature
        temporal_input_adapted = self.projection_input(temporal_fused_input) # [B, N, 128]

        temporal_feature_short_adapted = self.projection_input2(temporal_feature_short) # [B, N, 128]
        tmp_feature_1, attn = self.attention1(
            temporal_input_adapted, temporal_feature_short_adapted, temporal_feature_short_adapted,
            attn_mask=mask
        )
        temporal_feature_attn = temporal_input_adapted + self.dropout(tmp_feature_1)
        temporal_hybrid_feature = self.norm1(temporal_feature_attn)

        relational_feature_adapted = self.projection_input3(relational_feature) # [B, N, 128]
        tmp_feature_2, attn = self.attention2(
            temporal_hybrid_feature, relational_feature_adapted, relational_feature_adapted,
            attn_mask=mask
        )
        feature_attn = temporal_hybrid_feature + self.dropout(tmp_feature_2)
        hybrid_feature = self.norm2(feature_attn)

        return hybrid_feature

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