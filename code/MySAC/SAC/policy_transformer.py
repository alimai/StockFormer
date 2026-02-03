from email import iterators
from Transformer.models.embed import DataEmbedding
from torch import nn
import torch
import itertools

from Transformer.models.attn import AttentionLayer, FullAttention
from Transformer.models.layer import MultiHeadAttention

import pdb


class policy_transformer_stock_atten2(nn.Module): # attention(long, short), attention(hybrid, relational) 
    def __init__(self, d_model=128, n_heads=4, dropout=0.0, lr=0.0001, output_attention=False, device=None):
        super().__init__()
        self.attention = AttentionLayer(FullAttention(False, attention_dropout=dropout,
                                      output_attention=output_attention), d_model, n_heads)
        self.attention2 = AttentionLayer(FullAttention(False, attention_dropout=dropout,
                                      output_attention=output_attention), d_model, n_heads)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

        # 检测GPU可用性并决定使用GPU还是CPU
        if device is None:
            if torch.cuda.is_available():
                device = 'cuda:0'
            else:
                device = 'cpu'

        self.optimizer = torch.optim.Adam(itertools.chain(self.attention.parameters(), self.attention2.parameters()), lr=lr)
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

        hybrid_feature = self.norm(relational_feature)
        combined_feature = torch.cat((hybrid_feature, additional_feature), dim=-1) # [B, N, D+x]

        return hybrid_feature#combined_feature#






