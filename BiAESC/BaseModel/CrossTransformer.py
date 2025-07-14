import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Cross_Transformer(nn.Module):
    def __init__(self, hidden_dim, num_heads=12):
        super(Cross_Transformer, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        # 使用 PyTorch 自带的 MultiheadAttention 实现 Cross Attention
        self.multihead_attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)

        # FFN + ReLU
        self.mlp_layer = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, hidden_dim)
        )

        # Layer Normalization
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, query, key, value):
        """
        query, key, value: [n, d]  → reshape to [1, n, d] for batch_first
        """
        query = query.unsqueeze(0).to(device)  # [1, n, d]
        key = key.unsqueeze(0).to(device)
        value = value.unsqueeze(0).to(device)

        # 多头注意力计算
        attn_output, _ = self.multihead_attn(query, key, value)  # [1, n, d]

        # 残差连接 + LN
        attn_output = self.layer_norm(attn_output + query)  # [1, n, d]

        # FFN + 残差连接 + LN
        ffn_output = self.mlp_layer(attn_output)  # [1, n, d]
        final_output = self.layer_norm(ffn_output + attn_output)  # [1, n, d]

        return final_output.squeeze(0)  # [n, d]


##############################  Test  ###################################
# # 模拟输入
# ha = torch.randn(10, 768)  # Query
# hg = torch.randn(10, 768)  # Key, Value
#
# # 实例化模型
# cross_attn_module = Cross_Transformer(hidden_dim=768, num_heads=12)
#
# # 前向计算
# final_output = cross_attn_module(ha, hg, hg)
# print(final_output.shape)  # torch.Size([10, 768])
