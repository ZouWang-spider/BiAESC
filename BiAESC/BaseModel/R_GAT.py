import torch
import torch.nn as nn
import torch.nn.functional as F

#此代码来自R-GAT的原始论文公式：Relational graph attention network for aspect-based sentiment analysis
#原理分别使用依存特征权重和注意力权重来实现GAT的计算，然后将两种计算结果进行拼接与线性映射

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#multi-head attention + dep  R-GAT
class RelationalGraphAttentionLayer(nn.Module):
    def __init__(self, in_dim, out_dim, r_dim, num_heads):
        super().__init__()
        self.M = num_heads
        self.out_dim = out_dim

        self.W_m = nn.ModuleList([nn.Linear(in_dim, out_dim, bias=False) for _ in range(self.M)])

        # Relation attention branch
        self.W_r = nn.Linear(r_dim, out_dim, bias=False)
        self.bm1 = nn.Parameter(torch.zeros(out_dim))
        self.W_m2 = nn.Linear(out_dim, 1, bias=False)
        self.bm2 = nn.Parameter(torch.zeros(1))

        # Node attention branch
        self.attn_fc = nn.Linear(2 * out_dim, 1, bias=False)
        self.leaky_relu = nn.LeakyReLU(0.2)

        # Final projection after concat
        self.linear_out = nn.Linear(2 * out_dim * self.M, out_dim)  # project concat(h_att || h_rel)

    def forward(self, h, edge_index, r_ij):
        N = h.size(0)
        src, dst = edge_index[0], edge_index[1]

        h_att_all = []
        h_rel_all = []

        for m in range(self.M):
            Wh = self.W_m[m](h)  # [N, out_dim]
            Wh_src = Wh[src]
            Wh_dst = Wh[dst]

            ## -- β_ij: relation attention
            rel_score = F.relu(self.W_r(r_ij) + self.bm1)
            rel_score = self.W_m2(rel_score) + self.bm2
            rel_score = rel_score.squeeze(-1)
            rel_alpha = torch.exp(rel_score)
            rel_denom = torch.zeros(N, device=h.device).index_add(0, dst, rel_alpha)
            beta_ij = rel_alpha / (rel_denom[dst] + 1e-9)
            rel_msg = Wh_src * beta_ij.unsqueeze(-1)
            rel_agg = torch.zeros(N, self.out_dim, device=h.device)
            rel_agg.index_add_(0, dst, rel_msg)
            h_rel_all.append(rel_agg)

            ## -- α_ij: node attention
            attn_input = torch.cat([Wh_src, Wh_dst], dim=-1)
            attn_score = self.leaky_relu(self.attn_fc(attn_input).squeeze(-1))
            attn_alpha = torch.exp(attn_score)
            attn_denom = torch.zeros(N, device=h.device).index_add(0, dst, attn_alpha)
            alpha_ij = attn_alpha / (attn_denom[dst] + 1e-9)
            attn_msg = Wh_src * alpha_ij.unsqueeze(-1)
            attn_agg = torch.zeros(N, self.out_dim, device=h.device)
            attn_agg.index_add_(0, dst, attn_msg)
            h_att_all.append(attn_agg)

        # Concatenate across heads: [N, M*out_dim]
        h_att = torch.cat(h_att_all, dim=-1)
        h_rel = torch.cat(h_rel_all, dim=-1)

        # Final concat and projection
        x = torch.cat([h_att, h_rel], dim=-1)  # [N, 2*M*out_dim]
        h_out = F.relu(self.linear_out(x))     # [N, out_dim]

        return h_out


class RGAT_Model(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, r_dim, num_heads):
        super().__init__()
        self.rgat1 = RelationalGraphAttentionLayer(in_dim, hidden_dim, r_dim, num_heads)
        self.rgat2 = RelationalGraphAttentionLayer(hidden_dim, out_dim, r_dim, num_heads)

    def forward(self, node_feats, edge_index, rel_feats):
        # 确保输入张量在 GPU 上
        node_feats = node_feats.to(device)
        edge_index = edge_index.to(device)
        rel_feats = rel_feats.to(device)

        x = self.rgat1(node_feats, edge_index, rel_feats)  # [N, hidden_dim]
        x = self.rgat2(x, edge_index, rel_feats)           # [N, out_dim]
        return x



############################################### Test ###########################################
# from BiAESC.BaseModel.BiAffine import BiAffine, BERT_Embedding
#
# # 示例输入
# sentence = "The food is delicious but prices are high"
# text_graph, G, rels, pos_tags = BiAffine(sentence)
#
# # 特征提取，其中pooling_feature为h_se和h_ls的拼接特征
# pooling_feature, dep_feature = BERT_Embedding(sentence, pos_tags, rels)
# # pooling_feature: [N, 768], dep_feature: [E, 768]
#
# # 转换图结构为 tensor，确保 edge_index 是 LongTensor [2, E]
# edge_index = torch.tensor(text_graph, dtype=torch.long, device=device)   # [2, E]
# rel_feats = dep_feature.to(device)                   # [E, r_dim]
# node_feats = pooling_feature.to(device)                                         # [N, in_dim]
#
# # 参数设置
# hidden_dim = 128
# out_dim = 768
# num_heads = 12
# # 初始化
# rgat_model = RGAT_Model(in_dim=768, hidden_dim=hidden_dim, out_dim=out_dim, r_dim=768, num_heads=num_heads).to(device)
#
#
# #模型调用
# out = rgat_model(node_feats, edge_index, rel_feats)  # [N, out_dim]
# print(out.shape)