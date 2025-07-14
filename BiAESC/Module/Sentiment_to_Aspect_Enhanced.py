import torch
import torch.nn as nn
from BiAESC.BaseModel.GCN import GCNModel
from BiAESC.BaseModel.CrossTransformer import Cross_Transformer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class S2A_SemanticRefinement(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=768, num_layers=2, num_heads=12, dropout=0.3):
        super(S2A_SemanticRefinement, self).__init__()

        # 1. GCN: 输入为拼接后的 [n, d], GCN模型初始化
        self.gcn = GCNModel(input_dim, hidden_dim, num_layers)

        # 2. Self-Attention: 用于获取 Query 表示（基于 h_a）, self_Attention模型初始化
        self.self_attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout, batch_first=True)

        # 3. Cross Attention 模块（自定义）, Multi_Cross_Attention模型初始化
        self.cross_attn = Cross_Transformer(hidden_dim=hidden_dim, num_heads=num_heads)

        # 4. LayerNorm + Dropout
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h_a, h_s, text_graph):
        # Step 1: 拼接 h_a 和 h_s → [n, 2d]
        h_c = torch.cat([h_a, h_s], dim=-1)

        # 使用 MaxPool1d 做降维：1536 → 768
        pool = nn.MaxPool1d(kernel_size=2, stride=2)
        h_concat = pool(h_c)  # → [n, 768]

        # Step 2: GCN 提取 Aspect-Sentiment 依赖语义 → [n, d]
        gcn_output = self.gcn(h_concat, text_graph)  # 输出为 [n, d]

        # Step 3: Self-Attention 作用在 h_a 上 → 获取 Query 表示
        h_a_input = h_a.unsqueeze(0)  # [1, n, d]
        query_self, _ = self.self_attn(h_a_input, h_a_input, h_a_input)  # [1, n, d]

        # Step 4: Cross Attention：Q = SelfAtt(Q), K/V = GCN 输出
        cross_output = self.cross_attn(query_self.squeeze(0), gcn_output, gcn_output)  # [n, d]

        # Step 5: 残差连接 + LayerNorm
        h_ae = self.layer_norm(query_self.squeeze(0) + self.dropout(cross_output))  # [n, d]

        return h_ae  # [n, d]



################################  Test  ##############################
# from BiAESC.BaseModel.BiAffine import BiAffine, BERT_Embedding
# from BiAESC.BaseModel.Expert_network import ExpertNetwork
#
# sentence ='The chocolate cake is delicious but the donuts are terrible'
# text_graph, G, rels, pos_tags = BiAffine(sentence)
# pooling_feature, dep_feature = BERT_Embedding(sentence, pos_tags, rels)
#
# pooling_feature = pooling_feature.to(device)
#
# #专家网络层
# expert_network = ExpertNetwork(input_dim=768, expert_dim=768).to(device)
# h_a, h_s = expert_network(pooling_feature)
#
# # 确保h_a 和 h_s 在 GPU 上
# h_a = h_a.to(device)
# h_s = h_s.to(device)
#
# #aspect_to_sentiment module
# S2A_module = S2A_SemanticRefinement(input_dim=768, hidden_dim=768, num_layers=2, num_heads=12, dropout=0.3).to(device)
# h_se = S2A_module(h_a, h_s, text_graph)  # [n, d]
# print(h_se.shape)
