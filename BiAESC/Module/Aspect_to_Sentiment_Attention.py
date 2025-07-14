import torch
import torch.nn as nn
import networkx as nx
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class A2S_LocalAwareness(nn.Module):
    def __init__(self, hidden_dim=768, num_heads=12, dropout=0.3):
        super(A2S_LocalAwareness, self).__init__()

        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)

        # Self-Attention: 用于获取 Query 表示（基于 h_a）, self_Attention模型初始化
        self.self_attn = torch.nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)

        # 4. LayerNorm + Dropout
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)


    def forward(self, h_a, h_s, G):
        # 将输入数据转移到 GPU
        h_a = h_a.to(device)
        h_s = h_s.to(device)

        #获取依存距离矩阵
        num_nodes = G.number_of_nodes()
        # Calculate the SRD values by node distance matrix(Phan et al.)
        dep_dis_matrix = torch.zeros((num_nodes, num_nodes), device=device)
        # Calculate the shortest path length between two nodes
        for i in range(num_nodes):
            for j in range(num_nodes):
                paths = nx.shortest_path_length(G, source=i, target=j, weight='weight')
                dep_dis_matrix[i, j] = paths

        # Md->Wd 使用高斯衰减函数处理依存距离矩阵
        sigma = 1.0
        Wd = torch.exp(- (dep_dis_matrix ** 2) / (2 * sigma ** 2))
        # print(SRD_weight)

        # 计算ha的注意力权重
        ha_feature = h_a.unsqueeze(0).to(device)
        attn_output, attn_weights = self.self_attn(ha_feature, ha_feature,ha_feature)
        Wa = attn_weights.squeeze().to(device)

        #SRD_weight_matrix  结构权重 × 注意力分数
        Wf = Wd * Wa

        # 设置动态阈值：均值 + γ × 标准差
        mean_val = torch.mean(Wf)
        std_val = torch.std(Wf)
        gamma = 0.5  # 可调参数（越大越保守）
        threshold = mean_val + gamma * std_val

        #对Wf权重矩阵进行权重线性衰减
        Ww = torch.ones_like(Wf, device=device)
        for i in range(num_nodes):
            for j in range(num_nodes):
                weight = Wf[i, j]
                if weight < threshold:
                    # 对低权重做线性衰减：越小越弱化，最低接近0
                    Ww[i, j] = weight / threshold  # 归一化成 [0,1]
                else:
                    Ww[i, j] = 1.0  # 保留或增强

        #使用Ww引导情感增强向量hs的计算
        # # 1. 直接用Ww加权h_s
        # h_s_weighted = torch.matmul(Ww, h_s)  # [n, d]

        # 2. 用Ww引导self-attention计算
        n, d = h_s.shape
        Q = self.query(h_s).to(device)  # [n, d]
        K = self.key(h_s).to(device) # [n, d]
        V = self.value(h_s).to(device)  # [n, d]

        # 计算注意力分数 Q K^T / sqrt(d)
        scores = torch.matmul(Q, K.transpose(0, 1)) / (d ** 0.5)  # [n, n]
        # 用 Ww 引导注意力分数，点乘
        guided_scores = scores * Ww  # [n, n]
        # softmax 归一化
        attn_weights = F.softmax(guided_scores, dim=-1)  # [n, n]
        # 加权 V
        attn_output = torch.matmul(attn_weights, V)  # [n, d]
        # 残差连接 + 层归一化
        h_sl = self.layer_norm(h_s + self.dropout(attn_output))  # [n, d]

        # # 3. 拼接
        # concat_output = torch.cat([h_s_weighted, attn_output], dim=-1)  # [n, 2*d]
        #
        # # 使用 MaxPool1d 做降维：1536 → 768
        # pool = nn.MaxPool1d(kernel_size=2, stride=2)
        # concat_output = pool(concat_output)  # → [n, 768]

        return h_sl



################################  Test  ##############################
# from BiAESC.BaseModel.BiAffine import BiAffine, BERT_Embedding
# from BiAESC.BaseModel.Expert_network import ExpertNetwork
#
# sentence ='The chocolate cake is delicious but the donuts are terrible'
# text_graph, G, rels, pos_tags = BiAffine(sentence)
# pooling_feature, dep_feature = BERT_Embedding(sentence, pos_tags, rels)
#
# # 将输入特征转移到 GPU
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
# # aspect_guided sentiment local attention
# A2S_module = A2S_LocalAwareness(hidden_dim=768, num_heads=12, dropout=0.3).to(device)
# h_ls = A2S_module(h_a, h_s, G)
# print(h_ls.shape)