import torch
import torch.nn as nn
import torch.nn.functional as F


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Mixture-of-Experts
class ExpertNetwork(nn.Module):
    def __init__(self, input_dim=768, expert_dim=768):
        super(ExpertNetwork, self).__init__()
        # aspect expert层参数
        self.W_a = nn.Linear(input_dim, expert_dim)
        # sentiment expert层参数
        self.W_s = nn.Linear(input_dim, expert_dim)

        # gate aspect层参数
        self.W_g1 = nn.Linear(input_dim, expert_dim)
        # gate sentiment层参数
        self.W_g2 = nn.Linear(input_dim, expert_dim)

    def forward(self, H_shared):  # H_shared shape: [n, input_dim]
        # 确保输入张量在 GPU 上
        H_shared = H_shared.to(device)

        # 方面专家网络
        h_a = F.relu(self.W_a(H_shared))  # [n, expert_dim]
        # 情感专家网络
        h_s = F.relu(self.W_s(H_shared))  # [n, expert_dim]

        # 门控计算
        gate_a = torch.sigmoid(self.W_g1(H_shared))  # [n, expert_dim], 值在0-1之间
        gate_s = torch.sigmoid(self.W_g2(H_shared))  # [n, expert_dim]

        # 门控融合：加权融合专家输出和共享特征
        h_a = gate_a * h_a + (1 - gate_a) * H_shared  # [n, expert_dim]
        h_s = gate_s * h_s + (1 - gate_s) * H_shared  # [n, expert_dim]

        return h_a, h_s



##################################### Test #########################################
# from BiAESC.BaseModel.BiAffine import BiAffine, BERT_Embedding
#
# sentence = 'The chocolate cake is delicious but the donuts are terrible'
# text_graph, G, rels, pos_tags = BiAffine(sentence)
#
# pooling_feature, dep_feature = BERT_Embedding(sentence, pos_tags, rels)
#
# # 确保将输入张量放到与模型相同的设备上
# pooling_feature = pooling_feature.to(device)
#
# model = ExpertNetwork(input_dim=768, expert_dim=768).to(device)  # 将模型放到GPU上
# h_a, h_s = model(pooling_feature)
#
# print(h_a.shape)  # torch.Size([10, 768])
# print(h_s.shape)  # torch.Size([10, 768])

