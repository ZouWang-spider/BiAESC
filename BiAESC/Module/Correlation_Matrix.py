import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#构建结构感知的相关矩阵（输出为 [n, n, d]）
def build_dependency_guided_corr_tensor(h_cl, edge_index, dep_rel_feats, default_weight_scale=0.1):

    n, d = h_cl.size()
    device = h_cl.device
    corr_tensor = torch.zeros(n, n, d, device=device)

    # 建立边索引映射 {(i,j): rel_vec}
    edge_dict = {}
    for idx in range(edge_index.size(1)):
        src = edge_index[0, idx].item()
        dst = edge_index[1, idx].item()
        edge_dict[(src, dst)] = dep_rel_feats[idx].to(device)  # [d]

    # 默认依存权重向量（平均后缩放）
    default_rel_vec = torch.mean(dep_rel_feats, dim=0) * default_weight_scale  # [d]
    default_rel_vec = default_rel_vec.to(device)  # 确保默认向量在 GPU 上

    for i in range(n):
        for j in range(n):
            h_i = h_cl[i]
            h_j = h_cl[j]
            pair_feat = torch.cat([h_i, h_j], dim=-1)  # [2d]
            # 使用 MaxPool1d 进行降维（kernel_size=2, stride=2）：输出 [1, 1, 768]
            pair_feat = pair_feat.unsqueeze(0).unsqueeze(0)  # [1, 1, 1536]
            fused_feat = F.max_pool1d(pair_feat, kernel_size=2, stride=2).squeeze()  # [768]

            rel_vec = edge_dict.get((i, j), default_rel_vec)  # [d]

            # 结构控制的加权输出
            corr_tensor[i, j] = fused_feat * rel_vec  # [d]

    return corr_tensor  # [n, n, d]



# ####################################### Test ##################################
# h_cl = torch.randn(5, 768).to(device)  # 假设 5 个节点
# edge_index = torch.tensor([[0, 1, 3],
#                            [1, 2, 4]]).to(device)  # 三条边 (0->1), (1->2), (3->4)
# dep_rel_feats = torch.randn(3, 768).to(device)  # 每条边的依存关系特征
#
# # 定义投影层（只需初始化一次）
# proj_layer = torch.nn.Linear(2 * 768, 768).to(device)
#
# sim_matrix = build_dependency_guided_corr_tensor(h_cl, edge_index, dep_rel_feats)
# print("结构引导自相关矩阵形状:", sim_matrix.shape)

