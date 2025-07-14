import torch
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 构建Graph Contrastive Learning的正负样本对
def build_graph_pairs(text_graph, rels, start_idx, end_idx):
    """
    构建基于依存边的图对比学习样本对（正负样本），支持多词方面词。
    Args:
        text_graph: Tensor[2, E]，依存图边 [src, dst]
        rels: List[str]，每条边的依存标签
        start_idx: int，方面词起始位置
        end_idx: int，方面词结束位置

    Returns:
        pos_pairs: List[(i, j)]，正样本对
        neg_pairs: List[(i, j)]，负样本对
    """
    pos_pairs = set()
    neg_pairs = set()
    aspect_nodes = set(range(start_idx, end_idx + 1))
    dependency_target = {'nsubj', 'amod'}

    # 确保 text_graph 在 GPU 上
    text_graph = text_graph.to(device)
    E = text_graph.size(1)
    candidate_pos_pairs = set()  # 存储方面词直接连接的边（备选）

    for i in range(E):
        src = text_graph[0, i].item()
        dst = text_graph[1, i].item()
        rel = rels[i]

        # Step 1: 构建样本对
        edge = (src, dst)

        # Step 2: 排除自环
        if src == dst:
            continue

        # 如果是方面词之间的连接
        if src in aspect_nodes and dst in aspect_nodes:
            pos_pairs.add(edge)
            continue

        # 记录方面词直接连接的边（不考虑依存类型）
        if src in aspect_nodes or dst in aspect_nodes:
            candidate_pos_pairs.add(edge)

        # Step 3: 如果该边与方面词直接连接，且依存类型是目标类型
        if (src in aspect_nodes or dst in aspect_nodes) and rel in dependency_target:
            pos_pairs.add(edge)
        else:
            neg_pairs.add(edge)


    # 如果没有目标类型边，则降级使用直接连接边作为正样本
    if len(pos_pairs) == 0 and len(candidate_pos_pairs) > 0:
        for edge in candidate_pos_pairs:
            pos_pairs.add(edge)
            if edge in neg_pairs:
                neg_pairs.remove(edge)


    return list(pos_pairs), list(neg_pairs)




#构建图对比学习损失 Graph Contrastive Learning
def graph_contrastive_loss(h, pos_pairs, neg_pairs, temperature=0.5):
    """
    h: [n, d] 输出特征（R-GAT输出）
    pos_pairs: list of (i, j) indices for positive pairs
    neg_pairs: list of (i, j) indices for negative pairs
    """
    def sim(a, b):
        return F.cosine_similarity(a, b, dim=-1)  # [batch]

    # 确保所有输入张量都在同一个设备（GPU）
    h = h.to(device)  # 确保节点表示 h 在 GPU 上
    pos_pairs = torch.tensor(pos_pairs, dtype=torch.long).to(device)  # 转换为 GPU 张量
    neg_pairs = torch.tensor(neg_pairs, dtype=torch.long).to(device)  # 转换为 GPU 张量

    loss = 0.0
    for (i, j) in pos_pairs:
        z_i, z_j = h[i], h[j]
        pos_sim = sim(z_i.unsqueeze(0), z_j.unsqueeze(0)) / temperature

        neg_sim_sum = 0.0
        for (k, l) in neg_pairs:
            z_k, z_l = h[k], h[l]
            neg_sim = torch.exp(sim(z_i.unsqueeze(0), z_l.unsqueeze(0)) / temperature)
            neg_sim_sum += neg_sim

        loss += -torch.log(torch.exp(pos_sim) / (torch.exp(pos_sim) + neg_sim_sum + 1e-9))

    return loss / len(pos_pairs)



########################################  Test ##################################
# # 构造模拟 R-GAT 输出节点表示 h ∈ [6, 4]
# torch.manual_seed(42)
# h = torch.randn(6, 4).to(device)
#
# # 正样本对（模拟相关节点）
# pos_pairs = [(0, 1), (2, 3)]
#
# # 负样本对（模拟无关节点）
# neg_pairs = [(0, 4), (0, 5), (2, 4), (2, 5)]
#
# # 计算对比损失
# loss = graph_contrastive_loss(h, pos_pairs, neg_pairs, temperature=0.5)
# print("Graph Contrastive Loss:", loss.item())



# from BiAESC.BaseModel.BiAffine import BiAffine
#
# sentence = "my 3-year-old was amazed yesterday to find that ' real ' 10 pin bowling is nothing like it is on the wii ..."
# text_graph, G, rels, pos_tags = BiAffine(sentence)
# print(text_graph)  #tensor([[2, 2, 4, 4, 4],
#                           # [0, 1, 2, 3, 4]])
# print(rels)   #['det', 'nn', 'nsubj', 'cop', 'root']
#
# start_idx = 21
# end_idx = 21
#
# pos_pairs, neg_pairs = build_graph_pairs(text_graph, rels, start_idx, end_idx)
# print("正样本对:", pos_pairs)
# print("负样本对:", neg_pairs)