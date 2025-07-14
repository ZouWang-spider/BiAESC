import torch
import torch.nn as nn
import torch.nn.functional as F
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score

from BiAESC.DataProcess.Dataprocess import Dataset_Process
from BiAESC.BaseModel.BiAffine import BiAffine, BERT_Embedding
from BiAESC.BaseModel.Expert_network import ExpertNetwork
from BiAESC.Module.Aspect_to_Sentiment_Attention import A2S_LocalAwareness
from BiAESC.Module.Sentiment_to_Aspect_Enhanced import S2A_SemanticRefinement
from BiAESC.BaseModel.R_GAT import RGAT_Model
from BiAESC.Module.Graph_Contrastive_Loss import build_graph_pairs, graph_contrastive_loss
from BiAESC.Module.Correlation_Matrix import build_dependency_guided_corr_tensor

# 使用 device 变量来控制数据是否在 GPU 上
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#专家网络层
expert_network = ExpertNetwork(input_dim=768, expert_dim=768).to(device)

#aspect_to_sentiment Local Attention module
A2S_module = A2S_LocalAwareness(hidden_dim=768, num_heads=12, dropout=0.3).to(device)

#aspect_to_sentiment Semantic Enhancement module
S2A_module = S2A_SemanticRefinement(input_dim=768, hidden_dim=768, num_layers=2, num_heads=12, dropout=0.3).to(device)


# RGAT网络初始化
hidden_dim = 128
out_dim = 768
num_heads = 12
RGAT_model = RGAT_Model(in_dim=768, hidden_dim=hidden_dim, out_dim=out_dim, r_dim=768, num_heads=num_heads).to(device)


#分类器
ate_classifier = nn.Sequential(
    nn.Linear(768, 256),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(256, 3)
).to(device)

aesc_classifier = nn.Sequential(
    nn.Linear(768, 256),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(256, 7)
).to(device)


# 优化器：将所有模块参数传入
optimizer = torch.optim.Adam(
    list(expert_network.parameters())+
    list(A2S_module.parameters()) +
    list(S2A_module.parameters()) +
    list(RGAT_model.parameters()) +
    list(ate_classifier.parameters()) +
    list(aesc_classifier.parameters()),
    lr=1e-5
)


loss_fn = nn.CrossEntropyLoss()
#ATE 损失值计算
def ATE_loss(features, labels):
    # 标签映射
    label_map = {'O': 0, 'B': 1, 'I': 2}
    label_ids = torch.tensor([label_map[tag] for tag in labels], device=device)

    logits = ate_classifier(features.to(device))  # [n, 3]

    ############## 检查 logits 和 label_ids 的长度是否一致###############
    n_logits = logits.size(0)
    n_labels = label_ids.size(0)

    if n_logits > n_labels:
        # 如果 logits 长度大于 label_ids，填充 'O'
        pad_size = n_logits - n_labels
        pad_labels = torch.full((pad_size,), label_map['O'], dtype=torch.long, device=device)
        label_ids = torch.cat([label_ids, pad_labels], dim=0)
    elif n_logits < n_labels:
        # 如果 logits 长度小于 label_ids，裁剪 label_ids
        label_ids = label_ids[:n_logits]

    # 计算损失
    ate_loss = loss_fn(logits, label_ids)  # 标准多分类损失

    return ate_loss

def label_mapping_tensor(aesc_labels):
    label_map = {'B-POS': 0, 'I-POS': 1, 'B-NEG': 2, 'I-NEG': 3, 'B-NEU': 4, 'I-NEU': 5, 'O': 6}
    aesc_label_ids = torch.tensor([label_map[label] for label in aesc_labels], dtype=torch.long, device=device)
    return aesc_label_ids


#AESC loss 计算
def AESC_loss(sim_matrix, aesc_labels):
    #AESC标签预处理转为tensor
    labels = label_mapping_tensor(aesc_labels).to(device)
    n = sim_matrix.size(0)
    feature_dim = sim_matrix.size(2)

    logits_list = []
    for i in range(n):
        diag_feat = sim_matrix[i, i, :].unsqueeze(0).to(device)  # (1, 768)
        row_feat = sim_matrix[i, :, :].to(device)  # (n, 768)

        combined = torch.cat([diag_feat, row_feat], dim=0)  # (n+1, 768)
        pooled = combined.mean(dim=0)  # (768,)

        # 线性层计算：logits = pooled @ W.T + b
        logit = aesc_classifier(pooled)
        logits_list.append(logit)

    logits = torch.stack(logits_list, dim=0).to(device)  # (n, 7)

    #AESC 标签映射
    aesc_label_map = {'B-POS': 0, 'I-POS': 1, 'B-NEG': 2, 'I-NEG': 3, 'B-NEU': 4, 'I-NEU': 5, 'O': 6}

    ############## 检查 logits 和 labels 的长度是否一致###############
    n_logits = logits.size(0)
    n_labels = labels.size(0)

    if n_logits > n_labels:
        # 如果 logits 长度大于 label_ids，填充 'O'
        pad_size = n_logits - n_labels
        pad_labels = torch.full((pad_size,), aesc_label_map['O'], dtype=torch.long, device=device)
        labels = torch.cat([labels, pad_labels], dim=0)
    elif n_logits < n_labels:
        # 如果 logits 长度小于 label_ids，裁剪 label_ids
        labels = labels[:n_logits]

    aesc_loss = loss_fn(logits, labels)
    return aesc_loss, logits


# 只取三类标签对应的 index
label_id_map = {
    'POS': [0, 1],  # B-POS, I-POS
    'NEG': [2, 3],  # B-NEG, I-NEG
    'NEU': [4, 5],  # B-NEU, I-NEU
}


#获取数据集
data = Dataset_Process(r"D:\Project\BiAESC\Dataset\laptop14_train.txt")

num_epochs = 50
for epoch in range(num_epochs):
    total_loss = 0.0
    # ATE_total_loss =0.0
    aesc_correct_total = 0.0
    aesc_total = 0.0
    num_sentences = 0  # 在 epoch 开头初始化

    all_preds = []  # 用于存储所有batch的预测值
    all_labels = []  # 用于存储所有batch的真实标签

    features = []
    labels = []

    for sentence, aspect, polarity, start_idx, end_idx, ate_labels, aesc_labels in data:

        # Step 1: 获取图结构和表示
        text_graph, G, rels, pos_tags = BiAffine(sentence)
        pooling_feature, dep_feature = BERT_Embedding(sentence, pos_tags, rels)

        # Step 2: 专家网络，获得 h_a, h_s
        h_a, h_s = expert_network(pooling_feature.to(device))

        # Step 3: A2S attention 获取情感增强表示 h_sl
        h_sl = A2S_module(h_a, h_s, G)

        # Step 4: S2A 语义增强获取 h_ae
        h_ae = S2A_module(h_a, h_s, text_graph.to(device))  # [n, d]

        # ATE 损失计算
        ate_loss = ATE_loss(h_ae, ate_labels)
        # print("ate_loss", ate_loss)

        # Step 5. 将两种特征拼接并池化处理
        h_concat = torch.cat([h_sl, h_ae], dim=-1)  # [n, 1536]
        h_concat = h_concat.unsqueeze(1)
        fusion_feature = F.max_pool1d(h_concat, kernel_size=2, stride=2).squeeze(1)  # [n, 768]

        # Step 6: R-GAT 计算
        # 转换图结构为 tensor，确保 edge_index 是 LongTensor [2, E]
        edge_index = torch.tensor(text_graph, dtype=torch.long, device=device)  # [2, E]
        rel_feats = dep_feature.to(device)  # [E, r_dim]
        h_cl = RGAT_model(fusion_feature, edge_index, rel_feats)

        # Step 7: 构造图对比学习正负样本,图节点对比学习的损失计算
        pos_pairs, neg_pairs = build_graph_pairs(text_graph, rels, start_idx, end_idx)
        contrastive = graph_contrastive_loss(h_cl, pos_pairs, neg_pairs)
        contrastive_loss = contrastive[0]
        # print("contrastive_loss", contrastive_loss)

        # ASC的KL损失计算
        p = F.log_softmax(h_sl, dim=-1)  # log P
        q = F.softmax(h_cl, dim=-1)  # Q
        kl_loss_fn = nn.KLDivLoss(reduction='batchmean')
        asc_loss = kl_loss_fn(p, q)
        # print("asc_loss", asc_loss)

        #Step 8: 依存相关矩阵构造
        sim_matrix = build_dependency_guided_corr_tensor(h_cl, edge_index, dep_feature.to(device))
        # print(sim_matrix.shape)


        label_tensor = label_mapping_tensor(aesc_labels).cpu().numpy()
        n_sim = sim_matrix.size(0)
        n_label = label_tensor.shape[0]

        # 若标签长度小于sim_matrix长度，则补 'O' (即类别6)
        if n_label < n_sim:
            pad_len = n_sim - n_label
            label_tensor = np.concatenate([label_tensor, np.full((pad_len,), 6, dtype=np.int64)], axis=0)
        elif n_label > n_sim:
            label_tensor = label_tensor[:n_sim]  # 多余部分裁掉
        # print(label_tensor)
        for i in range(n_sim):
            label_id = label_tensor[i]
            # 判断该标签是否属于POS、NEG、NEU（排除O）
            for sent_type, ids in label_id_map.items():
                if label_id in ids:
                    vec = sim_matrix[i, i].detach().cpu().numpy()  # 取对角线上的 [d]
                    features.append(vec)
                    labels.append(sent_type)
                    break

        #AESC的表格填充斜对角线损失+每行作为7分类损失计算
        aesc_loss, logits = AESC_loss(sim_matrix, aesc_labels)
        # print(aesc_loss.item())
        # print(logits.shape)  torch.Size([30, 7])为每个单词的标签概率分布

        #权重设置方案1 0.2, 0.2, 0.3, 0.3
        loss_values = 0.2 * ate_loss + 0.2 * asc_loss + 0.3 * contrastive_loss + 0.3 * aesc_loss
        print(f"Epoch {epoch + 1}, ate_loss: {ate_loss:.4f}, asc_loss: {asc_loss:.4f}, contrastive_loss: {contrastive_loss:.4f}, aesc_loss: {aesc_loss:.4f},")
        num_sentences += 1

        # 对单独子网络的梯度进行裁剪
        # torch.nn.utils.clip_grad_norm_(expert_network.parameters(), max_norm=1.0)
        # torch.nn.utils.clip_grad_norm_(A2S_module.parameters(), max_norm=1.0)
        # torch.nn.utils.clip_grad_norm_(S2A_module.parameters(), max_norm=1.0)
        # torch.nn.utils.clip_grad_norm_(RGAT_model.parameters(), max_norm=1.0)
        # torch.nn.utils.clip_grad_norm_(ate_classifier.parameters(), max_norm=1.0)
        # torch.nn.utils.clip_grad_norm_(aesc_classifier.parameters(), max_norm=1.0)

        # 反向传播
        optimizer.zero_grad()
        loss_values.backward()
        optimizer.step()
        total_loss += loss_values.item()


        #AESC 标签映射 logits: (n, 7) aesc_labels: (n,) 每个词真实标签 0~6
        aesc_label_map = {'B-POS': 0, 'I-POS': 1, 'B-NEG': 2, 'I-NEG': 3, 'B-NEU': 4, 'I-NEU': 5, 'O': 6}

        #准确率计算 Accuracy
        with torch.no_grad():
            preds = torch.argmax(logits, dim=-1)  # (n,)
            aesc_label_ids = label_mapping_tensor(aesc_labels)

            ############## 检查 logits 和 aesc_label_ids 的长度是否一致###############
            n_logits = preds.size(0)
            n_labels = aesc_label_ids.size(0)
            if n_logits > n_labels:
                # 如果 logits 长度大于 label_ids，填充 'O'
                pad_size = n_logits - n_labels
                pad_labels = torch.full((pad_size,), aesc_label_map['O'], dtype=torch.long, device=aesc_label_ids.device)
                aesc_label_ids = torch.cat([aesc_label_ids, pad_labels], dim=0)
            elif n_logits < n_labels:
                # 如果 logits 长度小于 label_ids，裁剪 label_ids
                aesc_label_ids = aesc_label_ids[:n_logits]

            correct = (preds == aesc_label_ids).sum().item()
            total = aesc_label_ids.numel()
            aesc_correct_total += correct
            aesc_total += total

        # 在训练过程中，计算每个batch的 Precision, Recall 和 F1
        with torch.no_grad():
            preds = torch.argmax(logits, dim=-1)  # (n,)
            aesc_label_ids = label_mapping_tensor(aesc_labels)

            # 检查长度一致性
            n_logits = preds.size(0)
            n_labels = aesc_label_ids.size(0)
            if n_logits > n_labels:
                pad_size = n_logits - n_labels
                pad_labels = torch.full((pad_size,), aesc_label_map['O'], dtype=torch.long,
                                        device=aesc_label_ids.device)
                aesc_label_ids = torch.cat([aesc_label_ids, pad_labels], dim=0)
            elif n_logits < n_labels:
                aesc_label_ids = aesc_label_ids[:n_logits]

            # 将预测值和真实标签存入列表
            all_preds.extend(preds.cpu().numpy())  # 需要将预测值移至CPU上并转换为numpy数组
            all_labels.extend(aesc_label_ids.cpu().numpy())  # 同样处理标签

    # 计算P, R, F1
    precision = precision_score(all_labels, all_preds, average='weighted')  # 平均方式可以选择 'micro', 'macro', 'weighted', 'samples' 等
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')

    # 特征矩阵 (num_samples, d)
    X = np.array(features)
    y = np.array(labels)

    # 使用TSNE降维到2D
    tsne = TSNE(n_components=2, random_state=42)
    X_tsne = tsne.fit_transform(X)

    # KMeans 聚类（聚成3类）
    kmeans = KMeans(n_clusters=3, random_state=42)
    cluster_labels = kmeans.fit_predict(X)

    # 可视化
    df = pd.DataFrame()
    df["Dim1"] = X_tsne[:, 0]
    df["Dim2"] = X_tsne[:, 1]
    df["Sentiment"] = y
    df["Cluster"] = cluster_labels

    # 原始标签可视化
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    sns.scatterplot(data=df, x="Dim1", y="Dim2", hue="Sentiment",
                    palette={"POS": "green", "NEG": "red", "NEU": "blue"})
    plt.title("True Sentiment Labels")

    # 聚类结果可视化
    plt.subplot(1, 2, 2)
    sns.scatterplot(data=df, x="Dim1", y="Dim2", hue="Cluster", palette="deep")
    plt.title("KMeans Clustering")

    plt.tight_layout()
    plt.show()


    aesc_acc = aesc_correct_total / aesc_total if aesc_total > 0 else 0.0
    avg_loss = total_loss / num_sentences if num_sentences > 0 else 0.0
    print(f"Epoch {epoch + 1}, Total Loss: {avg_loss:.4f}, Accuracy: {aesc_acc:.4f}")
    # 打开文件，如果文件不存在则会创建
    with open("D:\Project\BiAESC\epoch_twitter.txt", "a") as f:  # "a" 表示追加模式
        f.write(f"Epoch {epoch + 1}, Total Loss: {avg_loss:.4f}, Accuracy: {aesc_acc:.4f}\n")
        f.write(f"Epoch {epoch + 1}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}\n")
