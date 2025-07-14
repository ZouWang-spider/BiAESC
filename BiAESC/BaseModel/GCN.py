import torch
import torch.nn as nn
import torch_geometric.nn as geo_nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class GCNModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(GCNModel, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.gcn_layers = nn.ModuleList()
        self.gcn_layers.append(geo_nn.GCNConv(input_size, hidden_size))  # 第一层使用 GCNConv

        for _ in range(1, num_layers):
            self.gcn_layers.append(geo_nn.GCNConv(hidden_size, hidden_size))  # 后续层继续使用 GCNConv


    def forward(self, node_features, edge_index):
        # 确保输入张量在 GPU 上
        node_features = node_features.to(device)
        edge_index = edge_index.to(device)

        for layer in self.gcn_layers:
            node_features = layer(node_features, edge_index)

        return node_features



####################################### Test ##################################
# #GCN model parameter
# input_size2 = 768
# hidden_size2 = 768
# num_layers2 = 2
# gcn = GCNModel(input_size2, hidden_size2, num_layers2).to(device)
#
# from BiAESC.BaseModel.BiAffine import BiAffine, BERT_Embedding
# sentence ='The chocolate cake is delicious but the donuts are terrible'
# text_graph, G, rels, pos_tags = BiAffine(sentence)
# pooling_feature, dep_feature = BERT_Embedding(sentence, pos_tags, rels)
#
# # 确保池化特征在 GPU 上
# pooling_feature = pooling_feature.to(device)
#
# #GCN模型 torch.Size([n, 5])
# output = gcn(pooling_feature, text_graph)
# print(output.shape)