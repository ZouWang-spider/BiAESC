import nltk
import dgl
import torch
import networkx as nx
import torch.nn as nn
from supar import Parser
from transformers import BertTokenizer, BertModel


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#使用BiAffine对句子进行处理得到arcs、rels、probs
def BiAffine(sentence):
      tokens = nltk.word_tokenize(sentence)

      # 获取词性标注
      ann = nltk.pos_tag(tokens)
      pos_tags = [pair[1] for pair in ann]


      # #POS feature
      # nlp = StanfordCoreNLP(r'D:/StanfordCoreNLP/stanford-corenlp-4.5.4', lang='en')
      # ann = nlp.pos_tag(sentence)
      # pos_tags = [pair[1] for pair in ann]
      # print(pos_tags)


      parser = Parser.load('D:/BiAffine/ptb.biaffine.dep.lstm.char')  # 'biaffine-dep-roberta-en'解析结果更准确
      dataset = parser.predict([tokens], prob=True, verbose=True)

      #dependency feature
      rels = dataset.rels[0]
      # print(f"arcs:  {dataset.arcs[0]}\n"
      #       f"rels:  {dataset.rels[0]}\n"
      #       f"probs: {dataset.probs[0].gather(1, torch.tensor(dataset.arcs[0]).unsqueeze(1)).squeeze(-1)}")

      # 构建句子的图，由弧-->节点
      arcs = dataset.arcs[0]  # 边的信息
      edges = [i + 1 for i in range(len(arcs))]
      for i in range(len(arcs)):
            if arcs[i] == 0:
                  arcs[i] = edges[i]

      # 将节点的序号减一，以便适应DGL graph从0序号开始
      arcs = [arc - 1 for arc in arcs]
      edges = [edge - 1 for edge in edges]
      graph = (arcs, edges)
      graph_line = '({}, {})\n'.format(graph[0], graph[1])  # 将图信息转为字符串
      # print("graph:", graph)
      # print(graph_line)

      # Create a DGL graph
      text_graph = torch.tensor(graph).to(device)
      # g = dgl.graph((arcs, edges))
      # nx.draw(g.to_networkx(), with_labels=True)
      # plt.show()

      # 创建一个有权图
      G = nx.Graph()
      for i, j in zip(arcs, edges):
            G.add_edge(i, j, weight=1)

      return text_graph, G, rels, pos_tags




#word embedding and part-of-speech embedding
def BERT_Embedding(sentence, pos, dep):
      text = nltk.word_tokenize(sentence)

      # 加载BERT模型和分词器
      model_name = 'D:/bert-base-cased'  # 您可以选择其他预训练模型
      tokenizer = BertTokenizer.from_pretrained(model_name)
      model = BertModel.from_pretrained(model_name).to(device)

      # 获取单词节点特征
      marked_text1 = ["[CLS]"] + text + ["[SEP]"]
      marked_text2 = ["[CLS]"] + pos + ["[SEP]"]
      marked_text3 = ["[CLS]"] + dep + ["[SEP]"]

      # 将分词转化为词向量 word embedding
      input_ids1 = torch.tensor(tokenizer.encode(marked_text1, add_special_tokens=True)).unsqueeze(0).to(device)  # 添加批次维度
      outputs1 = model(input_ids1)
      # Part-of-speech
      input_ids2 = torch.tensor(tokenizer.encode(marked_text2, add_special_tokens=True)).unsqueeze(0).to(device)  # 添加批次维度
      outputs2 = model(input_ids2)

      # dep
      input_ids3 = torch.tensor(tokenizer.encode(marked_text3, add_special_tokens=True)).unsqueeze(0).to(device)  # 添加批次维度
      outputs3 = model(input_ids3)

      # 获取词向量
      word_embeddings = outputs1.last_hidden_state
      pos_embeddings = outputs2.last_hidden_state
      dep_embeddings = outputs3.last_hidden_state



      # 提取单词对应的词向量（去掉特殊标记的部分）
      word_embeddings = word_embeddings[:, 1:-1, :]  # 去掉[CLS]和[SEP]标记
      pos_embeddings = pos_embeddings[:, 1:-1, :]  # 去掉[CLS]和[SEP]标记
      dep_embeddings = dep_embeddings[:, 1:-1, :]  # 去掉[CLS]和[SEP]标记

      # 使用切片操作去除第一个和最后一个元素
      word_feature = word_embeddings[0][1:-1, :]  # 单词特征
      pos_feature = pos_embeddings[0][1:-1, :]  # 单词特征
      dep_feature = dep_embeddings[0][1:-1, :]  # 单词特征


      #concat on last dim → [n, 768*2]
      concat_feature = torch.cat([word_feature, pos_feature], dim=-1)  # → [10, 1536]

      # 使用 MaxPool1d 做降维：1536 → 768
      pool = nn.MaxPool1d(kernel_size=2, stride=2).to(device)

      pooling_feature = pool(concat_feature)  # → [n, 768]

      return pooling_feature, dep_feature




############################################### Test ###########################################
# sentence = "The food is delicious but prices are high"
# text_graph, G, rels, pos_tags = BiAffine(sentence)
# # print(G)
# # print(text_graph)  #tensor([[2, 2, 4, 4, 4, 4, 7, 9, 9, 4], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]])
# # print(rels)    #dependency relation ['det', 'nn', 'nsubj', 'cop', 'root', 'cc', 'det', 'nsubj', 'cop', 'conj']
# # print(pos_tags) #['DT', 'NN', 'NN', 'VBZ', 'JJ', 'CC', 'DT', 'NNS', 'VBP', 'JJ']
# # pooling_feature, dep_feature = BERT_Embedding(sentence, pos_tags, rels)
# # print(pooling_feature.shape)  # torch.Size([10, 768])
# # print(dep_feature.shape)    # torch.Size([10, 768])
