
#数据集预处理
# def Dataset_Process(file_path):
#     data = []
#     polarity_map = {1: "POS", 0: "NEU", -1: "NEG"}  # 映射成字符串标签
#
#     with open(file_path, 'r', encoding='utf-8') as f:
#         lines = [line.strip() for line in f if line.strip()]
#         for i in range(0, len(lines), 3):
#             sentence_with_T = lines[i]
#             aspect_term = lines[i + 1]
#             polarity = int(lines[i + 2])
#
#             # 替换 $T$ 为 aspect term
#             sentence = sentence_with_T.replace('$T$', aspect_term)
#
#             # ===> 获取 aspect 在 sentence 中的词级位置索引
#             # 分词（可换成更好的 tokenizer）
#             words = sentence.split()
#             aspect_words = aspect_term.split()
#
#             # 尝试在 word list 中定位 aspect term（连续匹配）
#             start_idx = -1
#             for j in range(len(words) - len(aspect_words) + 1):
#                 if words[j:j + len(aspect_words)] == aspect_words:
#                     start_idx = j
#                     break
#
#             if start_idx == -1:
#                 raise ValueError(f"Aspect term '{aspect_term}' not found in sentence '{sentence}' after replacement.")
#
#             end_idx = start_idx + len(aspect_words) - 1
#
#             # 构造ATE任务标签
#             ate_labels = []
#             for idx in range(len(words)):
#                 if idx == start_idx:
#                     ate_labels.append('B')
#                 elif start_idx < idx <= end_idx:
#                     ate_labels.append('I')
#                 else:
#                     ate_labels.append('O')
#
#
#             #构造表格填充AESC任务标签
#             sent_tag = polarity_map.get(polarity, "NEU")  # 默认 NEU 避免错误
#             aesc_labels = []
#             for idx in range(len(words)):
#                 if idx == start_idx:
#                     aesc_labels.append(f'B-{sent_tag}')
#                 elif start_idx < idx <= end_idx:
#                     aesc_labels.append(f'I-{sent_tag}')
#                 else:
#                     aesc_labels.append('O')
#
#
#             # 保存结构：句子、方面词、极性、起始位置、结束位置
#             data.append((sentence, aspect_term, polarity, start_idx, end_idx, ate_labels, aesc_labels))
#
#     return data

import re
from nltk.tokenize import word_tokenize

def clean_word(w):
    return re.sub(r'^[^\w]+|[^\w]+$', '', w)

def Dataset_Process(file_path):
    data = []
    polarity_map = {"POS": 1, "NEU": 0, "NEG": -1}

    with open(file_path, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]
        for line in lines:
            try:
                sentence_part, label_part = line.split("####")
                raw_tokens = label_part.split()

                words = []
                tags = []
                for tok in raw_tokens:
                    if '=' not in tok:
                        continue
                    word, tag = tok.split("=")
                    word = clean_word(word)
                    if word:
                        words.append(word)
                        tags.append(tag)

                sentence = sentence_part.strip()

                # 找出所有的方面词及其位置
                aspects = []
                i = 0
                while i < len(tags):
                    tag = tags[i]
                    if tag.startswith("T-"):
                        sent_tag = tag[2:]  # POS / NEG / NEU
                        start = i
                        aspect_tokens = [words[i]]
                        i += 1
                        while i < len(tags) and tags[i] == tag:
                            aspect_tokens.append(words[i])
                            i += 1
                        end = start + len(aspect_tokens) - 1
                        aspect_text = ' '.join(aspect_tokens)

                        # 构造 ate 和 aesc 标签
                        ate_labels = ['O'] * len(words)
                        aesc_labels = ['O'] * len(words)
                        ate_labels[start] = 'B'
                        aesc_labels[start] = f'B-{sent_tag}'
                        for j in range(start + 1, end + 1):
                            ate_labels[j] = 'I'
                            aesc_labels[j] = f'I-{sent_tag}'

                        polarity = polarity_map[sent_tag]
                        data.append((sentence, aspect_text, polarity, start, end, ate_labels, aesc_labels))
                    else:
                        i += 1

            except Exception as e:
                print(f"[PARSE ERROR] line skipped: {line}\nReason: {e}")

    return data






############################  test  #################################
# file_path = r"D:\Project\BiAESC\Dataset\laptop14_train.txt"
# data = Dataset_Process(file_path)
# for sentence, aspect_term, polarity, start_idx, end_idx, ate_labels, aesc_labels in data:
#     print(sentence)
#     print(aspect_term)
#     print(polarity)
#     print(start_idx)
#     print(end_idx)
#     print(ate_labels)
#     print(aesc_labels)
