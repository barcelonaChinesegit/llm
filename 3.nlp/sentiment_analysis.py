import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import re
from pathlib import Path
from sklearn.model_selection import train_test_split

# 模块 1：数据预处理与词表构建
class IMDBProcessor:
    def __init__(self, max_vocab_size=50000, max_seq_len=200):
        self.max_vocab_size = max_vocab_size
        self.max_seq_len = max_seq_len
        # 预留两个特殊标记：<PAD> 用于补齐长度，<UNK> 用于未知词汇
        self.char_to_idx = {"<PAD>": 0, "<UNK>": 1}
        self.idx_to_char = ["<PAD>", "<UNK>"]
        self.vocab_size = 2

    def tokenizer(self, text):
        """清洗并分词"""
        text = str(text)
        text = re.sub(r"<br />", " ", text) #删掉文本中的"<br/>"
        text = re.sub(r"[^A-Za-z0-9]", " ", text) #仅保留文本中有效内容（有点草率，标点都没了)
        # 注：这里加了if tok过滤，防止连续空格切出空字符串"
        return [tok.lower() for tok in text.split() if tok]

    def build_vocab(self, texts):
        """构建词表 (基于词频截断)"""
        from collections import Counter
        all_words = [word for text in texts for word in self.tokenizer(text)]
        # 把所有影评压成一个一维的超长扁平列表 (list[str])。
        # all_words:['i', 'love', 'it', 'the', 'movie', 'is', 'bad', 'i', 'love', ...]
        word_counts = Counter(all_words)
        # 只保留出现频率最高的前max_vocab_size个词
        common_words = [word for word, count in word_counts.most_common(self.max_vocab_size - 2)]
        for word in common_words:
            self.char_to_idx[word] = self.vocab_size
            self.idx_to_char.append(word)
            self.vocab_size += 1

    def text_to_indices(self, text):
        """将单个句子转换为等长(max_seq_len)的整数ID列表"""
        sentence = self.tokenizer(text)
        # sentence：['i', 'love', 'it', 'the', 'movie']
        # 查表，如果遇到没见过的词，使用 <UNK> 的 ID (1)
        indices = [self.char_to_idx.get(token, 1) for token in sentence]
        # 长度处理：截断或补齐
        if len(indices) >= self.max_seq_len:
            indices = indices[:self.max_seq_len]  # 截断
        else:
            indices = indices + [0] * (self.max_seq_len - len(indices))  # 使用 <PAD>(0) 补齐
        return indices

# 模块 2：构建 PyTorch Dataset
class IMDBDataset(Dataset):
    def __init__(self, data_list, labels):
        # 转换为 PyTorch 张量
        self.data = torch.tensor(data_list, dtype=torch.long)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# 模块 3：定义现代 RNN 分类模型
class RNNClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers=1, output_dim=2):
        super(RNNClassifier, self).__init__()
        # 1. 词嵌入层：padding_idx=0 保证补齐符不参与梯度更新
        self.embedding = nn.Embedding(num_embeddings=vocab_size,
                                      embedding_dim=embed_dim,
                                      padding_idx=0)
        # 2. RNN 核心层：batch_first=True 让输入维度变为 (batch, seq_len, features)
        self.rnn = nn.RNN(input_size=embed_dim,
                          hidden_size=hidden_dim,
                          num_layers=num_layers,
                          batch_first=True,
                          dropout=0.35,
                          bidirectional=True)
        # 3. 线性分类输出层
        # 因为是双向RNN，正向和反向的特征拼在一起，所以输入维度是 hidden_dim * 2
        self.fc1 = nn.Linear(hidden_dim*2, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x 形状: (batch_size, seq_len)
        embedded = self.embedding(x) # embedded 形状: (batch_size, seq_len, embed_dim)
        output, h_n = self.rnn(embedded) # h_n 形状: (num_layers, batch_size, hidden_dim)
        # 提取最后一个时间步的隐藏状态
        last_hidden = torch.cat((h_n[-2,:,:],h_n[-1,:,:]),dim=1) # 形状变为 (batch_size, hidden_dim*2)
        # 经过全连接层+ 激活函数 + Dropout
        sigma1 = self.fc1(last_hidden)
        sigma1 = self.relu(sigma1)
        sigma1 = self.dropout(sigma1)
        out = self.fc2(sigma1)
        return out

# 模块 4：训练流程架构
def main():
    # 0.定义超参
    max_vocab_size = 20000 #词表大小上限
    max_seq_len = 150 #每句token上限
    batch_size = 125
    epochs = 30
    save_path = Path.cwd()/"best_lstm_model.pth"

    # 1. 读取数据
    path = Path.cwd() / "sentiment_analysis.csv"
    df = pd.read_csv(path, encoding='utf-8', encoding_errors='ignore')
    train_df,test_df = train_test_split(df,test_size=0.2,train_size=0.8,random_state=420)
    train_texts = train_df['movie_review'].tolist()
    train_labels = train_df['sentiment'].tolist()
    test_texts = test_df['movie_review'].tolist()
    test_labels = test_df['sentiment'].tolist()

    # 2. 数据清洗与构建词表
    processor = IMDBProcessor(max_vocab_size=max_vocab_size, max_seq_len=max_seq_len)
    processor.build_vocab(train_texts)
    print(f"词表大小: {processor.vocab_size}")

    # 3. 文本转张量并构建 DataLoader
    # 处理训练集
    train_encoded = [processor.text_to_indices(text) for text in train_texts]
    train_dataset = IMDBDataset(train_encoded, train_labels)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    # 处理测试集
    test_encoded = [processor.text_to_indices(text) for text in test_texts]
    test_dataset = IMDBDataset(test_encoded, test_labels)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 4. 初始化模型、优化器与损失函数
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"device: {device}")
    model = RNNClassifier(vocab_size=processor.vocab_size,
                          embed_dim=100,
                          hidden_dim=128,
                          num_layers=3,
                          output_dim=2).to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # 尝试从硬盘加载预训练参数
    if save_path.exists():
        print(f"发现本地保存的模型权重：{save_path}")
        model.load_state_dict(torch.load(save_path, map_location=device))
        print("模型权重加载成功！将在已有基础上继续...")
    else:
        print("未找到本地权重文件，将从头开始训练全新模型。")

    # 5. 开始训练 (Training Loop)
    best_acc = 0.0
    for epoch in range(epochs):
        model.train() # 将模型设置为训练模式
        total_loss = 0
        correct = 0
        for batch_texts, batch_labels in train_loader:
            batch_texts, batch_labels = batch_texts.to(device), batch_labels.to(device)
            # 步骤 1：清空历史梯度
            optimizer.zero_grad()
            # 步骤 2：前向传播 (查表 -> RNN -> 全连接预测)
            predictions = model(batch_texts)
            # 步骤 3：计算 Loss
            loss = criterion(predictions, batch_labels)
            # 步骤 4：反向传播计算梯度
            loss.backward()
            # 步骤 5：优化器更新权重(包括Embedding词典也会被更新！)
            optimizer.step()
            total_loss += loss.item()
            predicted_classes = torch.argmax(predictions,1)
            correct += (predicted_classes == batch_labels).sum().item()

        avg_train_loss = total_loss / len(train_loader)
        train_accuracy = correct / len(train_dataset) * 100
        print(f"Epoch: {epoch+1:02d} | 损失(Train Loss): {avg_train_loss:.4f} | 准确率(Train Acc): {train_accuracy:.2f}%")

        model.eval() # 切换为评估模式
        test_loss = 0
        test_correct = 0
        with torch.no_grad(): # 禁止计算梯度，节省显存并加速
            for batch_texts, batch_labels in test_loader:
                batch_texts, batch_labels = batch_texts.to(device), batch_labels.to(device)
                predictions = model(batch_texts)
                loss = criterion(predictions, batch_labels)
                test_loss += loss.item()
                predicted_classes = torch.argmax(predictions, 1)
                test_correct += (predicted_classes == batch_labels).sum().item()

        avg_test_loss = test_loss / len(test_loader)
        test_accuracy = test_correct / len(test_dataset) * 100
        print(f"Epoch: {epoch+1:02d} | 损失(Test Loss): {avg_test_loss:.4f} | 准确率(Test Acc): {test_accuracy:.2f}%")

        if test_accuracy > best_acc:
            best_acc = test_accuracy
            # 把当前表现最好的模型权重保存到硬盘上
            torch.save(model.state_dict(), save_path)
            print("发现新纪录，模型已保存！")

if __name__ == "__main__":
    main()