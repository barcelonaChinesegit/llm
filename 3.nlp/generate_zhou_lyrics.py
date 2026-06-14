import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path


# 1. 数据集构建 (Dataset & DataLoader)
class LyricsDataset(Dataset):
    def __init__(self, corpus_chars, seq_len):
        """
        初始化歌词数据集
        corpus_chars: 原始文本字符串
        seq_len: 序列长度 (时间步数)，即：一曲歌歌词长度
        """
        # 1.清洗：将换行符替换为空格
        corpus_chars = corpus_chars.replace('\n', ' ')
        self.seq_len = seq_len
        # 2.构建词表
        self.chars = sorted(list(set(corpus_chars)))
        self.vocab_size = len(self.chars)
        self.char_to_idx = {ch: i for i, ch in enumerate(self.chars)}
        self.idx_to_char = {i: ch for i, ch in enumerate(self.chars)}
        # 3.将文本转化为索引序列
        self.corpus_indices = [self.char_to_idx[c] for c in corpus_chars]
        # self.corpus_indices的形状：(文件总字数,)

    def __len__(self):
        # 减去seq_len保证能取到目标值(target)
        return len(self.corpus_indices) - self.seq_len

    def __getitem__(self, idx):
        # 特征 X: 长为 seq_len 的序列
        # 标签 Y: 错位 1 个时间步的序列
        x = self.corpus_indices[idx: idx + self.seq_len]
        y = self.corpus_indices[idx + 1: idx + self.seq_len + 1]
        # x和y的形状：(self.seq_len,)
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)


# 2. 模型定义 (Model Architecture)
class CharRNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_size, num_layers=2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # Embedding层
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_dim)
        self.rnn = nn.RNN(
            input_size=embed_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.3
        )
        # 线性层解码输出
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden=None):
        # x形状: (batch_size, seq_len)
        emb = self.embedding(x)
        # emb形状:(batch_size, seq_len, embed_dim)
        out, hidden = self.rnn(emb, hidden)
        # out形状:(batch_size, seq_len, hidden_size)
        # 调整形状以适应全连接层和交叉熵损失
        out = out.reshape(-1, self.hidden_size)
        # out形状: (batch_size * seq_len, hidden_size)
        out = self.fc(out)
        # out形状: (batch_size * seq_len, vocab_size)
        return out,hidden

    def init_hidden(self, batch_size, device):
        # 初始化h_0
        return torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)

# 3. 文本生成逻辑 (Text Generation)
def generate_text(model, dataset, prefix, num_chars, device):
    """
    基于前缀字符串生成后续文本
    """
    model.eval()
    state = model.init_hidden(batch_size=1, device=device)
    outputs = [dataset.char_to_idx[c] for c in prefix] #输出内容在输入基础上拼接

    # 模型读入前缀将记忆保存在state中
    with torch.no_grad():
        for i in range(len(prefix) - 1):
            x = torch.tensor([outputs[i]]).reshape(1,-1).to(device)
            _, state = model(x, state)

        # 每次只读最后一个字，每次值输出一个字，输出的字加到末尾下轮继续读。
        for _ in range(num_chars):
            x = torch.tensor([outputs[-1]]).reshape(1,-1).to(device)
            pred, state = model(x, state)
            # 获取概率最大的字符索引（也可以使用 multinomial 采样增加随机性）
            next_idx = pred.argmax(dim=1).item()
            outputs.append(next_idx)

    return ''.join([dataset.idx_to_char[i] for i in outputs])


# 4. 训练与主程序 (Training & Main Loop)
def main():
    # 超参数配置
    seq_len = 10
    embed_dim = 256
    batch_size = 128
    hidden_size = 512
    num_layers = 2
    epochs = 20
    lr = 1e-3
    save_path = Path.cwd() / "best_generate_model.pth"
    file_path = Path.cwd() / 'jaychou_lyrics.txt'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. 加载数据
    with open(file_path, encoding='utf-8') as f:
        corpus_chars = f.read()
    dataset = LyricsDataset(corpus_chars, seq_len)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    # 2. 实例化模型、损失函数和优化器
    model = CharRNN(dataset.vocab_size, embed_dim, hidden_size, num_layers).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # 尝试从硬盘加载预训练参数
    if save_path.exists():
        print(f"发现本地保存的模型权重：{save_path}")
        model.load_state_dict(torch.load(save_path,map_location=device))
    else:
        print("未找到本地权重文件，将从头开始训练全新模型。")

    # 3. 训练循环
    print(f"开始训练，设备: {device}，词表大小: {dataset.vocab_size}")
    best_loss = float('inf')
    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for batch_idx, (batch_x, batch_y) in enumerate(dataloader):
            # 0.参数准备
            # batch_x和batch_y的形状：(batch_size,seq_len)
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            # 初始化隐藏层 (每一个batch都要重复初始化，要不计算图会太长而断掉)
            hidden = model.init_hidden(batch_size, device)
            # 1.梯度清零
            optimizer.zero_grad()
            # 2.前向传播
            output,_ = model(batch_x, hidden)
            # 3.计算损失：y需要展平匹配预测输出的形状
            loss = criterion(output, batch_y.reshape(-1))
            # 4.反向传播与梯度裁剪（防止梯度爆炸）
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)
            # 5.梯度更新
            optimizer.step()

            total_loss += loss.item()
            if (batch_idx + 1) % 100 == 0 or batch_idx + 1 == len(dataloader):
                current_avg_loss = total_loss / (batch_idx + 1)
                print(f'Epoch [{epoch + 1}/{epochs}], '
                      f'Batch [{batch_idx + 1}/{len(dataloader)}], '
                      f'Loss: {current_avg_loss}, '
                      f'Perplexity: {np.exp(current_avg_loss):.4f}')

        avg_loss = total_loss / len(dataloader)
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(),save_path)
            print(f"发现新纪录(Avg Loss: {best_loss:.4f})，模型已保存！")

        if (epoch + 1) % 5 == 0 or epoch == 0:
            sample = generate_text(model, dataset, prefix="想要有直升机", num_chars=10, device=device)
            print(f'Sample: {sample}\n')

def use():
    # 超参数配置
    seq_len = 10
    embed_dim = 256
    hidden_size = 512
    num_layers = 2
    save_path = Path.cwd() / "best_generate_model.pth"
    file_path = Path.cwd() / 'jaychou_lyrics.txt'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    text = input("给歌词开头：")
    num_chars = int(input("写多少词："))

    # 加载数据
    with open(file_path, encoding='utf-8') as f:
        corpus_chars = f.read()
    dataset = LyricsDataset(corpus_chars, seq_len)

    # 实例化模型
    model = CharRNN(dataset.vocab_size, embed_dim, hidden_size, num_layers).to(device)

    # 尝试从硬盘加载参数
    if save_path.exists():
        print(f"发现本地保存的模型权重：{save_path}")
        model.load_state_dict(torch.load(save_path,map_location=device))
    else:
        print("未找到本地权重文件，模型无法使用。")
        exit()

    sample = generate_text(model, dataset, prefix=text, num_chars=num_chars, device=device)
    print(f'Sample: {sample}\n')

if __name__ == '__main__':
    # main()
    use()
