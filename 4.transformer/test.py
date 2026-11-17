import nltk
from nltk.corpus import sentence_polarity
from collections import defaultdict, Counter
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset,DataLoader
from torch.nn.utils.rnn import pad_sequence
import os
import os.path
from pathlib import Path

# 一.词汇表类
class Vocab:
    """
    词汇表类，用于根据token构建token-id映射；可以同时接纳Token和text两种类型的数据
    1.对原始文字数据，调用build方法，进行分词、并完成词频筛选
    2.对Token数据，使用init中的流程，完成添加未知词、词汇表构建并根据词汇表进行编码
    3.建好词汇表后，再调用单独的方法来进行编码
    """
    #__init__的作用：添加未知词、词汇表构建
    def __init__(self, tokens=None):
        # tokens：一个包含所有token且去重好的的list;(这个列表里只能有token本身，不能再包含其他内容或者其他层次)
        self.idx_to_token = list() # idx_to_token是列表，包含了数据集中所有的单词
        self.token_to_idx = dict() # token_to_idx是词汇表，是不重复的单词 + 索引构成的结果
        if tokens is not None:
            # 如果tokens中不包含"<unk>"这个词（未知词），则添加"<unk>"
            if "<unk>" not in tokens:
                tokens = tokens + ["<unk>"]
            # 遍历tokens，将每个token添加到idx_to_token列表，并在token_to_idx字典中映射其索引
            # 基于添加了未知词的Tokens，直接创造出列表+词汇表
            for token in tokens:
                self.idx_to_token.append(token)
                self.token_to_idx[token] = len(self.idx_to_token) - 1
            # 设置未知词的索引，将未知的词设置为一个单独的属性self.unk
            self.unk = self.token_to_idx['<unk>']

    # build的作用：分词、词频筛选
    #调用魔法命令classmethod，这个命令允许我们在不进行实例化的情况下使用类中的方法
    @classmethod
    def build(cls, text, min_freq=1, reserved_tokens=None):
        # 参数：
        # cls是Vocab这个类本身，这魔法命令classmethod的要求，有了cls就可以在不进行实例化的情况下直接调用build功能
        # text是需要构建词汇表和词典的文本，在这个文本上我们可以直接开始进行词频筛选
        # 注:这个文本的范围很广泛，只要不是单一token list，都可以被认为是文本
        # text可以接受的格式是:
        # text1 = ["我爱中国","我的大好河山"]
        # text2 = "我爱中国"
        # text3 = ["我","爱","中","国","我","的","大","好","河","山"]
        # min_freq是我们用于筛选的最小频率，低于该频率阈值的词会被删除
        # reserved_token是我们可以选择性输入的"通用词汇表"，假设text本身太短词太少的话
        # reserved_token可以帮助我们构建更大的词典、从而构建更大的词向量空间

        # 创建一个defaultdict字典，用于统计每个单词的出现频率
        token_freqs = defaultdict(int)
        # 非常粗糙的分词，中文是按字分，英文按字母分了(可以升级用Jieba或NLTK分词)
        for sentence in text:
            for token in sentence:
                token_freqs[token] += 1
        # 创建一个空列表uniq_tokens，用于存储"<unk>"和输入用来保底的reserved_tokens
        uniq_tokens = ["<pad>"] + ["<unk>"] + (reserved_tokens if reserved_tokens else []) # <pad>为0，<unk>为1，从2开始是正常的token id
        # 加上过滤后的Tokens列表
        uniq_tokens += [token for token, freq in token_freqs.items() if freq >= min_freq and token != "<unk>"]
        # 去重，同时保留原来的顺序
        uniq_tokens = list(dict.fromkeys(uniq_tokens))
        return cls(uniq_tokens) # 创建一个类的实例对象，然后将这个对象返回。

    def __len__(self):
        return len(self.idx_to_token)

    def __getitem__(self, token):
        return self.token_to_idx.get(token, self.unk)

    def convert_tokens_to_ids(self, tokens):
        return [self[token] for token in tokens]

    def convert_ids_to_tokens(self, indices):
        return [self.idx_to_token[index] for index in indices]

    def save_vocab(vocab, path):
        # 将词汇表保存到指定路径的文件中
        with open(path, 'w') as writer:
            writer.write("\n".join(vocab.idx_to_token))

    def read_vocab(path):
        # 从指定路径的文件中读取词汇表
        with open(path, 'r') as f:
            tokens = f.read().split('\n')
        # 返回一个新的Vocab实例，初始化它的词汇表
        return Vocab(tokens)

# 二.加载数据集，利用Vocab类构建词表并编码，最后划分训练集和测试集
def load_sentence_polarity():
    from nltk.corpus import sentence_polarity
    # 1.用全部句子构建词汇表
    # sentence_polarity.sents()是list[list[str]]，不是单一token list，所以只能交给build，不能直接交给__init__
    # Vocab.build是类方法返回Vocab的实例化对象
    vocab = Vocab.build(sentence_polarity.sents())
    # 2.构建训练集：正面句子标记为0，负面句子标记为1，各取前4000句
    # 每个样本是一个元组：(句子编码后的id列表, 标签)
    train_data = ([(vocab.convert_tokens_to_ids(sentence),0) for sentence in sentence_polarity.sents(categories="pos")[:4000]]+
                  [(vocab.convert_tokens_to_ids(sentence),1) for sentence in sentence_polarity.sents(categories="neg")[:4000]])
    # 3.构建测试集：使用剩余的句子(每类5331句，剩1331句)
    test_data = ([(vocab.convert_tokens_to_ids(sentence),0) for sentence in sentence_polarity.sents(categories="pos")[4000:]]+
                  [(vocab.convert_tokens_to_ids(sentence),1) for sentence in sentence_polarity.sents(categories="neg")[4000:]])
    return train_data,test_data,vocab

# 三.自定义Dataset类(重写__init__和__len__及__getitem__)
class TransformerDataset(Dataset):
    def __init__(self,data):
        super().__init__()
        # 初始化数据集，将传入的数据保存在实例变量data中
        self.data = data

    def __len__(self):
        # 返回数据集的大小，即：有多少条(句子+label)元组
        return len(self.data)

    def __getitem__(self, index):
        # 根据索引index获取数据集中的第index个样本，即(句子, label)
        return self.data[index]

# 四.collate_fn：DataLoader拼batch时调用，负责batch内的padding
def collate_fn(examples):
    # examples是一个list，里面装着batch_size个(句子, label)样本
    # 1.将每个样本的句子部分(id列表)转换为张量，此时句子长度不一致
    # 注意：collate_fn这里没有调用nn.Embedding是正确的。因为它只负责整理batch，把每一句都变成一样长，而不是进行模型计算
    # inputs是一维Tensor列表，列表每行是保存原先一句话的TokenID的一维Tensor，即:[35, 128, 56]变成tensor([35, 128, 56])
    inputs = [torch.tensor(ex[0]) for ex in examples]
    # 2.将每个样本的标签部分转换为长整型张量，形状(batch_size,)
    targets = torch.tensor([ex[1] for ex in examples], dtype=torch.long)
    # 3.对batch内的样本进行padding，按本batch最长的句子在末尾补<pad>的id:0
    # batch_first=True → 输出形状(batch_size, seq_len)
    inputs = pad_sequence(inputs,batch_first=True,padding_value=0,padding_side="right")
    return inputs, targets

# 五.length_to_mask：生成填充掩码(浮点版)
def length_to_mask(seq, pad_token=0):
    # seq: (batch_size, seq_len)
    # 注意:padding_mask不是直接加到词向量上的，而是加到Attention计算出来的注意力分数上的，而且是在Softmax之前相加
    # 是PAD的位置 → -1e9(加到注意力分数上，softmax后≈0)；正常词 → 0(加了等于没加)
    padding_mask = (seq == pad_token).float()*-1e9 #(batch_size, seq_len)
    # 或者直接padding_mask = (seq == pad_token)，transformer的src_key_padding_mask支持bool矩阵
    return padding_mask

# 六.PositionalEncoding：位置编码类
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=512, batch_first=True):
        """
        创造位置编码层
        参数:
        d_model (int): 嵌入向量的维度。
        dropout (float): Dropout 概率。
        max_len (int): 序列的最大长度（注意与矩阵中的seq_len区分开来）
        """
        super().__init__()
        # 1.位置编码作为精确到每一个维度的信息，容易导致过拟合，因此在实际实践位置编码的时候我们会加上扛过拟合的dropout
        self.dropout = nn.Dropout(p=dropout)
        # 2.创建一个空白的编码矩阵,形状为(max_len, d_model);通过给空白的位置编码矩阵填上具体值,来构成真正的位置编码矩阵
        pe = torch.zeros(max_len, d_model)
        # 3.接下来开始生成正弦余弦编码公式中的各个元素
        # 3.1位置索引(pos)，形状为(max_len, 1)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        # 3.2计算公式中的分母
        # 由于sin和cos公式的分母是一致的，所以只需计算出一组分母，就可给sin和cos函数通用，这组分母的数量是 d_model/2
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        # 3.3计算出正弦余弦值，并赋值给原本设置好的位置编码矩阵
        # 计算正弦位置编码: PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
        # 所有行上、从0开始、每2列赋予一个sin值
        pe[:, 0::2] = torch.sin(position * div_term)
        # 计算余弦位置编码: PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
        # 所有行上，从1开始，每2列赋予一个cos值
        pe[:, 1::2] = torch.cos(position * div_term)
        if batch_first:
            # 4.使用unsqueeze添加batch维度，将原本二维的结构转变为三维，形状变为 (1, max_len, d_model)
            pe = pe.unsqueeze(0)
        else:
            # 4.使用unsqueeze添加batch维度，将原本二维的结构转变为三维，形状变为 (1, max_len, d_model)
            # 5.使用transpose交换了维度，因此pe的最终结构是(max_len, 1, d_model)，这是batch_first=False下合理的结构
            pe = pe.unsqueeze(0).transpose(0, 1)  # (max_len, 1, d_model)
        # 位置编码与batch无关，对所有batch都一样;
        # 所以forward相加原始张量时会通过广播的方式复制成(batch_size, seq_len, input_dimension)或(seq_len, batch_size, input_dimension)
        # 6.将位置编码矩阵注册为模型的缓冲区，不会作为模型参数更新
        self.register_buffer('pe', pe) # 这一步相当于定义了self.pe，但规定pe不参与模型参数更新
        self.batch_first = batch_first

    def forward(self, x):
        """
        前向传播函数，给输入添加位置编码。
        参数:
        x (Tensor): embedding后的张量，形状为 (batch_size, seq_len, d_model)
        返回值:
        Tensor: 添加了位置编码的输入张量，形状不变
        """
        # 添加位置编码，注意这里对位置编码进行了切片以匹配输入的序列长度
        # 也就是说，会根据x中的seq_len对原本设置的max_len进行裁剪
        if self.batch_first:
            # batch_first：需要截断的是中间的维度(seq_len维度)
            x = x + self.pe[:, :x.size(1), :]
        else:
            # 非batch_first：需要截断的是第一个维度
            x = x + self.pe[:x.size(0), :]
        # 应用 Dropout 并返回
        return self.dropout(x)

# 七.EncoderOnlyTransformer：情感分类模型
# 注意：PositionalEncoding和nn.TransformerEncoderLayer均已支持batch_first=True/False；
# 但collate_fn及EncoderOnlyTransformer.forward仍按batch_first=True编写，
# 因此完整分类模型目前只支持batch_first=True，不能直接设置为False。
class EncoderOnlyTransformer(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_class,
                 dim_feedforward=256, num_head=2, num_layers=2,
                 dropout=0.1, max_len=128, activation: str = "relu",
                 batch_first=True):
        """
        参数:
        vocab_size (int): 词汇表大小，即词汇表中Token的总数量（包含特殊Token）。
        embedding_dim (int): 词嵌入向量的维度，每个Token映射后的特征维度。
        num_class (int): 分类类别数量，二分类任务通常设置为2。
        dim_feedforward (int): Transformer编码层中前馈神经网络的隐藏层维度，默认为256。
        num_head (int): 多头自注意力机制中的注意力头数量，必须能整除embedding_dim，默认为2。
        num_layers (int): Transformer Encoder中堆叠的编码层数量，默认为2。
        dropout (float): Dropout概率，用于缓解过拟合，默认为0.1。
        max_len (int): 位置编码支持的最大序列长度，默认为128。
        activation (str): 前馈神经网络使用的激活函数，可选"relu"或"gelu"，默认为"relu"。
        batch_first (bool): 是否将Batch维度放在第一维，默认为True。
        """
        super().__init__()
        # 1.词嵌入层：(batch, seq_len) → (batch, seq_len, embedding_dim)
        self.embedding_dim = embedding_dim
        self.embeddings = nn.Embedding(vocab_size, embedding_dim,padding_idx=0)
        # 2.位置编码层
        self.position_embedding = PositionalEncoding(embedding_dim, dropout, max_len, batch_first=batch_first)
        # 3.encoder编码层与encoder编码层的打包
        encoder_layer = nn.TransformerEncoderLayer(embedding_dim, num_head, dim_feedforward, dropout, activation, batch_first=batch_first)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        # 4.输出层：情感分类是二分类，num_class=2
        # 这里输出2个类别的分数，配合nn.CrossEntropyLoss使用(其实也可以将num_class改成1,输出结果套上Sigmoid,再配BCE)
        self.output = nn.Linear(embedding_dim, num_class)

    def forward(self, inputs):
        # inputs: (batch_size, seq_len)，是padding后的id序列
        # 1.生成填充掩码(在模型内部直接生成，调用时只需要传inputs)
        padding_mask = length_to_mask(inputs)  # (batch_size, seq_len)
        # 2.获取词嵌入表示
        src = self.embeddings(inputs)  # (batch_size, seq_len, embedding_dim)
        # 3.添加位置编码
        src = self.position_embedding(src)  # 形状不变
        # 4.通过Transformer编码器进行处理
        src = self.transformer(src, src_key_padding_mask=padding_mask)  # 形状不变
        # 5.把整句压成一个向量：这里取序列最后一个位置(也可以用平均池化 src.mean(dim=1))
        src = src[:, -1, :]  # (batch_size, embedding_dim)
        # 6.通过输出层得到每个类别的分数
        output = self.output(src)  # (batch_size, num_class)
        return output

def main():
    # 超参数、数据、模型、损失、优化器
    embedding_dim = 128  # 词嵌入的维度为128
    num_class = 2  # 类别数量为2
    batch_size = 32  # 每个批次的大小为32
    num_epoch = 15  # 训练的轮数为15
    lr = 0.001 #学习率为0.001

    # 1.加载数据
    # 首次运行会下载数据集(下载到默认路径后,再次运行,自动加载)
    nltk.download("sentence_polarity")
    train_data, test_data, vocab = load_sentence_polarity()
    train_dataset = TransformerDataset(train_data)  # 创建训练数据集
    test_dataset = TransformerDataset(test_data)  # 创建测试数据集
    # 训练集：32个一组，打乱顺序，用collate_fn做padding
    train_data_loader = DataLoader(train_dataset,batch_size=batch_size,shuffle=True,collate_fn=collate_fn)
    # 测试集：1个一组(单句不需要padding)，不打乱
    test_data_loader = DataLoader(test_dataset,batch_size=1,shuffle=False,collate_fn=collate_fn)
    # for inputs, targets in train_data_loader:
    #     print(inputs)  # 第一个batch中的实际数据
    #     print(targets)  # 每个句子对应的标签(0积极，1消极)
    #     print(inputs.shape)  # torch.Size([32, xx])  → (batch_size, 本batch最长句长)
    #     print(targets.shape)  # torch.Size([32])
    #     print(length_to_mask(inputs).shape)  # torch.Size([32, xx])，PAD处为-1e9，其余为0
    #     break

    # 2.加载模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = EncoderOnlyTransformer(len(vocab), embedding_dim, num_class)
    model.to(device)

    # 3.损失函数与优化器
    loss_fn = nn.CrossEntropyLoss()  # 多类别交叉熵损失：输入为原始Logits；标签的形状必须是(BatchSize,)，数据类型必须是整型(torch.long或torch.int64)
    optimizer = optim.Adam(model.parameters(), lr=lr)  # Adam优化器，学习率为0.001

    # 4.训练过程
    model.train()  # 设置模型为训练模式(启用dropout)
    for epoch in range(num_epoch):
        total_loss_for_epoch = 0
        for batch_x, batch_y in train_data_loader:
            # (1).取出输入和标签：inputs (32, seq_len) 和 targets (32,) 并移动到指定设备
            inputs = batch_x.to(device)
            targets = batch_y.to(device)
            # (2).前向传播
            logits = model(inputs)
            # (3).计算损失
            loss = loss_fn(logits, targets)
            # (4).清零梯度、反向传播、更新参数
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # 5.累加各Batch的平均损失(.item()将单元素张量转换成Python数字)
            total_loss_for_epoch += loss.item()
        # 输出每个轮次的总损失
        print(f"Loss: {total_loss_for_epoch:.2f}")

    # 5.测试过程
    acc = 0
    total = 0
    model.eval()  # 设置为评估模式(关闭dropout)
    with torch.no_grad():  # 禁用梯度计算，节省内存
        for batch_x, batch_y in test_data_loader:
            # (1).将输入和标签移动到指定设备
            inputs = batch_x.to(device)
            targets = batch_y.to(device)
            # (2).前向传播，得到原始Logits
            output = model(inputs)  # (batch_size, num_class)
            # (3).获取预测类别
            preds = output.argmax(dim=1)  # (batch_size,)
            # (4).统计预测正确的样本数量
            acc += (preds == targets).sum().item()
            # (5).累计当前Batch的样本数量
            total += targets.size(0)
    # (6).计算整个测试集的准确率
    print(f"Acc: {acc/total:.2f}")

if __name__ == "__main__":
    main()