from pathlib import Path
import pandas as pd
import numpy as np

PATH = Path.cwd()/"DLdata"
train_data = pd.read_csv(PATH/"cnews.train.txt", sep="\t", names = ["label","content"])
test_data =pd.read_csv(PATH/"cnews.test.txt", sep="\t", names = ["label", "content"])
# print(train_data.head())
#   label                                            content
# 0    体育  马晓旭意外受伤让国奥警惕 无奈大雨格外青睐殷家军记者傅亚雨沈阳报道 来到沈阳，国奥队依然没有...
# 1    体育  商瑞华首战复仇心切 中国玫瑰要用美国方式攻克瑞典多曼来了，瑞典来了，商瑞华首战求3分的信心也...
# print(train_data.groupby("label").count())
# label     content
# 体育        5000
# 娱乐        5000
# ...
# 随机抽样 - 训练集
sampled_df = train_data.groupby('label').apply(lambda x: x.sample(n=500,random_state=1424)).reset_index(drop=True)
# 保存为txt文件
output_file_path = PATH/"cnews_train_sampled.txt"
with open(output_file_path, 'w', encoding='utf-8') as f:
    for index, row in sampled_df.iterrows():
        f.write(f"{row['label']}\t{row['content']}\n")
print(f"Sampled data saved to {output_file_path}")
# 随机抽样 - 测试集
sampled_df = test_data.groupby('label').apply(lambda x: x.sample(n=500,random_state=1424)).reset_index(drop=True)
# 保存为txt文件
output_file_path = PATH/"cnews_test_sampled.txt"
with open(output_file_path, 'w', encoding='utf-8') as f:
    for index, row in sampled_df.iterrows():
        f.write(f"{row['label']}\t{row['content']}\n")
print(f"Sampled data saved to {output_file_path}")

def read_file(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            parts = line.strip().split('\t')
            if len(parts) == 2:
                label, content = parts
                data.append(content) #只要content，不要label
    return data

class calculate_stats:
    def __init__(self,data):
        self.total_samples = len(data)
        self.len_ = []
        for content in data:
            self.len_.append(len(content))

        self.lower_quartile = np.percentile(self.len_, 25)
        self.median = np.median(self.len_)
        self.upper_quartile = np.percentile(self.len_, 75)
        self.percentile_90 = np.percentile(self.len_, 90)

    def stats(self):
        # 输出结果
        print(f"总字数: {sum(self.len_)}")
        print(f"样本数量: {self.total_samples}")
        print(f"平均每篇文章的字数: {sum(self.len_)/self.total_samples}")
        print(f"最长句子的字数:{max(self.len_)}")
        print(f"最短句子的字数:{min(self.len_)}")
        print(f"句子长度的25%分位数:{self.lower_quartile}")
        print(f"句子长度的50%分位数:{self.median}")
        print(f"句子长度的75%分位数:{self.upper_quartile}")
        print(f"句子长度的90%分位数:{self.percentile_90}")