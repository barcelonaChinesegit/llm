from pathlib import Path
import pandas as pd
import numpy as np

PATH = Path.cwd()/"DLdata"
train_data = pd.read_csv(PATH/"cnews.train.txt", sep="\t", names = ["label","content"])
test_data =pd.read_csv(PATH/"cnews.test.txt", sep="\t", names = ["label", "content"])

# print(train_data.head())
# #   label                                            content
# # 0    体育  马晓旭意外受伤让国奥警惕 无奈大雨格外青睐殷家军记者傅亚雨沈阳报道 来到沈阳，国奥队依然没有...
# # 1    体育  商瑞华首战复仇心切 中国玫瑰要用美国方式攻克瑞典多曼来了，瑞典来了，商瑞华首战求3分的信心也...
# print(train_data.groupby("label").count())
# # label     content
# # 体育        5000
# # 娱乐        5000
# # 家居        5000
# # 房产        5000
# # 教育        5000
# # 时尚        5000
# # 时政        5000
# # 游戏        5000
# # 科技        5000
# # 财经        5000

# 随机抽样 - 训练集
np.random.seed(1412)
sampled_df = train_data.groupby('label').apply(lambda x: x.sample(n=500)).reset_index(drop=True)

# 保存为txt文件
output_file_path = PATH/"cnews_train_sampled.txt"
with open(output_file_path, 'w', encoding='utf-8') as f:
    for index, row in sampled_df.iterrows():
        f.write(f"{row['label']}\t{row['content']}\n")
print(f"Sampled data saved to {output_file_path}")

