import pandas as pd
import matplotlib.pyplot as plt

# 加载CSV文件
df = pd.read_csv('/Users/fanzhe/Downloads/model_output/training_stsbenchmark_sentence-transformers-all-MiniLM-L12-v2-2024-02-22_01-05-14/eval/similarity_evaluation_sts-dev_results.csv')



# 获取唯一的epoch值
epochs = df['epoch'].unique()

# 设置不同的颜色以区分不同的epoch
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']

plt.figure(figsize=(10, 6))

# 对于每个epoch，绘制其按steps排序的cosine_pearson和cosine_spearman值
for i, epoch in enumerate(epochs):
    # 筛选当前epoch的数据并按steps排序
    epoch_data = df[df['epoch'] == epoch].sort_values(by='steps')

    # 绘制 cosine_pearson
    plt.plot(epoch_data['steps'], epoch_data['cosine_pearson'], label=f'Epoch {epoch} Cosine Pearson',
             color=colors[i % len(colors)], marker='o')
    plt.scatter(epoch_data['steps'], epoch_data['cosine_pearson'], color=colors[i % len(colors)])

    # 绘制 cosine_spearman
    plt.plot(epoch_data['steps'], epoch_data['cosine_spearman'], label=f'Epoch {epoch} Cosine Spearman',
             color=colors[i % len(colors)], marker='s')
    plt.scatter(epoch_data['steps'], epoch_data['cosine_spearman'], color=colors[i % len(colors)])

# 添加图例
plt.legend()

# 添加标题和轴标签
plt.title('Cosine Similarity Metrics Over Steps for Different Epochs')
plt.xlabel('Steps')
plt.ylabel('Similarity Score')

# 显示图表
plt.show()
