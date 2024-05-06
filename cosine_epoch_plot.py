import pandas as pd
import matplotlib.pyplot as plt

# 读取CSV文件
df = pd.read_csv('/Users/fanzhe/Downloads/model_output_record/training_polarity_sentence-transformers-all-MiniLM-L12-v2-2024-04-05_02-58-31/eval/accuracy_evaluation_polarity-dev_results.csv')

# 筛选steps值为-1的行
filtered_df = df[df['steps'] == -1]

# 确保数据是按照epoch排序的
filtered_df = filtered_df.sort_values(by='epoch')

# 绘制折线图
plt.figure(figsize=(10, 6))
# plt.plot(filtered_df['epoch'], filtered_df['cosine_spearman'], label='Cosine Spearman')
# plt.plot(filtered_df['epoch'], filtered_df['cosine_pearson'], label='Cosine Pearson')
plt.plot(filtered_df['epoch'], filtered_df['accuracy'], label='accuracy')

plt.xlabel('Epoch')
plt.ylabel('Value')
# plt.title('cosine spearman/cosine pearson vs Epoch(max_distance: 3, split_method: 2)')
plt.title('Accuracy vs Epoch(max_distance: 5, split_method: 2)')

plt.legend()
plt.show()
