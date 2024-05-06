import matplotlib.pyplot as plt
import pandas as pd

# 假设的数据
distances = [1, 2, 3, 5]  # 横坐标：distance
split_method1 = [0.8445153285727189, 0.8407305126020289, 0.8635237155347827, None]  # 纵坐标：split method1 的 accuracy
split_method2 = [0.7542364335606256, 0.6159045643038368, 0.5084601235190208, 0.33825530698003514]  # 纵坐标：split method2 的 accuracy

# 绘制折线图
plt.figure(figsize=(10, 6))
plt.plot(distances, split_method1, marker='o', label='Split Method 1')
plt.plot(distances, split_method2, marker='s', label='Split Method 2')
for i, txt in enumerate(split_method1):
    if txt is not None:
        plt.text(distances[i], txt, f'{txt:.2f}', fontsize=9)

for i, txt in enumerate(split_method2):
    if txt is not None:
        plt.text(distances[i], txt, f'{txt:.2f}', fontsize=9)
# 添加标题和标签
plt.title('Relatedness vs Distance')
plt.xlabel('Distance')
plt.ylabel('Spearman')
plt.legend()

# 显示图表
plt.show()


