import matplotlib.pyplot as plt
import pandas as pd

# 假设的数据
distances = [1, 2, 3, 5, 10]  # 横坐标：distance
split_method1 = [0.7886720341454116, 0.6401260894779031, 0.6335196119725803, 0.7877112812997623, 0.8701727747121112]  # 纵坐标：split method1 的 accuracy
split_method2 = [0.779760767376643, 0.6265087059211207, 0.5017832815080587, 0.38002520761365155, 0.33515144870635993]  # 纵坐标：split method2 的 accuracy
split_method3 = [0.8491679422442744, 0.8253889004715298, 0.8131672813076711, None, None]  # 纵坐标：split method2 的 accuracy
# 绘制折线图
plt.figure(figsize=(10, 10))
plt.plot(distances, split_method1, marker='o', label='Split Method 1')
plt.plot(distances, split_method2, marker='s', label='Split Method 2')
plt.plot(distances, split_method3, marker='p', label='Split Method 2')
for i, txt in enumerate(split_method1):
    if txt is not None:
        plt.text(distances[i], txt, f'{txt:.2f}', fontsize=9)

for i, txt in enumerate(split_method2):
    if txt is not None:
        plt.text(distances[i], txt, f'{txt:.2f}', fontsize=9)
for i, txt in enumerate(split_method3):
    if txt is not None:
        plt.text(distances[i], txt, f'{txt:.2f}', fontsize=9)
# 添加标题和标签
# plt.figtext(0.5, -0.05, 'This is a simple text below the plot.', wrap=True, horizontalalignment='center', fontsize=12)

plt.title('Relatedness vs Distance')
plt.xlabel('Distance')
plt.ylabel('Spearman')
plt.legend()
# 显示图表
plt.show()


