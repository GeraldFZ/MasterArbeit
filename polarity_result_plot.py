import matplotlib.pyplot as plt

# 假设的数据
distances = [1, 2, 3, 5]  # 横坐标：distance
split_method1 = [0.7220902612826603, 0.8290797987059669, None, None]  # 纵坐标：split method1 的 accuracy
split_method2 = [0.7255175159235668, 0.6468166143393895, None, None]  # 纵坐标：split method2 的 accuracy
split_method3 = [0.709440647937231, 0.5986928104575163, 0.5296125554212011, None]
# 绘制折线图
plt.figure(figsize=(10, 10))
plt.plot(distances, split_method1, marker='o', label='Split Method 1')
plt.plot(distances, split_method2, marker='s', label='Split Method 2')
plt.plot(distances, split_method3, marker='p', label='Split Method 3')
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
plt.title('Polarity Consistency Accuracy vs Distance')
plt.xlabel('Distance')
plt.ylabel('Accuracy')
plt.legend()

# 显示图表
plt.show()
