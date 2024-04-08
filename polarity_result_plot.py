import matplotlib.pyplot as plt

# 假设的数据
distances = [1, 2, 3]  # 横坐标：distance
split_method1 = [0.7580590430946725, 0.8125, 0.9518168338143146]  # 纵坐标：split method1 的 accuracy
split_method2 = [0.7633359872611465, 0.6337948427608214, 0.5788978543257227]  # 纵坐标：split method2 的 accuracy

# 绘制折线图
plt.figure(figsize=(10, 6))
plt.plot(distances, split_method1, marker='o', label='Split Method 1')
plt.plot(distances, split_method2, marker='s', label='Split Method 2')

# 添加标题和标签
plt.title('Polarity Consistency Accuracy vs Distance')
plt.xlabel('Distance')
plt.ylabel('Accuracy')
plt.legend()

# 显示图表
plt.show()
