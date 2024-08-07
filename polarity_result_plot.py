import matplotlib.pyplot as plt

# 假设的数据
distances = [1, 2, 3, 5, 10]  # 横坐标：distance
split_method1 = [0.7283678316932474, 0.8189701653486701, 0.9464277172786556, 0.9873948224125507, 0.9980742152911177]  # 纵坐标：split method1 的 accuracy
split_method2 = [0.7470143312101911, 0.6448093056770806, 0.5956145746643992, 0.5481450691925299, 0.5298006004309618]  # 纵坐标：split method2 的 accuracy
split_method3 = [0.7284231840040496, 0.6409025848278185, 0.597301286105840232, 0.5642534103026069, 0.539390914667748]
# 1-10, 2-10, 3-10
# 数据量信息
data_info = [
    "Method 1",
    "Distance 1: Train 47145, Dev 5893, Test 5894",
    "Distance 2: Train 178041, Dev 22255, Test 22256",
    "Distance 3: Train 501752, Dev 62719, Test 62719",
    "Distance 5: Train 2231461, Dev 278933, Test 278933",
    "Distance 10: Train 6613407, Dev 826676, Test 826676",
    "Method 2",
    "Distance 1: Train 48271, Dev 5637, Test 5024",
    "Distance 2: Train 181893, Dev 21230, Test 19429",
    "Distance 3: Train 512163, Dev 59753, Test 55274",
    "Distance 5: Train 2294728, Dev 256423, Test 238176",
    "Distance 10: Train 6867405, Dev 740827, Test 658527",
    "Method 3",
    "Distance 1: Train 44805, Dev 2994, Test 3951",
    "Distance 2: Train 162031, Dev 8857, Test 11877",
    "Distance 3: Train 429419, Dev 17396, Test 23715",
    "Distance 5: Train 1758640, Dev 40891, Test 54394",
    "Distance 10: Train 5138358, Dev 82987, Test 96177"
]

# 绘制折线图
plt.figure(figsize=(12, 6))
ax = plt.subplot(111)  # 添加子图

plt.plot(distances, split_method1, marker='o', label='Split Method 1')
plt.plot(distances, split_method2, marker='s', label='Split Method 2')
plt.plot(distances, split_method3, marker='p', label='Split Method 3')

# 设置y轴范围和刻度
plt.ylim(0, 1)  # 设置y轴的范围从0到1
plt.yticks([0,0.1, 0.2,0.3, 0.4,0.5, 0.6,0.7, 0.8,0.9, 1.0])  # 设置y轴的刻度

# 添加网格线
plt.grid(True, which='both', linestyle='--', linewidth=0.5)

# 添加标题和标签
plt.title('Accuracy vs Distance')
plt.xlabel('Distance')
plt.ylabel('Cosine accuracy')
plt.legend()
# 添加数据量信息的旁注
data_text = "\n".join(data_info)
# 留出空间在右侧显示
plt.subplots_adjust(right=0.7)  # 调整图表大小以留出空间
ax.text(1.02, 0.5, data_text, transform=ax.transAxes, verticalalignment='center')

# 显示图表
plt.show()