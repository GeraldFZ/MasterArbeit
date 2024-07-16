import matplotlib.pyplot as plt

# 假设的数据
distances = [1, 2, 3, 5, 10]  # 横坐标：distance
split_method1 = [0.7283678316932474, 0.8076029825204742, 0.9121688414393041, 0.9776421666576659, 0.9215591186009153]  # 纵坐标：split method1 的 accuracy
split_method2 = [0.7470143312101911, 0.6399861159319681, 0.5856883805272702, , 0.5216129032258064]  # 纵坐标：split method2 的 accuracy
split_method3 = [0.7284231840040496, 0.6360080746908907, 0., 0., 0.]
# 1-10, 2-10, 3-10
# 数据量信息
data_info = [
    "Method 1",
    "Distance 1: Train 47145, Dev 5893, Test 5894",
    "Distance 2: Train 130896, Dev 16362, Test 16362",
    "Distance 3: Train 323710, Dev 40464, Test 40464",
    "Distance 5: Train 1036948, Dev 129618, Test 129619",
    "Distance 10: Train 527886, Dev 65986, Test 65986",
    "Method 2",
    "Distance 1: Train 48271, Dev 5637, Test 5024",
    "Distance 2: Train 133622, Dev 15593, Test 14405",
    "Distance 3: Train 330270, Dev 38523, Test 35845",
    "Distance 5: Train , Dev , Test ",
    "Distance 10: Train 559796, Dev 56662, Test 43400",
    "Method 3",
    "Distance 1: Train 44805, Dev 2994, Test 3951",
    "Distance 2: Train 117226, Dev 5863, Test 7926",
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
plt.title('Accuracy vs Single Distance')
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