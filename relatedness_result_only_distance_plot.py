import matplotlib.pyplot as plt
import pandas as pd

# 假设的数据
distances = [1, 2, 3, 5, 10]  # 横坐标：distance
split_method1 = [0.7729954774299691, 0.7005292651350472, 0.577526935386136, 0., 15188742421788876]  
split_method2 = [0.753716178618729, 0.6169659246044585, 0.50640784879033, 0.3489178016577572, 0.27185143779377663]  
# split_method3 = [0.8331035768801544, 0.8229363304593558, 0.8078326747774912, , 0.6911911372358717] 
split_method3 = [0.8260917103373687, 0.7985107125786965, 0.7267841706612673, 0.5899739568392547, 0.5057420250071126] 

# 3-5,3-10

# 绘制折线图
data_info = [
    "Method 1",
    "Distance 1: Train 78524, Dev 9815, Test 9816",
    "Distance 2: Train 193287, Dev 24161, Test 24161",
    "Distance 3: Train 404961, Dev 50620, Test 50621",
    "Distance 5: Train 2386982, Dev 298373, Test 298373",
    "Distance 10: Train 575160, Dev 71895, Test 71895",
    "Method 2",
    "Distance 1: Train 80295, Dev 9384, Test 8476",
    "Distance 2: Train 258890, Dev 30276, Test 27324",
    "Distance 3: Train 633910, Dev 72825, Test 67217",
    "Distance 5: Train 2455913, Dev 272825, Test 254990",
    "Distance 10: Train 7044829, Dev 758133, Test 677893",
    "Method 3",
    "Distance 1: Train 107846, Dev 7054, Test 9444",
    "Distance 2: Train 256537, Dev 14592, Test 19836",
    "Distance 3: Train 553195, Dev 24529, Test 33139",
    "Distance 5: Train 1910910, Dev 50109, Test 65505",
    "Distance 10: Train 5302606, Dev 93878, Test 108374"
]

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
plt.title('Cosine similarity spearman vs Distance')
plt.xlabel('Distance')
plt.ylabel('Cosine similarity spearman')
plt.legend()
# 添加数据量信息的旁注
data_text = "\n".join(data_info)
# 留出空间在右侧显示
plt.subplots_adjust(right=0.7)  # 调整图表大小以留出空间
ax.text(1.02, 0.5, data_text, transform=ax.transAxes, verticalalignment='center')

# 显示图表
plt.show()


