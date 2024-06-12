import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

# 定义区间
intervals = [0, 1000, 10000, 100000, 1000000, 10000000, np.inf]
labels = ['<1000', '1000-10000', '10000-100000', '100000-1000000', '1000000-10000000', '>10000000']

# 初始化计数器
counts = [0] * (len(intervals) - 1)

# 计算每个区间的文件数量
sums = [0] * (len(intervals) - 1)

# 计算每个区间的文件数量和pair_nums总和
group_lessthan1000 = []
group_1000_10000 = []
group_10000_100000 = []
group_100000_1000000 = []
group_1000000_10000000 = []
group_morethan10000000 = []





save_dir = '/mount/studenten5/projects/fanze/masterarbeit_data/dataset_visualization'
debates_path = '/mount/studenten5/projects/fanze/masterarbeit_data/csv_nofilter'
files_path = [os.path.join(debates_path,file) for file in os.listdir(debates_path) if file.endswith('.csv')]
debates_pair_size = {}
total_pairs_num = 0
for file in files_path:
    try:
        df = pd.read_csv(file)
        pair_num = len(df)
        debates_pair_size[os.path.basename(file)] = pair_num
        total_pairs_num = total_pairs_num+ pair_num
        print(os.path.basename(file), pair_num)
    except Exception as e:
        print(f"Error reading {files_path}: {e}")

    if pair_num < 1000:
        group_lessthan1000.extend(file)
    if pair_num < 10000 and pair_num>= 1000:
        group_1000_10000.extend(file)
    if pair_num < 100000 and pair_num>= 10000:
        group_10000_100000.extend(file)
    if pair_num < 1000000 and pair_num>= 100000:
        group_100000_1000000.extend(file)
    if pair_num < 10000000 and pair_num>= 1000000:
        group_1000000_10000000.extend(file)
    if pair_num > 10000000:
        group_morethan10000000.extend(file)

print('group_lessthan1000',len(group_lessthan1000), 'group_1000_10000',len(group_1000_10000), 'group_10000_100000',len(group_10000_100000),'group_100000_1000000',len(group_100000_1000000),'group_1000000_10000000',len(group_1000000_10000000),'group_morethan10000000',len(group_morethan10000000))
print(len(files_path), total_pairs_num)

files = list(debates_pair_size.keys())
pair_nums = list(debates_pair_size.values())


for num in pair_nums:
    for i in range(1, len(intervals)):
        if num <= intervals[i]:
            counts[i-1] += 1  # 增加计数器
            sums[i-1] += num  # 累加pair_nums
            break
# 绘制直方图
plt.figure(figsize=(10, 6))
plt.hist(pair_nums, bins=30, color='blue', alpha=0.7)
plt.title('CSV Files Line Count Histogram')
plt.xlabel('pairs quantity')
plt.ylabel('Frequency')
histogram_path = os.path.join(save_dir, 'line_count_histogram.png')
plt.savefig(histogram_path)  # 保存直方图
plt.close()
# 绘制散点图
plt.figure(figsize=(10, 6))
plt.scatter(range(len(pair_nums)), pair_nums, color='red')
plt.title('CSV Files Line Count Scatter Plot')
plt.xlabel('File Index')
plt.ylabel('pairs quantity')
scatter_path = os.path.join(save_dir, 'line_count_scatter.png')
plt.savefig(scatter_path)  # 保存散点图
plt.close()

plt.figure(figsize=(10, 6))
plt.hist(pair_nums, bins=30, color='blue', alpha=0.7, log=True)
plt.title('CSV Files Line Count Histogram with Log Scale')
plt.xlabel('Pairs Quantity')
plt.ylabel('Frequency (Log Scale)')
plt.savefig(os.path.join(save_dir, 'line_count_histogram_log_scale.png'))
plt.close()

plt.figure(figsize=(10, 6))
plt.scatter(range(len(pair_nums)), pair_nums, color='red')
plt.yscale('log')
plt.title('CSV Files Line Count Scatter Plot with Log Scale')
plt.xlabel('File Index')
plt.ylabel('Pairs Quantity (Log Scale)')
plt.savefig(os.path.join(save_dir, 'line_count_scatter_log_scale.png'))
plt.close()



sorted_pairs = sorted(debates_pair_size.items(), key=lambda item: item[1], reverse=True)

# 解包排序后的数据
sorted_files, sorted_pair_nums = zip(*sorted_pairs)

# 绘制柱状图
plt.figure(figsize=(140, 10))  # 可能需要根据文件数量调整图的大小
plt.bar(range(len(sorted_files)), sorted_pair_nums, color='skyblue')
plt.xticks(range(len(sorted_files)), sorted_files, rotation=90)  # 将文件名旋转90度以防止重叠

# 添加标题和轴标签
plt.title('File vs Pair Nums')
plt.xlabel('Files')
plt.ylabel('Pair Nums')

# 显示图形
plt.tight_layout()  # 调整布局以防止标签被截断




# 如果需要保存图像到指定目录
save_dir = '/mount/studenten5/projects/fanze/masterarbeit_data/dataset_visualization'
plt.savefig(os.path.join(save_dir, 'files_vs_pair_nums_bar_chart2.png'))
plt.close()

plt.figure(figsize=(12, 10))
plt.bar(labels, sums, color='skyblue')  # 使用sums而不是counts
plt.title('Total Pair Numbers by Interval')
plt.xlabel('Pair Numbers Interval')
plt.ylabel('Total Pair Numbers')
plt.xticks(rotation=45)  # 旋转x轴标签以便更好地展示

# 保存图像到指定目录
plt.savefig(os.path.join(save_dir, 'total_pair_nums_by_interval.png'))
plt.close()
