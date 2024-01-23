import os
import pandas as pd

debates_path = '/mount/studenten5/projects/fanze/masterarbeit_data/csv_nofilter'
all_files = os.listdir(debates_path)

csv_files = [file for file in all_files if file.endswith('.csv')]

files_has_0_distance = set()  # 使用集合来避免重复
print("running")
for file in csv_files:
    file_path = os.path.join(debates_path, file)
    try:
        df = pd.read_csv(file_path)
        if (df['distance'] == 0).any():  # 检查是否有distance为0的行
            files_has_0_distance.add(file_path)
            print(file_path)
    except Exception as e:  # 添加错误处理
        print(f"Error reading {file_path}: {e}")

file_name = "/mount/studenten5/projects/fanze/masterarbeit/files_has_0_distance.txt"

with open(file_name, "w") as file:
    for line in files_has_0_distance:
        file.write(line + "\n")
