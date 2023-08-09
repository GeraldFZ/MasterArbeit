import os
from langdetect import detect


# 遍历文件夹中的所有txt文件
def filter_english_txt_files(folder_path):
    english_txt_files = []
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".txt"):
            file_path = os.path.join(folder_path, file_name)
            with open(file_path, 'r', encoding='utf-8') as file:
                file_content = file.read()
                if detect(file_content) == "en":
                    english_txt_files.append(file_path)
    return english_txt_files


if __name__ == "__main__":
    folder_path = "your_folder_path_here"  # 替换为你的文件夹路径
    english_txt_files = filter_english_txt_files('/Users/fanzhe/Desktop/master_thesis/Data/kialo_debatetree_data/discussions')
    num = 0
    if len(english_txt_files) > 0:
        print("英文文本文件列表：")
        for file_path in english_txt_files:
            print(file_path)
            num = num + 1

            print(num)
    else:
        print("未找到英文文本文件。")
