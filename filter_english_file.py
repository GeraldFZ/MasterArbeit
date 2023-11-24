from langdetect import detect
import os
import shutil

# 遍历文件夹中的所有txt文件


def filter_english_txt_files_and_copy(folder_path, destination_folder):
    english_txt_files = []
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".txt"):
            file_path = os.path.join(folder_path, file_name)
            with open(file_path, 'r', encoding='utf-8') as file:
                file_content = file.read()
                if detect(file_content) == "en":
                    english_txt_files.append(file_path)
                    shutil.copy(file_path, destination_folder)
    return english_txt_files

if __name__ == "__main__":

    # print(filter_english_txt_files_and_copy('/Users/fanzhe/Desktop/master_thesis/Data/kialo_debatetree_data/discussions', '/Users/fanzhe/Desktop/master_thesis/Data/kialo_debatetree_data/results'))
    english_txt_files = filter_english_txt_files_and_copy('/Users/fanzhe/Desktop/master_thesis/Data/kialo_debatetree_data/testsample', '/Users/fanzhe/Desktop/master_thesis/Data/kialo_debatetree_data/testsample_english')
    # '/home/users0/fanze/masterarbeit/results'

    # english_txt_files = filter_english_txt_files_and_copy('/home/users0/fanze/masterarbeit/origindata/origindata/discussions', '/home/users0/fanze/masterarbeit/englishdebates')

    num = 0
    if len(english_txt_files) > 0:
        print("英文文本文件列表：")
        for file_path in english_txt_files:
            print(file_path)
            num = num + 1

            print(num)
    else:
        print("未找到英文文本文件。")