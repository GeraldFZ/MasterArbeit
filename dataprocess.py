# import os
#
# def number_files(directory):
#     file_dict = {}
#     debatetree_index= 1
#     for filename in os.listdir(directory):
#         if filename.endswith(".txt"):
#             filepath = os.path.join(directory, filename)
#             #生成键值对存入file_dict字典中
#             file_dict[filepath] = debatetree_index
#             debatetree_index += 1
#     return file_dict
#
#
# def modify_first_digit_in_number(text, specified_digit):
#     lines = text.split('\n')
#     modified_lines = []
#
#     for line in lines:
#         new_line = line
#         # 使用正则表达式找到类似1.2.3.1.2.的序号
#         import re
#         matches = re.findall(r'\d+\.\d+\.\d+\.\d+\.\d+\.', line)
#         for match in matches:
#             parts = match.split('.')
#             if len(parts) > 0:
#                 # 将第一个数字替换为指定的数字
#                 modified_number = f"{specified_digit}.{ '.'.join(parts[1:]) }."
#                 new_line = new_line.replace(match, modified_number, 1)
#         modified_lines.append(new_line)
#
#     modified_text = '\n'.join(modified_lines)
#     return modified_text
#
#
# # 指定包含txt文件的目录
# directory = "/Users/fanzhe/Desktop/master_thesis/Data/kialo_debatetree_data/dataprocesstest"
# file_dict = number_files(directory)
# # modify_first_digit_in_number(directory)
#
# # # 打印每个文件对应的编号
# for filepath, file_number in file_dict.items():
#     print(f"File: {filepath}, Number: {file_number}")

import os
import re
import pandas as pd

def modify_and_number_txt_files(folder_path):
    # 获取文件夹中所有txt文件
    txt_files = [f for f in os.listdir(folder_path) if f.endswith('.txt')]

    for file_number, file_name in enumerate(txt_files, start=1):
        file_path = os.path.join(folder_path, file_name)

        # 读取文本文件内容
        modified_content = []
        with open(file_path, 'r') as file:
            for line in file:
                # 使用正则表达式找到所有类似于1.2.3.1.2.的序号

                pattern = r'(?m)^\b(?<!\d)(\d+(\.\d+)*)\b(?=\.\s|$)'
                # pattern = r'(?m)^\b(?<!\d)(\d+(\.\d+)*\.)\b(?=\s|$)'

                matches = re.findall(pattern, line)
                if matches:
                    for match in matches:
                        old_number = f"{match[0]}" + '.'
                        # print(old_number)
                        new_number = f"{file_number}" + old_number[old_number.find('.'):]
                        modified_line = line.replace(old_number, new_number)
                        print(old_number, new_number)
                        modified_content.append(modified_line)
                        break


        # 将内容写回文件
        with open(file_path, 'w') as file:
            file.writelines(modified_content)

        # 打印文件名和序号
        print(f'{file_name} - {file_number}')

# 替换目标文件夹的路径
folder_path = '/Users/fanzhe/Desktop/master_thesis/Data/kialo_debatetree_data/discussions'
modify_and_number_txt_files(folder_path)


# def convert_txt_to_csv(input_folder, output_folder):
#     # 获取文件夹中所有txt文件的路径
#     txt_files = [f for f in os.listdir(input_folder) if f.endswith('.txt')]
#
#     for txt_file in txt_files:
#         txt_file_path = os.path.join(input_folder, txt_file)
#
#         # 读取txt文件内容，这里假设每个txt文件中的内容以空格分隔，并且有三列
#         with open(txt_file_path, 'r') as file:
#             content = file.read()
#             lines = content.strip().split('\n')
#             data = [line.split() for line in lines]
#
#         # 转换为DataFrame
#         df = pd.DataFrame(data, columns=['Column1', 'Column2', 'Column3'])
#
#         # 将DataFrame保存为csv文件
#         csv_file_name = os.path.splitext(txt_file)[0] + '.csv'
#         csv_file_path = os.path.join(output_folder, csv_file_name)
#         df.to_csv(csv_file_path, index=False)
#



# 替换输入和输出文件夹的路径
# input_folder = 'path/to/input_folder'
# output_folder = 'path/to/output_folder'
#
# convert_txt_to_csv(input_folder, output_folder)