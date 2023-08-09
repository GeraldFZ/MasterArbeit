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