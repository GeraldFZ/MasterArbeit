import re
import os


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
                # pattern = r'(?m)^\b(?<!\d)(\d+(\.\d+)*)\.\s'
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


if __name__ == '__main__':
    modify_and_number_txt_files('/Users/fanzhe/Desktop/master_thesis/Data/kialo_debatetree_data/results')
    # modify_and_number_txt_files('/home/users0/fanze/masterarbeit/englishdebates')
