import re
import os
import sys


def modify_and_number_txt_files(folder_path):
    # 获取文件夹中所有txt文件
    txt_files = [f for f in os.listdir(folder_path) if f.endswith('.txt')]

    for file_number, file_name in enumerate(txt_files, start=1):
        file_path = os.path.join(folder_path, file_name)

        # 读取文本文件内容
        modified_content = []
        with open(file_path, 'r') as file:
            lines = file.readlines()
            # for line_index, line in enumerate(lines, start=1):
            #
            #
            #
            #     # 使用正则表达式找到所有类似于1.2.3.1.2.的序号
            #
            #     pattern = r'(?m)^\b(?<!\d)(\d+(\.\d+)*)\b(?=\.\s|$)'
            #     # pattern = r'(?m)^\b(?<!\d)(\d+(\.\d+)*)\.\s'
            #     # pattern = r'(?m)^\b(?<!\d)(\d+(\.\d+)*\.)\b(?=\s|$)'
            #
            #     matches = re.findall(pattern, line)
            #     if matches:
            #         # print("haha",matches[0][0])
            #         if line_index > 4 and matches[0][0].count('.') == 0 :
            #             lines[line_index - 2] = lines[line_index - 2].strip() + ' ' + lines[line_index - 1]
            #             del lines[line_index - 1]
            #             print("haha")
            #         for match in matches:
            #             old_number = f"{match[0]}" + '.'
            #             # print(old_number)
            #             new_number = f"{file_number}" + old_number[old_number.find('.'):]
            #             modified_line = line.replace(old_number, new_number)
            #             print(old_number, new_number)
            #             modified_content.append(modified_line)
            #             break

            # 收集需要合并的行索引
            lines_to_merge = []
            for line_index, line in enumerate(lines, start=1):
                pattern = r'(?m)^\b(?<!\d)(\d+(\.\d+)*)\b(?=\.\s|$)'
                matches = re.findall(pattern, line)
                if matches and line_index > 4 and matches[0][0].count('.') == 0:
                    lines_to_merge.append(line_index)

            # 根据收集的索引合并行
            for index in sorted(lines_to_merge, reverse=True):
                lines[index - 2] = lines[index - 2].strip() + ' ' + lines[index - 1]
                del lines[index - 1]

            # 修改序号并更新modified_content
            modified_content = []
            for line in lines:
                matches = re.findall(pattern, line)
                if matches:
                    old_number = f"{matches[0][0]}."
                    new_number = f"{file_number}" + old_number[old_number.find('.'):]
                    line = line.replace(old_number, new_number)
                modified_content.append(line)

        # 读取文本文件内容
        # with open(file_path, 'r') as file:
        #     lines = file.readlines()
        #
        # # 修改行内容和合并行
        # modified_content = []
        # for line_index, line in enumerate(lines, start=1):
        #     # 使用正则表达式找到所有类似于1.2.3.1.2.的序号
        #     pattern = r'(?m)^\b(?<!\d)(\d+(\.\d+)*)\b(?=\.\s|$)'
        #     matches = re.findall(pattern, line)
        #     if matches:
        #         # 修改序号
        #         old_number = f"{matches[0][0]}."
        #         new_number = f"{file_number}" + old_number[old_number.find('.'):]
        #         line = line.replace(old_number, new_number)
        #
        #     modified_content.append(line)



        # 将内容写回文件
        with open(file_path, 'w') as file:
            file.writelines(modified_content)

        # 打印文件名和序号
        print(f'{file_name} - {file_number}')


if __name__ == '__main__':
    # modify_and_number_txt_files('/Users/fanzhe/Desktop/master_thesis/Data/kialo_debatetree_data/testsample_english')
    modify_and_number_txt_files('/mount/studenten5/projects/fanze/masterarbeit_data/englishdebates/')

    # modify_and_number_txt_files(sys.argv[1])

    # modify_and_number_txt_files ('/Users/fanzhe/Desktop/master_thesis/Data/kialo_debatetree_data/testsample_english/')
