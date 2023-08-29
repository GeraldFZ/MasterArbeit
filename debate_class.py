import os
import re

class Debate:
    def __init__(self, debate_topic):
        self.debate_topic = debate_topic
        self.arguments = []

    class Argument:
        def __init__(self, index, content, relative_polarity,relative_polarity_value ):
            self.index = index
            self.content = content
            self.relative_polarity = relative_polarity
            self.relative_polarity_value = relative_polarity_value
            # self.absolute_polarity = absolute_polarity
            # self.absolute_polarity_value = absolute_polarity_value
        def find_last_dot(self, lst):
            dot_positions = [i for i, char in enumerate(lst) if char == "."]
            if len(dot_positions) >= 2:
                last_dot_position = dot_positions[-2]
                return last_dot_position
            else:
                return None

        # def absolute_polarity_compute(self, Arguments):
        #
        #     # for argument in Arguments:
        #     #     origin_index = argument.index
        #     relative_polarity_value = self.relative_polarity_value
        #     # origin_index_list = list(origin_index)
        #
        #     # print("index:", origin_index, "relative_polarity_value:", relative_polarity_value)
        #
        #     result = relative_polarity_value
        #     current_index_list = list(self.index)
        #     last_dot_position = self.find_last_dot(current_index_list)
        #
        #     while last_dot_position is not None:
        #         for argument in Arguments:
        #             if argument.index == "".join(current_index_list[:last_dot_position + 1]):
        #                 result = argument.relative_polarity_value * result
        #                 current_index_list = "".join(current_index_list[:last_dot_position + 1])
        #                 last_dot_position = self.find_last_dot(current_index_list)
        #                 print("absolute_polarity=", result)
        #
        #                 break  # 找到匹配项后，跳出内部循环
                    # return result
        def absolute_polarity_compute(self, relative_polarity_mapping):
            result = self.relative_polarity_value

            current_index_list = list(self.index)
            last_dot_position = self.find_last_dot(current_index_list)

            while last_dot_position is not None:
                index_key = "".join(current_index_list[:last_dot_position + 1])
                if index_key in relative_polarity_mapping:
                    result *= relative_polarity_mapping[index_key]

                current_index_list = current_index_list[:last_dot_position]
                last_dot_position = self.find_last_dot(current_index_list)

            return result





    def add_argument(self,index, content, relative_polarity, relative_polarity_value):
        # if relative_polarity == "pro":
        #     relative_polarity_value = 1
        # elif relative_polarity == "con":
        #     relative_polarity_value = -1
        argument = self.Argument(index, content, relative_polarity, relative_polarity_value)

        self.arguments.append(argument)

    def load_debate_from_txt(self, file_path):
        with open(file_path, "r", encoding='utf-8') as file:
            lines = file.readlines()


            for index_of_argument_debate, line in enumerate(lines):

                match = re.match(r'(?m)^\b(?<!\d)(\d+(\.\d+)*)\.\s', line)
                if index_of_argument_debate == 0:
                    self.debate_topic = line.replace(match.group(0), '').strip()
                    index = match.group(0)
                    text_content = line.replace(match.group(0), '').strip()
                    relative_polarity = "Pro"
                    content = text_content.strip()
                    relative_polarity_value = 10 if relative_polarity == "Pro" else -10


                    self.add_argument(index, content, relative_polarity, relative_polarity_value)
                elif match and index_of_argument_debate != 0:
                    # argument.relative_polarity_value = None
                    index = match.group(0)
                    text_content = line.replace(match.group(0), '').strip()
                    relative_polarity = text_content[:3].strip()
                    relative_polarity_value = 10 if relative_polarity == "Pro" else -10
                    content = text_content[4:].strip()
                    self.add_argument(index, content, relative_polarity, relative_polarity_value)




def load_debates_from_folder(folder_path):
    debate_instances = []
    txt_files = [file for file in os.listdir(folder_path) if file.endswith('.txt')]

    for txt_file in txt_files:
        debate_instance = Debate("")  # 创建空的Debate实例
        txt_file_path = os.path.join(folder_path, txt_file)
        debate_instance.load_debate_from_txt(txt_file_path)
        debate_instances.append(debate_instance)

    return debate_instances

# 示例用法
# if __name__ == "__main__":
#     folder_path = "your_folder_path"  # 替换为实际文件夹路径
#     debates = load_debates_from_folder('/Users/fanzhe/Desktop/master_thesis/Data/kialo_debatetree_data/dataprocesstest')
#
#     for debate in debates:
#         print("topic:", debate.debate_topic)
#         for argument in debate.arguments:
#             # absolute_polarity = argument.absolute_polarity_compute(debate.arguments)
#             # print(argument)
#             argument.absolute_polarity_compute(debate.arguments)
#
#             print("index:", argument.index,"content:", argument.content, "relative polarity:", argument.relative_polarity, "relative polarity value:", argument.relative_polarity_value)


if __name__ == "__main__":
    folder_path = "your_folder_path"  # 替换为实际文件夹路径
    debates = load_debates_from_folder('/Users/fanzhe/Desktop/master_thesis/Data/kialo_debatetree_data/dataprocesstest')

    # 创建索引和相对极性值的映射
    relative_polarity_mapping = {}
    for debate in debates:
        for argument in debate.arguments:
            relative_polarity_mapping[argument.index] = argument.relative_polarity_value

    for debate in debates:
        print("topic:", debate.debate_topic)
        for argument in debate.arguments:
            absolute_polarity = argument.absolute_polarity_compute(relative_polarity_mapping)
            print("index:", argument.index, "content:", argument.content, "relative polarity:", argument.relative_polarity, "relative polarity value:", argument.relative_polarity_value, "absolute polarity:", absolute_polarity)