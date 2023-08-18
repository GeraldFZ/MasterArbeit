import os
import re

class Debate:
    def __init__(self, debate_topic):
        self.debate_topic = debate_topic
        self.arguments = []

    class Argument:
        def __init__(self, index, content, relative_polarity, absolute_polarity):
            self.index = index
            self.content = content
            self.relative_polarity = relative_polarity
            self.absolute_polarity = absolute_polarity

    def add_argument(self, index, content, relative_polarity, absolute_polarity):
        argument = self.Argument(index, content, relative_polarity, absolute_polarity)
        self.arguments.append(argument)


    def load_debate_from_txt(self, file_path):
        with open(file_path, "r", encoding='utf-8') as file:
            lines = file.readlines()


            for line in lines[0:]:
                # match = re.match(r'(?m)^\b(?<!\d)(\d+(\.\d+)*)\b(?=\.\s|$)', line)
                match = re.match(r'(?m)^\b(?<!\d)(\d+(\.\d+)*)\.\s', line)

                if match:
                    index = match.group(0)
                    content = line.replace(match.group(0), '').strip()
                    relative_polarity= content[:3]
                    absolute_polarity = -1
                    self.add_argument(index, content, relative_polarity, absolute_polarity)
                    if line == lines[0]:
                        self.debate_topic = content



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
if __name__ == "__main__":
    folder_path = "your_folder_path"  # 替换为实际文件夹路径
    debates = load_debates_from_folder('/Users/fanzhe/Desktop/master_thesis/Data/kialo_debatetree_data/dataprocesstest')

    for debate in debates:
        print(debate.debate_topic)
        for argument in debate.arguments:
            print(f"Argument Index: {argument.index}", argument.content, argument.relative_polarity)

