import os
import re
import random
import pandas
import dataprocess
from tqdm import tqdm
import time

class Debate:
    def __init__(self, debate_topic):
        self.debate_topic = debate_topic
        self.arguments = []

    class Argument:
        def __init__(self, index, content, relative_polarity, relative_polarity_value):
            self.index = index
            self.content = content
            self.relative_polarity = relative_polarity
            self.relative_polarity_value = relative_polarity_value

        def find_last_dot(self, lst):
            dot_positions = [i for i, char in enumerate(lst) if char == "."]
            if len(dot_positions) >= 2:
                last_dot_position = dot_positions[-2]
                return last_dot_position
            else:
                return None

        def distance_relatedness_compute(self, Arguments, count1):
            index = self.index.strip(". ")
            index_list = index.split(".")
            # distance = 0  # 默认值
            # relatedness = 1
            relatedness_distance_set = []

            # for argument in Arguments:
            for count2, argument in enumerate(Arguments, start=1):

                index_2 = argument.index.strip(". ")
                index_2_list = index_2.split(".")
                if count1 < count2:
                    # print(count1,count2)
                    # common_length = min(len(index_list), len(index_2_list))
                    # for i in range(common_length):
                    #     if index_list[i] != index_2_list[i]:
                    #         break
                    #
                    # distance = (len(index_list) - i) + (len(index_2_list) - i)
                    if len(index_list) == len(index_2_list):
                        if index_list == index_2_list:
                            # print("same", index_list, index_2_list)
                            # print("相同", index_list, index_2_list)

                            distance = 0
                            relatedness = 1
                            return_index_1 = index+"."
                            return_index_2 = index_2+"."
                            content_2 = argument.content
                            content_1 = self.content
                            polarity_2 = argument.absolute_polarity_compute(Arguments)

                            if distance <3 :


                                # with relatedness
                                # relatedness_distance_set.append(
                                #         {"index_1": return_index_1, "content_1": content_1, "index_2": return_index_2, "content_2": content_2, "polarity_2": polarity_2,  "distance": distance,
                                #              "relatedness": relatedness})
                                relatedness_distance_set.append(
                                    {"index_1": return_index_1, "content_1": content_1, "index_2": return_index_2,
                                     "content_2": content_2, "polarity_2": polarity_2, "distance": distance,
                                     })

                                    # print("相同", relatedness_distance_set)


                        if index_list != index_2_list:
                            # print("长短一样但不相同", index_list, index_2_list)

                            for i in range(len(index_list)):

                                if index_list[i] != index_2_list[i]:

                                    distance = len(index_list[i:]) + len(index_2_list[i:])
                                    relatedness = 1 / distance
                                    return_index_1 = index+"."
                                    return_index_2 = index_2+"."
                                    content_1 = self.content
                                    content_2 = argument.content
                                    polarity_2 = argument.absolute_polarity_compute(Arguments)

                                    if distance < 3:
                                        # with relatedness
                                        # relatedness_distance_set.append(
                                        #         {"index_1": return_index_1, "content_1": content_1, "index_2": return_index_2, "content_2": content_2, "polarity_2": polarity_2,  "distance": distance,
                                        #              "relatedness": relatedness})
                                        relatedness_distance_set.append(
                                            {"index_1": return_index_1, "content_1": content_1,
                                             "index_2": return_index_2,
                                             "content_2": content_2, "polarity_2": polarity_2, "distance": distance,
                                             })
                                        break

                                            # print("长短一样但不相同", relatedness_distance_set)



                    elif len(index_list) != len(index_2_list):
                        long_list = max(index_list, index_2_list, key=len)
                        short_list = min(index_list, index_2_list, key=len)
                        if all(short_elem == long_elem for short_elem, long_elem in zip(short_list, long_list)):
                            # print("长短不一而且第一个不同的元素没有出现在短的列表中", relatedness_distance_set)
                            distance = len(long_list) - len(short_list)
                            # print(long_list,short_list)
                            relatedness = 1 / distance
                            return_index_1 = index+"."
                            content_1 = self.content
                            return_index_2 = index_2+"."
                            content_2 = argument.content
                            polarity_2 = argument.absolute_polarity_compute(Arguments)

                            if distance <3 :
                                # with relatedness
                                # relatedness_distance_set.append(
                                #         {"index_1": return_index_1, "content_1": content_1, "index_2": return_index_2, "content_2": content_2, "polarity_2": polarity_2,  "distance": distance,
                                #              "relatedness": relatedness})
                                relatedness_distance_set.append(
                                    {"index_1": return_index_1, "content_1": content_1, "index_2": return_index_2,
                                     "content_2": content_2, "polarity_2": polarity_2, "distance": distance,
                                     })
                                        # print("长短不一而且第一个不同的元素没有出现在短的列表中", relatedness_distance_set)




                        else:

                            for p in range(min(len(index_list), len(index_2_list))):
                                if index_list[p] != index_2_list[p]:
                                    # print(p)
                                    distance = len(index_list[p:]) + len(index_2_list[p:])
                                    relatedness = 1 / distance
                                    return_index_1 = index+"."
                                    return_index_2 = index_2+"."
                                    content_1 = self.content
                                    content_2 = argument.content
                                    polarity_2 = argument.absolute_polarity_compute(Arguments)

                                    if distance < 3:
                                        # with relatedness
                                        # relatedness_distance_set.append(
                                        #         {"index_1": return_index_1, "content_1": content_1, "index_2": return_index_2, "content_2": content_2, "polarity_2": polarity_2,  "distance": distance,
                                        #              "relatedness": relatedness})
                                        relatedness_distance_set.append(
                                            {"index_1": return_index_1, "content_1": content_1,
                                             "index_2": return_index_2,
                                             "content_2": content_2, "polarity_2": polarity_2, "distance": distance,
                                             })
                                        # print("长短不一但是第一个不同的元素出现在短的列表中", relatedness_distance_set)
                                        break


            return relatedness_distance_set

        # def cross_debate_relatedness_compute(self, Arguments):
        #     for argument in Arguments:
        #         index = self.index.strip(".")
        #         index_list = index.split(".")
        #         relatedness_distance_set = []
        #         for debate in debates:
        #             for argument in debates:
        #                 index_2 = argument.index.strip(".")
        #                 index_2_list = index_2.split(".")
        #                 distance = float('inf')
        #                 relatedness = 0

        def absolute_polarity_compute(self, Arguments):
            origin_index = self.index.strip()
            relative_polarity_value = self.relative_polarity_value
            origin_index_list = list(origin_index)
            result = relative_polarity_value
            current_index_list = origin_index_list[:]
            last_dot_position = self.find_last_dot(current_index_list)

            # print("functest", "index:", origin_index, "current_index_list", current_index_list,
            #       "relative_polarity_value:", relative_polarity_value)
            found_root_argument = 0

            while current_index_list.count('.') >= 1 and found_root_argument == 1:
                if current_index_list.count('.') >= 2:
                    for argument in Arguments:
                        # print(current_index_list.count('.'))
                        # 找到父节点


                        if (argument.index.strip() == "".join(current_index_list[:last_dot_position + 1]).strip()):
                            if (argument.relative_polarity != None):

                                current_index_list = list(argument.index.strip())
                                # print("argument.index", argument.index.strip())
                                # print("目前的极值", result, "父节点的极值", argument.relative_polarity_value)

                                # result = argument.relative_polarity_value * result
                                result = 0
                                # print("计算后的结果", result)
                                # print("current_index_list", current_index_list)
                                last_dot_position = self.find_last_dot(current_index_list)


                            elif (argument.relative_polarity == None):
                                current_index_list = list(argument.index.strip())
                                print("haha", argument.index)

                                #
                                found_root_argument = 1
                                return result
                elif current_index_list.count('.') == 1:
                    result = relative_polarity_value
                    # print(result)
                    return result

            return result
        def test(self, Arguments):
            num = []
            for argument in Arguments:
                num.append({"index_1": "return_index_1", "index_2": "return_index_2", "distance": "distance",
                                     "relatedness": "relatedness"})
            return num




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
            # for num, (index_of_argument_debate, line) in enumerate(enumerate(lines), start=1):

                match = re.match(r'(?m)^\b(?<!\d)(\d+(\.\d+)*)\.\s', line)
                if match and not re.match(r"^(Pro:|Con:)", line.replace(match.group(0), '').strip()):
                    self.debate_topic = line.replace(match.group(0), '').strip()
                    index = match.group(0).strip()
                    text_content = line.replace(match.group(0), '').strip()
                    relative_polarity = None
                    content = text_content.strip()
                    # relative_polarity_value = 1 if relative_polarity == "Pro" else -1
                    relative_polarity_value = 1


                    self.add_argument(index, content, relative_polarity, relative_polarity_value)
                elif match and re.match(r"^(Pro:|Con:)", line.replace(match.group(0), '').strip()):
                    # argument.relative_polarity_value = None
                    index = match.group(0)
                    text_content = line.replace(match.group(0), '').strip()
                    relative_polarity = text_content[:3].strip()
                    if relative_polarity == "Pro":

                        relative_polarity_value = 1

                    elif (relative_polarity == "Con"):
                        relative_polarity_value = -1

                    else:
                        raise ValueError
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


if __name__ == "__main__":
    # debates = load_debates_from_folder('/Users/fanzhe/Desktop/master_thesis/Data/kialo_debatetree_data/englishdebates')
    debates = load_debates_from_folder('/Users/fanzhe/Desktop/master_thesis/Data/kialo_debatetree_data/testsample_english')
    # debates = load_debates_from_folder('/home/users0/fanze/masterarbeit/MasterArbeit_test/MasterArbeit/testsample_english/')

    # debates = load_debates_from_folder('/home/users0/fanze/masterarbeit/englishdebates')
    start_time = time.time()


    argument_pair_num_list = []
    argument_pair_num_list_after_filtered = []
    for debate in tqdm(debates):
        print("topic:", debate.debate_topic)
        absolute_polarity_set = []
        csv_set = []





        # 添加绝对极值属性
        for argument in tqdm(debate.arguments):

            absolute_polarity = argument.absolute_polarity_compute(debate.arguments)
            absolute_polarity_set.append(absolute_polarity)

        # print(absolute_polarity_set,type(absolute_polarity_set))
        # for argument, absolute_polarity in zip(debate.arguments, absolute_polarity_set):
        debate_pair_num = 0
        for count1, (argument, absolute_polarity) in tqdm(enumerate(zip(debate.arguments, absolute_polarity_set), start=1)):


            argument.absolute_polarity = absolute_polarity

            argument.distance_relatedness_set = argument.distance_relatedness_compute(debate.arguments, count1)

            #count all pairs in a debate
            # debate_pair_num = debate_pair_num + count1
            #
            # csv_set.append(argument.distance_relatedness_set)


            print("index:", argument.index, "content:", argument.content, "relative polarity:", argument.relative_polarity, "relative polarity value:", argument.relative_polarity_value,"absolute polarity:", argument.absolute_polarity , "argumentpair num:", argument.distance_relatedness_set, "len(argument.distance_relatedness_set):", len(argument.distance_relatedness_set), len(debate.arguments),"", "")

            # index_2 = (item['index_2'] for item in argument.distance_relatedness_set)
            # print(index_2)

        # add to csv

            for item in tqdm(argument.distance_relatedness_set):
                item['polarity_1'] = argument.absolute_polarity
                item['polarity_consistency'] = 1 if item['polarity_1'] == item['polarity_2'] else -1
            csv_set.extend(argument.distance_relatedness_set)




        df = pandas.DataFrame(csv_set)
        # df['polarity_1'] = argument.absolute_polarity
        print("asdf", argument.absolute_polarity)

        df['polarity_consistency'] = df.apply(lambda row: 1 if row['polarity_1'] == row['polarity_2'] else -1, axis=1)
        filenum_index = argument.index.find(".")
        if filenum_index != -1:
            filenum = argument.index[:filenum_index]
        else:
            filenum = argument.index

        output_folder = "/Users/fanzhe/Desktop/master_thesis/Data/kialo_debatetree_data/csv_sample"
        # output_folder = '/home/users0/fanze/masterarbeit/csv'

        output_file = str(filenum) + ".csv"
        df.to_csv(f"{output_folder}/{output_file}", index= False)
        argument_pairs_num_in_the_debate_after_filtered = len(csv_set)

        print('pairs after filtered in the debate:', argument_pairs_num_in_the_debate_after_filtered)
        argument_pair_num_list_after_filtered.append(argument_pairs_num_in_the_debate_after_filtered)



        # print("debate_pair_num：", debate_pair_num)
        # argument_pair_num_list.append(debate_pair_num)
    # print("argument_pair_num_list:", argument_pair_num_list)
    # print("total argument pairs:", sum(argument_pair_num_list))
    print(argument_pair_num_list_after_filtered)
    print("total argument pairs after filter:", sum(argument_pair_num_list_after_filtered),len(argument_pair_num_list_after_filtered))
    end_time = time.time()
    total_time = end_time - start_time
    print(f"Total runtime: {total_time:.2f} seconds")



