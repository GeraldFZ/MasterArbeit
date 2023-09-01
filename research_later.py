# def absolute_polarity_compute(self, Arguments):
#     origin_index = self.index
#     relative_polarity_value = self.relative_polarity_value
#     origin_index_list = list(origin_index)
#     result = relative_polarity_value
#     current_index_list = origin_index_list[:]
#     last_dot_position = self.find_last_dot(current_index_list)
#
#     print("index:", origin_index, "relative_polarity_value:", relative_polarity_value)
#
#     while last_dot_position is not None:
#         for argument in Arguments:
#             if argument.index == "".join(current_index_list[:last_dot_position + 1]):
#                 result = argument.relative_polarity_value * result
#                 current_index_list = "".join(current_index_list[:last_dot_position + 1])
#                 last_dot_position = self.find_last_dot(current_index_list)
#                 break  # 找到匹配项后，跳出内部循环
#
#     absolute_polarity = result
#     return absolute_polarity
#     print("absolute_polarity=", absolute_polarity)
# return result
# def absolute_polarity_compute(self, Arguments):
#     origin_index = self.index.strip()
#     relative_polarity_value = self.relative_polarity_value
#     origin_index_list = list(origin_index)
#     result = relative_polarity_value
#     current_index_list = origin_index_list[:]
#     last_dot_position = self.find_last_dot(current_index_list)
#
#     print("functest", "index:", origin_index, "current_index_list", current_index_list,
#           "relative_polarity_value:", relative_polarity_value)
#
#     while current_index_list.count('.') >= 1:
#         if current_index_list.count('.') >= 2:
#             for argument in Arguments:
#                 print(current_index_list.count('.'))
#                 # 找到父节点
#                 if argument.index.strip() == "".join(current_index_list[:last_dot_position + 1]).strip():
#                     current_index_list = list(argument.index.strip())
#                     print("argument.index", argument.index.strip())
#                     print("目前的极值", result, "父节点的极值", argument.relative_polarity_value)
#
#                     result = argument.relative_polarity_value * result
#                     print("计算后的结果", result)
#                     print("current_index_list", current_index_list)
#                     last_dot_position = self.find_last_dot(current_index_list)
#                     if self.find_last_dot(current_index_list) is None:
#                         return result
#             break
#                     # current_index_list = current_index_list[:last_dot_position + 1]
#                     # print("current_index_list", current_index_list)
#
#                         # break  # 找到匹配项后，跳出内部循环
#         elif current_index_list.count('.') == 1:
#             result = relative_polarity_value
#             # print(result)
#             return result
#
#         # break  # 终止外部循环




# if __name__ == "__main__":
#     folder_path = "your_folder_path"  # 替换为实际文件夹路径
#     debates = load_debates_from_folder('/Users/fanzhe/Desktop/master_thesis/Data/kialo_debatetree_data/dataprocesstest')
#
#     # 创建索引和相对极性值的映射
#     relative_polarity_mapping = {}
#     for debate in debates:
#         for argument in debate.arguments:
#             relative_polarity_mapping[argument.index] = argument.relative_polarity_value
#
#     for debate in debates:
#         print("topic:", debate.debate_topic)
#         for argument in debate.arguments:
#             absolute_polarity = argument.absolute_polarity_compute(relative_polarity_mapping)
#             print("index:", argument.index, "content:", argument.content, "relative polarity:", argument.relative_polarity, "relative polarity value:", argument.relative_polarity_value, "absolute polarity:", absolute_polarity)

        # def absolute_polarity_compute(self, relative_polarity_mapping):
        #     result = self.relative_polarity_value
        #
        #     current_index_list = list(self.index)
        #     last_dot_position = self.find_last_dot(current_index_list)
        #
        #     while last_dot_position is not None:
        #         index_key = "".join(current_index_list[:last_dot_position + 1])
        #         if index_key in relative_polarity_mapping:
        #             result *= relative_polarity_mapping[index_key]
        #
        #         current_index_list = current_index_list[:last_dot_position+1]
        #         last_dot_position = self.find_last_dot(current_index_list)
        #
        #     return result
