# # Arguments = [
# #     {"index": "2.", "relative_polarity_value": 10},
# #     {"index": "2.3.", "relative_polarity_value": -10},
# #     {"index": "3.2.", "relative_polarity_value": -80},
# #     {"index": "2.3.2.", "relative_polarity_value": -100},
# # # {"index": "2.3.2.16.", "relative_polarity_value": -10},
# # {"index": "2.3.2.1.", "relative_polarity_value": -20},
# # # {"index": "2.3.2.16.200.", "relative_polarity_value": -20},
# # {"index": "3.", "relative_polarity_value": -10},
# #
# #
# # ]
# #
# # def find_last_dot(list):
# #     dot_positions = [i for i, char in enumerate(list) if char == "."]
# #     if len(dot_positions) >= 2:
# #         last_dot_position = dot_positions[-2]
# #         return last_dot_position
# #     else:
# #         return None
# #
# # if __name__ == '__main__':
# #     absolute_polarity = None
# #
# #     for argument in Arguments:
# #         origin_index = argument["index"]
# #         relative_polarity_value = argument["relative_polarity_value"]
# #         origin_index_list = list(origin_index)
# #
# #         print("index:", origin_index, "relative_polarity_value:", relative_polarity_value)
# #
# #         result = relative_polarity_value
# #         last_dot_position = find_last_dot(origin_index)
# #         print(last_dot_position)
# #         if origin_index_list.count(".") >= 2:
# #             current_index_list = origin_index_list
# #             while current_index_list.count(".") >= 2:
# #
# #                 for argument in Arguments:
# #                     if last_dot_position is not None:
# #
# #                         current_index_list = "".join(current_index_list[:last_dot_position + 1])
# #                     if last_dot_position is not None:
# #
# #                         if argument["index"] == current_index_list:
# #                             result = argument["relative_polarity_value"] * result
# #                             # if last_dot_position is not None:
# #
# #                             last_dot_position = find_last_dot(current_index_list)
# #                         absolute_polarity = result
# #         else:
# #             absolute_polarity = relative_polarity_value
# #
# #
# #
# #
# #         print("absolute_polarity=", absolute_polarity)
# #
#
#
# list1 = [1, 2, 3, 4, 6]
# list2 = [1, 2, 3]
#
# for i, (item1, item2) in enumerate(zip(list1, list2)):
#     if item1 != item2:
#         print(f"The first difference is at index {i}: {item1} != {item2}")
#         break
# else:
#     print("Same")
#

from sentence_transformers import SentenceTransformer, InputExample
from torch.utils.data import DataLoader

model = SentenceTransformer('distilbert-base-nli-mean-tokens')
train_examples = [InputExample(texts=['My first sentence', 'My second sentence'], label=0.8),
   InputExample(texts=['Another pair', 'Unrelated sentence'], label=0.3)]
train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=2)












