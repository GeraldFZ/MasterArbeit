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

# from sentence_transformers import SentenceTransformer, InputExample
# from torch.utils.data import DataLoader
#
# model = SentenceTransformer('distilbert-base-nli-mean-tokens')
# train_examples = [InputExample(texts=['My first sentence', 'My second sentence'], label=0.8),
#    InputExample(texts=['Another pair', 'Unrelated sentence'], label=0.3)]
# train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=2)
#
#
#
#
#

list = [272, 97, 217, 34, 153, 102, 161, 4801, 849, 176, 102, 110, 36, 640, 218, 609, 11024, 85, 105, 911, 640, 310, 402, 48, 54810, 45, 76, 105, 38063, 48, 344, 94, 266, 293, 147, 241, 4801, 8180, 344, 69, 11024, 306, 513, 310, 173, 59796, 986, 1499, 79962, 127, 986, 616, 89, 86, 10347, 689, 147, 639, 64, 306, 776, 229, 695, 161, 89, 133, 44, 911, 79962, 45, 689, 4906, 1499, 57]
# list = [5, 10121, 48, 104, 77, 47, 302, 120, 40, 150, 405, 40, 365, 509, 106, 24706, 678912, 849, 52, 3769, 102, 236, 3115, 240, 10808, 98, 10250, 69, 161, 34, 324, 79, 1864, 1216, 652, 513, 2915, 177, 263, 68, 4432, 12867, 2536, 2028, 8524, 71, 33211, 395, 158, 129, 2419, 396, 413, 184, 240, 2057, 3219, 306, 4934, 18727, 2396, 304, 123, 96, 368, 686, 99, 188, 57387, 71, 50, 3832, 174, 78, 95, 480, 217, 17596, 277, 101, 19920, 109, 2418, 157, 617, 835, 447, 49, 1288, 83, 737, 273, 20887, 128, 2988, 693, 419, 47, 10887, 40, 77, 132, 21101, 797, 4379, 3180, 651, 13751, 136, 232, 643, 534, 288, 160, 207, 423, 214, 10461, 2494, 9216, 495, 102, 698, 266, 71, 302, 5654, 210, 135, 187, 1061, 734, 92, 241, 7254, 125, 117, 38384, 357, 100, 59, 1220, 245783, 156, 80, 94, 1745, 477, 187, 38351, 698, 206, 113, 181, 389, 237, 106, 68, 1070, 249, 37615, 2015, 6257, 212, 6739, 344, 2953, 12112, 3306, 14046, 7753, 284, 8414, 441, 30, 8722, 172, 82, 125, 85, 228, 50733, 314, 19202, 12286, 11304, 94, 156, 21875, 1803, 259, 1748, 4827, 8220, 4255, 26273, 162, 248, 59, 36, 83088, 975, 116912, 48098, 1229, 26, 1929, 85, 418, 100277, 53, 290, 1536, 102814, 65, 353046, 199, 73, 920, 256, 5592, 8052, 9813, 76, 129, 93, 446, 54, 49, 70123, 360, 205, 38, 493, 1504, 1262, 764, 10347, 1407, 105, 26409, 214, 109, 15, 79962, 23032, 245, 337, 74, 11024, 2568, 228, 396, 24184, 16347, 85, 73, 490, 196, 65, 245, 136, 2653, 5294, 72, 19309, 127, 88, 434, 221, 4646, 46009, 779, 592, 81, 57, 523, 135, 4379, 8018, 78, 4066, 39546, 5678, 96, 102, 51, 36, 6102, 1420, 438, 188, 57, 13702, 76, 46, 402, 33513, 181]
# list = [5778, 105, 1035, 1596, 276, 153, 136, 3741, 780, 106491, 171, 78, 300, 78, 351, 2218671, 45, 253116, 136, 325, 406, 1953, 595, 253, 3570, 595, 861, 378, 91, 30135, 1225, 3828, 45, 6216, 178503, 91, 41041, 3240, 190, 561, 325, 253, 120, 171, 528, 31375, 4435731, 561, 253, 435, 7750, 5565, 171, 66, 820, 10731, 561, 1540, 595, 153, 820, 3570, 406, 66, 190, 4465, 27495, 136, 528, 2016, 142845, 528, 741, 78, 137026, 595, 496, 5356, 45, 7021, 154290, 699153, 57291, 1891, 13366, 351, 93528, 817281, 1081, 155961, 1653, 276, 34716, 5151, 820, 269745, 528, 2080, 146611, 231, 105, 20706, 19900, 1596, 595, 1131760, 131841, 378, 16836, 780, 78, 1540, 5050, 1653, 5671, 435, 4278, 90951, 1891, 8128, 136, 6441, 891780, 153, 1275, 465, 20301, 171, 210, 130816, 356590, 990, 325, 561, 7875, 174936, 561, 10731, 28920, 171, 946, 741, 4851, 3655, 496, 861, 703, 101025, 136, 30628, 2556, 1653, 63190, 120, 630, 2016, 1770, 496, 378, 351, 241860, 8385, 861, 351, 66, 946, 406, 66, 22578, 561, 325, 561, 1046181, 3003, 741, 1081, 276, 6105, 153, 253, 780, 253, 31375, 544446, 741, 66, 253, 5050, 1176, 666, 231, 561, 351, 300, 496, 91, 5565, 496, 7142310, 351, 276, 400960, 351, 630, 22366, 1326, 153, 2415, 435, 120, 5460, 351, 55, 378, 14196, 1128, 1891, 15051, 741, 24753, 136, 16471, 378, 231, 528, 58311, 2775, 55, 5995, 78, 210, 406, 496, 253, 193131, 66, 325, 528, 474825, 23220, 3486, 31626, 8128, 465, 528, 18336, 45, 946, 1225, 435, 1596, 1326, 39060, 594595, 5253, 57630, 561, 561, 14196, 190, 1378, 210, 7260, 1596, 276, 351, 861, 1485, 3081, 341551, 419986, 861, 171, 16471, 78, 4656, 253, 2850, 3741, 66, 45, 153, 202566, 36585, 351, 93528, 5565, 55, 12090, 231, 2346, 861, 496, 1225, 325, 153, 2415, 780, 861, 136, 561, 1770, 1176, 5356, 1596, 1378, 23871, 528, 3240, 120, 253, 136, 351, 820, 28920, 351, 528, 1081, 45, 17391, 250278, 171, 4005, 42195, 435, 31626, 946, 6903, 1711, 190, 946, 6903, 1830, 310078, 528, 2775, 190, 820, 1953, 14365, 105, 378, 78, 153, 630, 153, 1081, 1326, 210, 16471, 2556, 22578, 55, 276, 136, 43365, 351, 359128, 190, 129286, 561, 136, 4851, 496, 528, 49770, 561, 325, 1128, 1711, 231, 136, 1830, 325, 741, 5151, 171, 21736, 190, 171, 496, 2080, 2016, 190, 45, 351, 325, 7875, 496, 666, 5995, 1891, 132355, 861, 182106, 210, 418155, 8128, 7021, 13695, 55, 703, 276, 128271, 69751, 29890, 561, 298378, 1836486, 561, 741, 276, 703, 1431, 35511, 1540, 19503, 496, 210, 351, 120, 1035, 157641, 210, 75855, 3160, 465, 6441, 153, 253, 528, 40755, 1378, 595, 300, 5253, 37950, 300, 595, 1431, 946, 702705, 253, 65341, 105, 1953, 595, 9453, 300, 7260, 1128, 1275, 325, 171, 3570, 55, 1711, 39060, 105, 55945, 2556, 55945, 120, 300, 190, 66, 1035, 406, 105, 406, 990, 66, 946, 1275, 253, 210925, 6431491, 3403, 120, 16110, 231, 666, 16110, 903, 90525, 325, 60378, 210, 351, 66, 946, 190, 5356, 7021, 1653, 1431, 9730, 496, 666, 153, 31375, 61425, 21321, 6328, 36585, 190, 252405, 1225, 496, 378, 10878, 1596, 1081, 595, 820, 11476, 14028, 1275, 36046, 251695, 10585, 780, 351, 253, 1275, 1953, 190, 465, 1013176, 190, 78, 18528, 496, 190, 300, 1711, 741, 95266, 741, 190, 117855, 325, 14028, 378, 2415, 3240, 1540, 105, 5565, 171, 3003, 780, 263901, 300, 10011, 3828, 1035, 91, 60726, 78, 231, 325, 147153, 2415, 22366, 9870, 2016, 62128, 300, 741, 1830, 2211, 1035, 630, 595, 1431, 780, 96580, 6555, 49770, 1711, 325, 2278, 528, 231, 1081, 23436, 630, 325, 496, 4851, 3403, 231, 780, 44850, 325, 253, 242556, 990, 300, 153, 4656, 2220778, 630, 153, 253, 5253, 1953, 496, 357435, 1540, 561, 253, 561, 1225, 630, 231, 91, 4560, 741, 475800, 9730, 39060, 528, 32896, 1378, 17205, 125250, 18145, 83845, 45451, 990, 43071, 1326, 45, 60726, 406, 120, 300, 210, 903, 501501, 861, 146070, 70876, 103740, 153, 435, 174345, 5050, 741, 7750, 32896, 65703, 20301, 274911, 406, 903, 136, 45, 727821, 2850, 733866, 387640, 5253, 45, 8256, 190, 1176, 366796, 91, 741, 5565, 922761, 136, 4099816, 630, 171, 2850, 630, 23220, 33411, 73153, 153, 351, 210, 2346, 136, 105, 483636, 1128, 741, 55, 1596, 5671, 5565, 2346, 75855, 6555, 190, 227475, 666, 231, 36, 572985, 87571, 561, 1596, 153, 73153, 11325, 595, 903, 191890, 145530, 231, 153, 1378, 630, 136, 820, 325, 12246, 35245, 171, 152076, 351, 210, 1431, 780, 27261, 393828, 3160, 2701, 190, 136, 1891, 378, 19900, 41905, 171, 15931, 157641, 20100, 253, 253, 91, 45, 21321, 6903, 1431, 528, 78, 91378, 171, 120, 703, 188191, 561]
result = len(list)

print(result)


