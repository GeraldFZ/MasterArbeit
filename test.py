Arguments = [
    {"index": "2.", "relative_polarity_value": 1},
    {"index": "2.3.", "relative_polarity_value": 1},
    {"index": "3.", "relative_polarity_value": 1},
    {"index": "2.3.2.", "relative_polarity_value": -1}
]

#
# def absolute_polarity_compute(index, relative_polarity_value):
#     # list_index = index.split(".")  # Split the index into a list of parts
#
#     return absolute_polarity


if __name__ == '__main__':
    for argument in Arguments:
        index = argument["index"]
        relative_polarity_value = argument["relative_polarity_value"]
        print(index, relative_polarity_value)
        list_index = list(index)
        # print(list_index)

        result = relative_polarity_value

        if list_index.count(".") != 1:
            print(list_index.count(".") != 1)
            for argument in Arguments:
                print(argument["index"], "".join(list_index[:-2]))
                # 将下边的判断条件右边改成找到上一个"."的位置而不是简单地减二
                if argument["index"] == "".join(list_index[:-2]):
                    result = argument["relative_polarity_value"] * result
                    list_index = "".join(list_index[:-2])
                    # print(list_index)
            absolute_polarity = result
        else:
            absolute_polarity = relative_polarity_value


        print("absolute_polarity=", absolute_polarity)

        # absolute_polarity_compute(index, relative_polarity_value)
        # print("absolute_polarity=", absolute_polarity_compute(index, relative_polarity_value))

# def reverse_index_numbering(lst):
#     dot_positions = [i for i, char in enumerate(lst) if char == "."]
#     dot_positions.reverse()
#     numbered_positions = {pos: i+1 for i, pos in enumerate(dot_positions)}
#     return numbered_positions
#
# # 测试列表
# test_list = ["a", "b", ".", "c", ".", "d", "e", ".", "f"]
# numbered_positions = reverse_index_numbering(test_list)
# print(numbered_positions)