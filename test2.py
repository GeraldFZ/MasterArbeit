def find_last_dot(lst):
    dot_positions = [i for i, char in enumerate(lst) if char == "."]
    if len(dot_positions) >= 2:
        last_dot_position = dot_positions[-2]
        return last_dot_position
    else:
        return None


if __name__ == '__main__':
    Arguments = [
        {"index": "2.", "relative_polarity_value": 10},
        {"index": "2.3.", "relative_polarity_value": -10},
        {"index": "2.3.2.16.200.", "relative_polarity_value": -20},

        {"index": "3.2.", "relative_polarity_value": -80},
        {"index": "2.3.2.", "relative_polarity_value": -100},
        {"index": "2.3.2.16.", "relative_polarity_value": -10},
        {"index": "2.3.2.1.", "relative_polarity_value": -20},
        {"index": "3.", "relative_polarity_value": -10},

    ]

    for argument in Arguments:
        origin_index = argument["index"]
        relative_polarity_value = argument["relative_polarity_value"]
        origin_index_list = list(origin_index)

        print("index:", origin_index, "relative_polarity_value:", relative_polarity_value)

        result = relative_polarity_value
        current_index_list = origin_index_list[:]
        last_dot_position = find_last_dot(current_index_list)

        while last_dot_position is not None:
            for argument in Arguments:
                if argument["index"] == "".join(current_index_list[:last_dot_position + 1]):
                    result = argument["relative_polarity_value"] * result
                    current_index_list = "".join(current_index_list[:last_dot_position + 1])
                    last_dot_position = find_last_dot(current_index_list)
                    break  # 找到匹配项后，跳出内部循环

        absolute_polarity = result
        print("absolute_polarity=", absolute_polarity)
