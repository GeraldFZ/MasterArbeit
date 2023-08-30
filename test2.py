class Argument:
    def __init__(self, index, relative_polarity_value):
        self.index = index
        self.relative_polarity_value = relative_polarity_value

    def find_last_dot(self, lst):
        dot_positions = [i for i, char in enumerate(lst) if char == "."]
        if len(dot_positions) >= 2:
            last_dot_position = dot_positions[-2]
            return last_dot_position
        else:
            return None

    def absolute_polarity_compute(self, Arguments):
        origin_index = self.index
        relative_polarity_value = self.relative_polarity_value
        origin_index_list = list(origin_index)
        result = relative_polarity_value
        current_index_list = origin_index_list[:]
        last_dot_position = self.find_last_dot(current_index_list)

        print("index:", origin_index, "relative_polarity_value:", relative_polarity_value)



        while last_dot_position is not None:
            for argument in Arguments:
                if argument.index == "".join(current_index_list[:last_dot_position + 1]):
                    result = argument.relative_polarity_value * result
                    current_index_list = "".join(current_index_list[:last_dot_position + 1])
                    last_dot_position = self.find_last_dot(current_index_list)
                    break  # 找到匹配项后，跳出内部循环

        absolute_polarity = result
        return absolute_polarity
        print("absolute_polarity=", absolute_polarity)
if __name__ == '__main__':
    Arguments = [
        Argument("2.", 10),
        Argument("2.3.", -10),
        Argument("3.2.", -80),
        Argument("2.3.2.", -100),
        Argument("2.3.2.1.", -20),
        Argument("3.", -10),
    ]
    for argument in Arguments:
        print(argument.absolute_polarity_compute(Arguments))


