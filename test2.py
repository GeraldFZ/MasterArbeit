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

    def relatedness_compute(self, Arguments):
        index = self.index.strip(".")
        index_list = index.split(".")
        distance = 0  # 默认值
        relatedness = 1

        for argument in Arguments:
            index_2 = argument.index.strip(".")
            index_2_list = index_2.split(".")
            if index_list == index_2_list:
                distance = 0
                relatedness = 1
            else:
                for i in range(min(len(index_list), len(index_2_list))):
                    if index_list[i] != index_2_list[i]:
                        print("haha", i)
                        print(index_list[i], index_2_list[i])

                        distance = len(index_list[i:]) + len(index_2_list[i:])
                        relatedness = 1 / distance
                        break
        return index_list, index_2_list, distance, relatedness


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
        print(argument.relatedness_compute(Arguments))



