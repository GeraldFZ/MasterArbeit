class Debate:
    def __init__(self, debate_topic):
        self.debate_topic = debate_topic
        self.arguments = []

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

        def distance_relatedness_compute(self, Arguments):
            index = self.index.strip(".")
            index_list = index.split(".")
            # distance = 0  # 默认值
            # relatedness = 1
            relatedness_distance_set = []

            for argument in Arguments:
                index_2 = argument.index.strip(".")
                index_2_list = index_2.split(".")
                if index_list == index_2_list:
                    # print("same", index_list, index_2_list)
                    # print("相同", index_list, index_2_list)

                    distance = 0
                    relatedness = 1
                    return_index_1 = index_list
                    return_index_2 = index_2_list
                    relatedness_distance_set.append(
                        {"index_1": return_index_1, "index_2": return_index_2, "distance": distance,
                         "relatedness": relatedness})



                # elif len(index_list) == len(index_2_list) and index_list != index_2_list:
                #     # print("长短一样但不相同", index_list, index_2_list)
                #
                #     for i in range(len(index_list)):
                #
                #         if index_list[i] != index_2_list[i]:
                #             # print(item1, item2)
                #
                #             distance = len(index_list[i:]) + len(index_2_list[i:])
                #             # print(index_list.index(item1),index_2_list.index(item2))
                #             relatedness = 1 / distance
                #             return_index_1 = index_list
                #             return_index_2 = index_2_list
                #             relatedness_distance_set.append(
                #                 {"index_1": return_index_1, "index_2": return_index_2, "distance": distance,
                #                  "relatedness": relatedness})


                elif len(index_list) != len(index_2_list):
                    long_list = max(index_list, index_2_list, key=len)
                    short_list = min(index_list, index_2_list, key=len)
                    if all(short_elem == long_elem for short_elem, long_elem in zip(short_list, long_list)):
                        # print("长短不一而且第一个不同的元素没有出现在短的列表中", index_list, index_2_list)
                        distance = len(long_list) - len(short_list)
                        # print(long_list,short_list)
                        relatedness = 1 / distance
                        return_index_1 = index_list
                        return_index_2 = index_2_list
                        relatedness_distance_set.append(
                            {"index_1": return_index_1, "index_2": return_index_2, "distance": distance,
                             "relatedness": relatedness})


                    else:
                        # print("长短不一但是第一个不同的元素出现在短的列表中", index_list, index_2_list, range(min(len(index_list), len(index_2_list)) - 1))

                        for p in range(min(len(index_list), len(index_2_list))):
                            if index_list[p] != index_2_list[p]:
                                # print(p)
                                distance = len(index_list[p:]) + len(index_2_list[p:])
                                relatedness = 1 / distance
                                return_index_1 = index_list
                                return_index_2 = index_2_list
                                relatedness_distance_set.append(
                                    {"index_1": return_index_1, "index_2": return_index_2, "distance": distance,
                                     "relatedness": relatedness})

            return relatedness_distance_set
        # def distance_relatedness_compute(self, Arguments):
        #     index = self.index.strip(".")
        #     index_list = index.split(".")
        #     relatedness_distance_set = []
        #
        #     for argument in Arguments:
        #         index_2 = argument.index.strip(".")
        #         index_2_list = index_2.split(".")
        #         if index_list == index_2_list:
        #             distance = 0
        #             relatedness = 1
        #             return_index_1 = index_list
        #             return_index_2 = index_2_list
        #             relatedness_distance_set.append(
        #                 {"index_1": return_index_1, "index_2": return_index_2, "distance": distance,
        #                  "relatedness": relatedness})
        #
        #         elif len(index_list) == len(index_2_list) and index_list != index_2_list:
        #             for i in range(len(index_list)):
        #                 if index_list[i] != index_2_list[i]:
        #                     distance = len(index_list[i:]) + len(index_2_list[i:])
        #                     relatedness = 1 / distance
        #                     return_index_1 = index_list
        #                     return_index_2 = index_2_list
        #                     relatedness_distance_set.append(
        #                         {"index_1": return_index_1, "index_2": return_index_2, "distance": distance,
        #                          "relatedness": relatedness})
        #
        #         elif len(index_list) != len(index_2_list):
        #             long_list = max(index_list, index_2_list, key=len)
        #             short_list = min(index_list, index_2_list, key=len)
        #             if all(short_elem == long_elem for short_elem, long_elem in zip(short_list, long_list)):
        #                 distance = len(long_list) - len(short_list)
        #                 relatedness = 1 / distance
        #                 return_index_1 = index_list
        #                 return_index_2 = index_2_list
        #                 relatedness_distance_set.append(
        #                     {"index_1": return_index_1, "index_2": return_index_2, "distance": distance,
        #                      "relatedness": relatedness})
        #
        #             else:
        #                 for p in range(min(len(index_list), len(index_2_list))):
        #                     if index_list[p] != index_2_list[p]:
        #                         distance = len(index_list[p:]) + len(index_2_list[p:])
        #                         relatedness = 1 / distance
        #                         return_index_1 = index_list
        #                         return_index_2 = index_2_list
        #                         relatedness_distance_set.append(
        #                             {"index_1": return_index_1, "index_2": return_index_2, "distance": distance,
        #                              "relatedness": relatedness})
        #
        #     return relatedness_distance_set



if __name__ == '__main__':
    debate_instance = Debate("Is Technology Beneficial?")

    # 向辩论实例添加不同的Argument实例作为辩论的不同论点
    debate_instance.arguments.append(debate_instance.Argument("2.", 0.8))
    debate_instance.arguments.append(debate_instance.Argument("2.2.3.", -0.5))
    debate_instance.arguments.append(debate_instance.Argument("2.5.3.6.", 0.2))
    debate_instance.arguments.append(debate_instance.Argument("2.2.5.", 0.2))
    debate_instance.arguments.append(debate_instance.Argument("2.2.6.3.2.", 0.2))

    for argument in debate_instance.arguments:
        # print( "output",argument.relatedness_compute(Arguments))
        # relatedness_distance_set.append(argument.relatedness_compute(Arguments))
        argument.relatedness_distance_set = argument.distance_relatedness_compute(debate_instance.arguments)

        # 在循环结束后打印长度
    for argument in debate_instance.arguments:
        print(len(debate_instance.arguments), len(argument.relatedness_distance_set))
