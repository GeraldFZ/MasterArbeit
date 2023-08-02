# import pickle as pkl
#
# data = pkl.load(open('/Users/fanzhe/Desktop/master_thesis/Data/kialo/english/1622.pkl', 'rb'))
# print(data.node)
# # import pickle
# #
# #
# # with open("example.pkl", "rb") as f:
# #     obj = pickle.load(f)
# #
# #
# # print(obj)

file_path = "/Users/fanzhe/Desktop/master_thesis/Data/kialo_debatetree_data/discussions/sollte-deutschland-eine-allgemeine-dienstpflicht-einf%C3%BChren-17976.txt"
with open(file_path, "r") as file:
    file_content = file.read()
    print(file_content)
