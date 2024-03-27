from sentence_transformers.evaluation import LabelAccuracyEvaluator

from sentence_transformers import SentenceTransformer,  InputExample, losses, models
from transformers import AutoTokenizer
import pandas as pd
import os
from torch.utils.data import DataLoader
import logging
import math
from datetime import datetime
from sklearn.model_selection import train_test_split
import random
import  re
import  yaml
import torch
import numpy as np

# model_name = SentenceTransformer('all-mpnet-base-v2')
model_name = 'sentence-transformers/all-MiniLM-L12-v2'



train_batch_size = 32
num_epochs = 10
word_embedding_model = models.Transformer(model_name)

# Apply mean pooling to get one fixed sized sentence vector
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
                               pooling_mode_mean_tokens=True,
                               pooling_mode_cls_token=False,
                               pooling_mode_max_tokens=False)



model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

train_batch_size = int(input("please input train_batch_size "))

num_epochs = int(input("please input num_epochs "))

distance_limit = int(input("please input max distance "))
csv_pair_size_limit = int(input("please input csv pair size limit "))
split_method_index = int(input("please input split method index(1: collect all pairs and then split, 2: split csv files first then collect pairs) "))
random_seed_num = int(input("please input random seed number "))
CUDA_num = int(input("please input CUDA number "))
device = torch.device(f"cuda:{CUDA_num}")
random.seed(random_seed_num)

logging.info("Read STSbenchmark train dataset")
# Apply mean pooling to get one fixed sized sentence vector


tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")


# # debates = load_debates_from_folder('/Users/fanzhe/Desktop/master_thesis/Data/kialo_debatetree_data/englishdebates')
# debates_path = '/Users/fanzhe/Desktop/master_thesis/Data/kialo_debatetree_data/csv_sample'

# # debates = load_debates_from_folder('/home/users0/fanze/masterarbeit/englishdebates')
# debates_path = '/mount/studenten5/projects/fanze/masterarbeit_data/csv_testmodel'
debates_path = '/mount/studenten5/projects/fanze/masterarbeit_data/csv_nofilter'

# debates_path = '/mount/studenten5/projects/fanze/masterarbeit_data/csv_nofilter'



# model_save_path = '/Users/fanzhe/Desktop/master_thesis/Data/model_ouput/training_stsbenchmark_'+model_name.replace("/", "-")+'-'+datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
model_save_path = '/mount/studenten5/projects/fanze/masterarbeit_data/model_output_record/training_polarity_'+model_name.replace("/", "-")+'-'+datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# 获取文件夹中的所有文件
all_files = os.listdir(debates_path)

# 筛选出CSV文件
csv_files = [file for file in all_files if file.endswith('.csv')]
def generate_yaml(model_name):

    config = {

        "training": {
            "train_batch_size": train_batch_size,
            "num_epochs": num_epochs,
            "evaluation_steps": 1000,
            "split_method": split_method_index,
            "max_pairs_size": csv_pair_size_limit,
            "max_distance": distance_limit,
            "train_ratio": 0.8,
            "dev_ratio" : 0.1,
            "test_ratio" : 0.1,
            "loss_function": "losses.CosineSimilarityLoss(model=model)"
        }
    }

    with open(model_save_path+'/model_config.yaml', 'w') as file:
        yaml.dump(config, file, default_flow_style=False)

def skip_argument(text):
    # 定义正则表达式
    pattern = r"-> See \d+(\.\d+)+\."

    # 使用re.match进行匹配
    return bool(re.match(pattern, text))

# 逐个读取CSV文件
# for file in csv_files:
#     file_path = os.path.join(debates_path, file)
#     # 读取CSV文件
#     df = pd.read_csv(file_path)
#     # print(file_path)
#
#     # 按行处理数据
#     files_has_0_distance = []
#     for index, row in df.iterrows():
#         if float(row['distance']) != 0 and not skip_argument(row['content_1']) and not skip_argument(row['content_2']):
#             if float(row['polarity_consistency']) == 1:
#
#                 score = 1 # Normalize score to range 0 ... 1
#                 inp_example = InputExample(texts=[row['content_1'], row['content_2']], label=score)
#                 # print(score, row['content_1'], row['content_2'])
#                 samples.append(inp_example)
#
#             if float(row['polarity_consistency']) == -1:
#                 if float(row['polarity_1']) == 0:
#                     # score = 1
#                     # inp_example = InputExample(texts=[row['content_1'], row['content_2']], label=score)
#                     pass
#
#                 elif float(row['polarity_1']) != 0:
#                     score = 0
#                     inp_example = InputExample(texts=[row['content_1'], row['content_2']], label=score)
#
#                     samples.append(inp_example)
#         elif float(row['distance']) == 0:
#             files_has_0_distance.append(file_path)
#
#             print(file_path, row['distance'])
def split_method_1(max_pairs_size, max_distance):
    random.seed(random_seed_num)
    torch.manual_seed(random_seed_num)
    np.random.seed(random_seed_num)

    all_files = os.listdir(debates_path)
    # csv_files = [file for file in all_files if file.endswith('.csv') and  (len(pd.read_csv(os.path.join(debates_path, file)))) < max_pairs_size  ]
    less_than_limit_files = []
    over_limit_files = []
    for file in all_files:
        if file.endswith('.csv'):
            if (len(pd.read_csv(os.path.join(debates_path, file)))) < max_pairs_size:
                less_than_limit_files.append(file)
            elif (len(pd.read_csv(os.path.join(debates_path, file)))) >= max_pairs_size:
                over_limit_files.append(file)
    random.shuffle(over_limit_files)
    random.shuffle(less_than_limit_files)

    # shuffled_csv_files = csv_files
    samples = []

    # 逐个读取CSV文件
    content_1_list = []

    for file in less_than_limit_files:

        file_path = os.path.join(debates_path, file)
        # 读取CSV文件
        df = pd.read_csv(file_path)

        number_of_pairs = len(df)
        print(file, number_of_pairs)
        print((len(pd.read_csv(os.path.join(debates_path, file)))))

        # 按行处理数据
        files_has_0_distance = []
        for index, row in df.iterrows():
            if float(row['distance']) != 0 and not skip_argument(row['content_1']) and not skip_argument(row['content_2']) and float(row['distance']) <= max_distance:

                if float(row['polarity_consistency']) == 1:

                    score = 1 # Normalize score to range 0 ... 1
                    inp_example = InputExample(texts=[row['content_1'], row['content_2']], label=score)
                    # print(score, row['content_1'], row['content_2'])
                    samples.append(inp_example)

                if float(row['polarity_consistency']) == -1:
                    if float(row['polarity_1']) == 0:
                        # score = 1
                        # inp_example = InputExample(texts=[row['content_1'], row['content_2']], label=score)
                        pass

                    elif float(row['polarity_1']) != 0:
                        score = 0
                        inp_example = InputExample(texts=[row['content_1'], row['content_2']], label=score)

                        samples.append(inp_example)

            elif float(row['distance']) == 0:
                files_has_0_distance.append(file_path)

                print(file_path, row['distance'])
    for file in over_limit_files:

        file_path = os.path.join(debates_path, file)
        # 读取CSV文件
        df = pd.read_csv(file_path)
        sampled_df = df.sample(n=max_pairs_size, random_state=random_seed_num)

        number_of_pairs = len(sampled_df)
        print(file, number_of_pairs)
        print(len(sampled_df))

        # 按行处理数据
        files_has_0_distance = []
        for index, row in sampled_df.iterrows():
            if float(row['distance']) != 0 and not skip_argument(row['content_1']) and not skip_argument(
                    row['content_2']) and float(row['distance']) <= max_distance:

                if float(row['polarity_consistency']) == 1:
                    score = 1  # Normalize score to range 0 ... 1
                    inp_example = InputExample(texts=[row['content_1'], row['content_2']], label=score)
                    # print(score, row['content_1'], row['content_2'])
                    samples.append(inp_example)

                if float(row['polarity_consistency']) == -1:
                    if float(row['polarity_1']) == 0:
                        # score = 1
                        # inp_example = InputExample(texts=[row['content_1'], row['content_2']], label=score)
                        pass

                    elif float(row['polarity_1']) != 0:
                        score = 0
                        inp_example = InputExample(texts=[row['content_1'], row['content_2']], label=score)

                        samples.append(inp_example)

            elif float(row['distance']) == 0:
                files_has_0_distance.append(file_path)

                print(file_path, row['distance'])
    # for content_1 in content_1_list:
    #     # print("testhahaha", content_1,type(content_1))
    #     rest_of_contents = [f for f in content_1_list if f["file_index"] != content_1["file_index"]]
    #     random_negative_arguments = []
    #     while len(random_negative_arguments) < negative_sample_num:
    #         random_index_content = random.choice(rest_of_contents)
    #         random_content = random_index_content["content"]
    #         # print(random_content)
    #         if random_content not in random_negative_arguments:
    #             random_negative_arguments.append(random_content)
    #
    #     for negative_argument in random_negative_arguments:
    #         # print("test", content_1, negative_argument, type(negative_argument))
    #         neg_inp_example = InputExample(texts=[content_1["content"], negative_argument], label=0.0)
    #         samples.append(neg_inp_example)
    # print("shuffle seed test, negative", random_negative_arguments[:10])

    file_name = "files_has_0_distance.txt"

    # 使用 'with' 语句打开文件进行写入，确保文件最后会被正确关闭
    with open(file_name, "w") as file:
        # 遍历列表，写入每一行
        for line in files_has_0_distance:
            file.write(line + "\n")  # "\n" 是换行符

    # print(samples, type(samples))

    random.shuffle(samples)
    shuffled_samples = samples
    sample_collection = shuffled_samples
    print("shuffle seed test", shuffled_samples[:10])

    train_ratio = 0.8
    dev_ratio = 0.1
    test_ratio = 0.1

    train_data, temp_data = train_test_split(sample_collection, test_size=(1 - train_ratio), shuffle=True,
                                             random_state=random_seed_num)
    print("shuffle seed test, trainset", train_data[:10])

    dev_data, test_data = train_test_split(temp_data, test_size=0.5, shuffle=True, random_state=random_seed_num)
    print("shuffle seed test, trainset", test_data[:10])

    # print(type(dev_data[0]))

    train_dataloader = DataLoader(train_data, shuffle=True, batch_size=train_batch_size)
    dev_dataloader = DataLoader(dev_data, shuffle=True, batch_size=train_batch_size)
    test_dataloader = DataLoader(test_data, shuffle=True, batch_size=train_batch_size)
    print("Number of training examples:", len(train_dataloader.dataset))
    print("Number of dev examples:", len(dev_dataloader.dataset))
    print("Number of test examples:", len(test_dataloader.dataset))
    return train_dataloader, dev_dataloader, test_dataloader, train_data, dev_data, test_data


# Split the csv files first method2
def split_method_2(max_pairs_size, max_distance):
    random.seed(random_seed_num)
    torch.manual_seed(random_seed_num)
    all_files = os.listdir(debates_path)
    # csv_files = [file for file in all_files if file.endswith('.csv') and (len(pd.read_csv(os.path.join(debates_path, file)))) < max_pairs_size]
    less_than_limit_files = []
    over_limit_files = []
    for file in all_files:
        if file.endswith('.csv'):
            if (len(pd.read_csv(os.path.join(debates_path, file)))) < max_pairs_size:
                less_than_limit_files.append(file)
            elif (len(pd.read_csv(os.path.join(debates_path, file)))) >= max_pairs_size:
                over_limit_files.append(file)
    random.shuffle(over_limit_files)
    random.shuffle(less_than_limit_files)
    csv_files_after_filter = over_limit_files + less_than_limit_files
    random.shuffle(csv_files_after_filter)

    train_ratio = 0.8
    dev_ratio = 0.1
    test_ratio = 0.1

    # 确保所有比例加起来等于1
    assert train_ratio + dev_ratio + test_ratio == 1, "Ratios must sum up to 1."

    # Calculate the number of files for train, dev, and test sets
    train_files_count = int(train_ratio * len(csv_files_after_filter))
    dev_files_count = int(dev_ratio * len(csv_files_after_filter))
    test_files_count = int(test_ratio * len(csv_files_after_filter))

    # Make sure the total count does not exceed the number of available files
    assert train_files_count + dev_files_count + test_files_count <= len(
        csv_files_after_filter), "File counts exceed total."

    # Split the file paths into training, development, and testing sets
    train_files = csv_files_after_filter[:train_files_count]
    dev_files = csv_files_after_filter[train_files_count:train_files_count + dev_files_count]
    test_files = csv_files_after_filter[train_files_count + dev_files_count:]
    print("train files lenth", len(train_files))
    print("dev files lenth", len(dev_files))
    print("test files lenth", len(test_files))

    train_data = []
    dev_data = []
    test_data = []
    files_has_0_distance = []

    for file in train_files:
        train_content_1_list = []
        file_path = os.path.join(debates_path, file)
        # read csv
        if file in less_than_limit_files:
            df = pd.read_csv(file_path)
        elif file in over_limit_files:
            pre_df = pd.read_csv(file_path)
            df = pre_df.sample(n=max_pairs_size, random_state=random_seed_num)

        print(file_path)

        # proces data by row
        for index, row in df.iterrows():
            if float(row['distance']) != 0 and not skip_argument(row['content_1']) and not skip_argument(
                    row['content_2']) and float(row['distance']) <= max_distance:
                if float(row['polarity_consistency']) == 1:
                    score = 1  # Normalize score to range 0 ... 1
                    inp_example = InputExample(texts=[row['content_1'], row['content_2']], label=score)
                    # print(score, row['content_1'], row['content_2'])
                    train_data.append(inp_example)

                if float(row['polarity_consistency']) == -1:
                    if float(row['polarity_1']) == 0:
                        # score = 1
                        # inp_example = InputExample(texts=[row['content_1'], row['content_2']], label=score)
                        pass

                    elif float(row['polarity_1']) != 0:
                        score = 0
                        inp_example = InputExample(texts=[row['content_1'], row['content_2']], label=score)

                        train_data.append(inp_example)

                # train_data.append(inp_example)
                if row['content_1'] not in train_content_1_list:
                    train_content_1_list.append(row['content_1'])
            elif float(row['distance']) == 0:
                files_has_0_distance.append(file_path)

                print(file_path, row['distance'])
    #     for content_1 in train_content_1_list:
    #         rest_of_csv = [f for f in csv_files_after_filter if f != file]
    #         random_negative_arguments = []
    #         while len(random_negative_arguments) < negative_sample_num:
    #             random_file = random.choice(rest_of_csv)
    #             random_file_path = os.path.join(debates_path, random_file)
    #             try:
    #
    #                 df_random_file = pd.read_csv(random_file_path)
    #                 random_value = df_random_file['content_1'].sample().iloc[0]
    #                 if random_value not in random_negative_arguments:
    #                     random_negative_arguments.append(random_value)
    #             except FileNotFoundError:
    #                 print(f"File not found: {random_file_path}")
    #
    #         for negative_argument in random_negative_arguments:
    #             neg_inp_example = InputExample(texts=[content_1, negative_argument], label=0.0)
    #             train_data.append(neg_inp_example)
    # print("shuffle seed test, negative", random_negative_arguments[:10])

    for file in dev_files:
        dev_content_1_list = []

        file_path = os.path.join(debates_path, file)
        # 读取CSV文件
        if file in less_than_limit_files:
            df = pd.read_csv(file_path)
        elif file in over_limit_files:
            pre_df = pd.read_csv(file_path)
            df = pre_df.sample(n=max_pairs_size, random_state=random_seed_num)
        # print(file_path)

        # 按行处理数据
        for index, row in df.iterrows():
            if float(row['distance']) != 0 and not skip_argument(row['content_1']) and not skip_argument(
                    row['content_2']) and float(row['distance']) <= max_distance:
                if float(row['polarity_consistency']) == 1:
                    score = 1  # Normalize score to range 0 ... 1
                    inp_example = InputExample(texts=[row['content_1'], row['content_2']], label=score)
                    # print(score, row['content_1'], row['content_2'])
                    dev_data.append(inp_example)

                if float(row['polarity_consistency']) == -1:
                    if float(row['polarity_1']) == 0:
                        # score = 1
                        # inp_example = InputExample(texts=[row['content_1'], row['content_2']], label=score)
                        pass

                    elif float(row['polarity_1']) != 0:
                        score = 0
                        inp_example = InputExample(texts=[row['content_1'], row['content_2']], label=score)

                        dev_data.append(inp_example)
                # print(score, row['content_1'], row['content_2'])

                dev_data.append(inp_example)
                if row['content_1'] not in dev_content_1_list:
                    dev_content_1_list.append(row['content_1'])
            elif float(row['distance']) == 0:
                files_has_0_distance.append(file_path)

                print(file_path, row['distance'])

        # for content_1 in dev_content_1_list:
        #     rest_of_csv = [f for f in csv_files_after_filter if f != file]
        #     random_negative_arguments = []
        #     while len(random_negative_arguments) < negative_sample_num:
        #         random_file = random.choice(rest_of_csv)
        #         random_file_path = os.path.join(debates_path, random_file)
        #         try:
        #
        #             df_random_file = pd.read_csv(random_file_path)
        #             random_value = df_random_file['content_1'].sample().iloc[0]
        #             if random_value not in random_negative_arguments:
        #                 random_negative_arguments.append(random_value)
        #         except FileNotFoundError:
        #             print(f"File not found: {random_file_path}")
        #
        #     for negative_argument in random_negative_arguments:
        #         neg_inp_example = InputExample(texts=[content_1, negative_argument], label=0.0)
        #         dev_data.append(neg_inp_example)
    for file in test_files:
        test_content_1_list = []

        file_path = os.path.join(debates_path, file)
        # 读取CSV文件
        if file in less_than_limit_files:
            df = pd.read_csv(file_path)
        elif file in over_limit_files:
            pre_df = pd.read_csv(file_path)
            df = pre_df.sample(n=max_pairs_size, random_state=random_seed_num)
        # print(file_path)

        # process the data by row
        for index, row in df.iterrows():
            if float(row['distance']) != 0 and not skip_argument(row['content_1']) and not skip_argument(
                    row['content_2']) and float(row['distance']) <= max_distance:
                if float(row['polarity_consistency']) == 1:
                    score = 1  # Normalize score to range 0 ... 1
                    inp_example = InputExample(texts=[row['content_1'], row['content_2']], label=score)
                    # print(score, row['content_1'], row['content_2'])
                    test_data.append(inp_example)

                if float(row['polarity_consistency']) == -1:
                    if float(row['polarity_1']) == 0:
                        # score = 1
                        # inp_example = InputExample(texts=[row['content_1'], row['content_2']], label=score)
                        pass

                    elif float(row['polarity_1']) != 0:
                        score = 0
                        inp_example = InputExample(texts=[row['content_1'], row['content_2']], label=score)

                        test_data.append(inp_example)
                # print(score, row['content_1'], row['content_2'])

                dev_data.append(inp_example)

                if row['content_1'] not in test_content_1_list:
                    test_content_1_list.append(row['content_1'])
            elif float(row['distance']) == 0:
                files_has_0_distance.append(file_path)

                print(file_path, row['distance'])
        # for content_1 in test_content_1_list:
        #     rest_of_csv = [f for f in csv_files_after_filter if f != file]
        #     random_negative_arguments = []
        #     while len(random_negative_arguments) < negative_sample_num:
        #         random_file = random.choice(rest_of_csv)
        #         random_file_path = os.path.join(debates_path, random_file)
        #         try:
        #
        #             df_random_file = pd.read_csv(random_file_path)
        #             random_value = df_random_file['content_1'].sample().iloc[0]
        #             if random_value not in random_negative_arguments:
        #                 random_negative_arguments.append(random_value)
        #         except FileNotFoundError:
        #             print(f"File not found: {random_file_path}")
        #
        #     for negative_argument in random_negative_arguments:
        #         neg_inp_example = InputExample(texts=[content_1, negative_argument], label=0.0)
        #         test_data.append(neg_inp_example)

    file_name = "files_has_0_distance.txt"

    # 使用 'with' 语句打开文件进行写入，确保文件最后会被正确关闭
    with open(file_name, "w") as file:
        # 遍历列表，写入每一行
        for line in files_has_0_distance:
            file.write(line + "\n")  # "\n" 是换行符
    print("shuffle seed test, trainset", train_data[:10])

    train_dataloader = DataLoader(train_data, shuffle=True, batch_size=train_batch_size)
    dev_dataloader = DataLoader(dev_data, shuffle=True, batch_size=train_batch_size)
    test_dataloader = DataLoader(test_data, shuffle=True, batch_size=train_batch_size)
    print("Number of training examples:", len(train_dataloader.dataset))
    print("Number of dev examples:", len(dev_dataloader.dataset))
    print("Number of test examples:", len(test_dataloader.dataset))

    return train_dataloader, dev_dataloader, test_dataloader, train_data, dev_data, test_data

    # Now, you can process these files or save them as needed
file_name = "files_has_0_distance.txt"

# 使用 'with' 语句打开文件进行写入，确保文件最后会被正确关闭


# print(samples, type(samples))



# print(shuffled_samples, type(shuffled_samples))

train_ratio = 0.8
dev_ratio = 0.1
test_ration = 0.1

# train_data, temp_data = train_test_split(shuffled_samples, test_size=(1 - train_ratio))
# dev_data, test_data = train_test_split(temp_data, test_size=0.5)

# print(type(dev_data[0]))

#
# train_dataloader = DataLoader(train_data, shuffle=True, batch_size=train_batch_size)
# dev_dataloader = DataLoader(dev_data, batch_size=train_batch_size)
if split_method_index == 1:
    train_dataloader, dev_dataloader, test_dataloader, train_data, dev_data, test_data = split_method_1(csv_pair_size_limit, distance_limit)

elif split_method_index == 2:
    train_dataloader, dev_dataloader, test_dataloader, train_data, dev_data, test_data = split_method_2(csv_pair_size_limit, distance_limit)
train_loss = losses.SoftmaxLoss(model=model, sentence_embedding_dimension=model.get_sentence_embedding_dimension(), num_labels=2)

logging.info("Read STSbenchmark dev dataset")
evaluator = LabelAccuracyEvaluator(dev_dataloader, softmax_model=train_loss, name='polarity-dev')

warmup_steps = math.ceil(len(train_dataloader) * num_epochs  * 0.1) #10% of train data for warm-up
logging.info("Warmup-steps: {}".format(warmup_steps))




model.fit(train_objectives=[(train_dataloader, train_loss)],
          evaluator=evaluator,
          epochs=num_epochs,
          evaluation_steps=1000,
          warmup_steps=warmup_steps,
          output_path=model_save_path)

model = SentenceTransformer(model_save_path)
model = model.to(device)

    # 进行后续处理

    # 然后使用这些数据进行训练或评估

# test_dataloader = DataLoader(test_data, shuffle=True, batch_size=train_batch_size)

test_evaluator = LabelAccuracyEvaluator(test_dataloader, softmax_model=train_loss, name='polarity-test')

test_evaluator(model, output_path=model_save_path)

generate_yaml(model_name.replace("/", "-")+'-'+datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))

num_training_examples = len(train_dataloader.dataset)
num_dev_examples = len(dev_dataloader.dataset)
num_test_examples = len(test_dataloader.dataset)




with open(model_save_path+'/dataset_size.yaml', 'w') as file:
    file.write(f"Number of training examples: {num_training_examples}\n")
    file.write(f"Number of dev examples: {num_dev_examples}\n")
    file.write(f"Number of test examples: {num_test_examples}\n")
#Our sentences we like to encode

# #Sentences are encoded by calling model.encode()
#
# #Print the embeddings

# print(sentences)
