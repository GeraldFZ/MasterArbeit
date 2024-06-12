from sentence_transformers.evaluation import LabelAccuracyEvaluator

from sentence_transformers import SentenceTransformer,  InputExample, losses, models, util, evaluation
from transformers import AutoTokenizer
import pandas as pd
import os
from torch.utils.data import DataLoader, Dataset
import logging
import math
from datetime import datetime
from sklearn.model_selection import train_test_split
import random
import  re
import  yaml
import torch
import numpy as np
import sympy as sp
from scipy.optimize import fsolve
from concurrent.futures import ProcessPoolExecutor, as_completed
import torch.nn as nn
import torch.optim as optim



# model_name = SentenceTransformer('all-mpnet-base-v2')
model_name = 'sentence-transformers/all-MiniLM-L12-v2'
word_embedding_model = models.Transformer(model_name)

# Apply mean pooling to get one fixed sized sentence vector
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
                               pooling_mode_mean_tokens=True,
                               pooling_mode_cls_token=False,
                               pooling_mode_max_tokens=False)
# dense_model = models.Dense(in_features=pooling_model.get_sentence_embedding_dimension(), out_features=1, activation_function=None)
out_features = 1  # 对于二元分类，通常输出层只有一个单元
dense_model = models.Dense(
    in_features=pooling_model.get_sentence_embedding_dimension(),
    out_features=out_features,
    activation_function=None  # 用于分类的输出层通常不需要激活函数，尤其是当后面会使用BCEWithLogitsLoss时
)
model = SentenceTransformer(modules=[word_embedding_model, pooling_model, dense_model])

train_batch_size = int(input("please input train_batch_size "))
num_epochs = int(input("please input num_epochs "))
distance_limit = int(input("please input max distance "))
csv_pair_size_limit = int(input("please input csv pair size limit "))
split_method_index = int(input("please input split method index(1: collect all pairs and then split, 2: split csv files first then collect pairs) "))
random_seed_num = int(input("please input random seed number "))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

random.seed(random_seed_num)

logging.info("Read STSbenchmark train dataset")
# Apply mean pooling to get one fixed sized sentence vector


tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
parameters = model.parameters()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


# # debates = load_debates_from_folder('/Users/fanzhe/Desktop/master_thesis/Data/kialo_debatetree_data/englishdebates')
# debates_path = '/Users/fanzhe/Desktop/master_thesis/Data/kialo_debatetree_data/csv_sample'
# # debates = load_debates_from_folder('/home/users0/fanze/masterarbeit/englishdebates')
# debates_path = '/mount/studenten5/projects/fanze/masterarbeit_data/csv_testmodel'
debates_path = '/mount/studenten5/projects/fanze/masterarbeit_data/csv_nofilter'
# debates_path = '/mount/studenten5/projects/fanze/masterarbeit_data/csv_nofilter'
# model_save_path = '/Users/fanzhe/Desktop/master_thesis/Data/model_ouput/training_stsbenchmark_'+model_name.replace("/", "-")+'-'+datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
model_save_path = '/mount/studenten5/projects/fanze/masterarbeit_data/model_output_record_realloss/training_polarity_'+model_name.replace("/", "-")+'-'+datetime.now().strftime("%Y-%m-%d_%H-%M-%S"+'_meth'+str(split_method_index)+'_pairsize'+str(csv_pair_size_limit)+'_dis'+str(distance_limit))






def skip_argument(text):
    pattern = r"-> See \d+(\.\d+)+\."

    return bool(re.match(pattern, text))
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
    if pd.isna(text):
        return False

    # 使用re.match进行匹配
    return bool(re.match(pattern, text))
class InputExample:
    def __init__(self, texts, label):
        self.texts = texts
        self.label = label
class CustomDataset(Dataset):
    
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text1, text2, label = self.data[idx]
        return text1, text2, torch.tensor(label, dtype=torch.float)
# class CustomDataset(Dataset):
#     def __init__(self, examples):
#         self.examples = examples

#     def __len__(self):
#         return len(self.examples)

#     def __getitem__(self, idx):
#         return self.examples[idx]
# 按照最大对数的限制读取并过滤文件，返回该文件中符合距离的argument对sample，不重复的content列表和0distance文件
def process_file(file_path, max_pairs_size, max_distance):
    df = pd.read_csv(file_path)
    sampled_df = df if len(df) < max_pairs_size else df.sample(n=max_pairs_size, random_state=random_seed_num)

    samples = []
    content_1_list = []
    files_has_0_distance = []

    for index, row in sampled_df.iterrows():
        if float(row['distance']) != 0 and not skip_argument(row['content_1']) and not skip_argument(row['content_2']) and float(row['distance']) <= max_distance:
            if float(row['polarity_consistency']) == 1:
                score = 1 
                inp_example = InputExample(texts=[row['content_1'], row['content_2']], label=score)
                # print(score, row['content_1'], row['content_2'])
                samples.append(inp_example)
                file_index, _ = os.path.splitext(os.path.basename(file_path))
                if row['content_1'] not in [content['content'] for content in content_1_list]:
                    content_1_list.append({"file_index": str(file_index), "content": row['content_1']})
            if float(row['polarity_consistency']) == -1:
                    if float(row['polarity_1']) == 0:
                        # score = 1
                        # inp_example = InputExample(texts=[row['content_1'], row['content_2']], label=score)
                        file_index, _ = os.path.splitext(os.path.basename(file_path))
                        if row['content_1'] not in [content['content'] for content in content_1_list]:
                            content_1_list.append({"file_index": str(file_index), "content": row['content_1']})
                        pass
                    elif float(row['polarity_1']) != 0:
                        score = 0
                        inp_example = InputExample(texts=[row['content_1'], row['content_2']], label=score)

                        samples.append(inp_example)
                        file_index, _ = os.path.splitext(os.path.basename(file_path))
                        if row['content_1'] not in [content['content'] for content in content_1_list]:
                            content_1_list.append({"file_index": str(file_index), "content": row['content_1']})
            
        elif float(row['distance']) == 0:
            files_has_0_distance.append(file_path)

    return samples, content_1_list, files_has_0_distance


def process_file_split_method_3(file_path, max_pairs_size, max_distance):
    df = pd.read_csv(file_path)
    sampled_df = df if len(df) < max_pairs_size else df.sample(n=max_pairs_size, random_state=random_seed_num)

    train_samples = []
    dev_samples =[]
    test_samples = []

    file_argument_list = []
    train_argument_list = []
    dev_argument_list = []
    test_argument_list = []

    files_has_0_distance = []

    for row in sampled_df.iterrows():
        if float(row['distance']) != 0 and not skip_argument(row['content_1']) and not skip_argument(row['content_2']) and float(row['distance']) <= max_distance:
            if row['content_1'] not in [content['content'] for content in file_argument_list]:
                file_argument_list.append({"file_index": str(file_index), "content": row['content_1']})
        elif float(row['distance']) == 0:
            files_has_0_distance.append(file_path)

        train_arg_num = math.ceil(train_ratio * len(file_argument_list))
        dev_arg_num = math.ceil(dev_ratio * len(file_argument_list))
        test_arg_num = math.ceil(test_ratio * len(file_argument_list))

        assert train_arg_num + dev_arg_num + test_arg_num <= len(file_argument_list), "Total size exceeds the list length."

        train_arg_set = file_argument_list[:train_arg_num]
        dev_arg_set = file_argument_list[train_arg_num:train_arg_num + dev_arg_num]
        test_arg_set = file_argument_list[train_arg_num + dev_arg_num:]
        
        def judge_same_set(content_1, content_2, train_set, dev_set, test_set):
            if content_1 in set and content_2 in train_set or content_1 in set and content_2 in dev_set or  content_1 in set and content_2 in test_set:
                return True
            else:
                return False

        for index, row in sampled_df.iterrows():
            if float(row['distance']) != 0 and not skip_argument(row['content_1']) and not skip_argument(row['content_2']) and float(row['distance']) <= max_distance:
                if judge_same_set(row['content_1'], row['content_2'], train_arg_set, dev_arg_set, test_arg_set):
                    if float(row['polarity_consistency']) == 1:
                        score = 1 
                        inp_example = InputExample(texts=[row['content_1'], row['content_2']], label=score)
                        # print(score, row['content_1'], row['content_2'])
                        if row['content_1'] in train_arg_set and row['content_2'] in train_arg_set:
                            train_samples.append(inp_example)
                            file_index, _ = os.path.splitext(os.path.basename(file_path))
                        if row['content_1'] in dev_arg_set and row['content_2'] in dev_arg_set:
                            dev_samples.append(inp_example)
                            file_index, _ = os.path.splitext(os.path.basename(file_path))
                        if row['content_1'] in test_arg_set and row['content_2'] in test_arg_set:
                            test_samples.append(inp_example)
                            file_index, _ = os.path.splitext(os.path.basename(file_path))
                        
                    if float(row['polarity_consistency']) == -1:
                            if float(row['polarity_1']) == 0:
                                # score = 1
                                # inp_example = InputExample(texts=[row['content_1'], row['content_2']], label=score)
                                file_index, _ = os.path.splitext(os.path.basename(file_path))
                                
                                pass
                            elif float(row['polarity_1']) != 0:
                                score = 0
                                inp_example = InputExample(texts=[row['content_1'], row['content_2']], label=score)

                                if row['content_1'] in train_arg_set and row['content_2'] in train_arg_set:
                                    train_samples.append(inp_example)
                                    file_index, _ = os.path.splitext(os.path.basename(file_path))
                                if row['content_1'] in dev_arg_set and row['content_2'] in dev_arg_set:
                                    dev_samples.append(inp_example)
                                    file_index, _ = os.path.splitext(os.path.basename(file_path))
                                if row['content_1'] in test_arg_set and row['content_2'] in test_arg_set:
                                    test_samples.append(inp_example)
                                    file_index, _ = os.path.splitext(os.path.basename(file_path))                       

    return train_samples, dev_samples, test_samples, file_argument_list, train_arg_set, dev_arg_set, test_arg_set, files_has_0_distance


def split_method_1(max_pairs_size, max_distance):
    random.seed(random_seed_num)
    torch.manual_seed(random_seed_num)
    np.random.seed(random_seed_num)

    all_files = os.listdir(debates_path)
    # csv_files = [file for file in all_files if file.endswith('.csv') and  (len(pd.read_csv(os.path.join(debates_path, file)))) < max_pairs_size  ]
    less_than_limit_files = [file for file in all_files if file.endswith('.csv') and len(pd.read_csv(os.path.join(debates_path, file))) < max_pairs_size]
    over_limit_files = [file for file in all_files if file.endswith('.csv') and len(pd.read_csv(os.path.join(debates_path, file))) >= max_pairs_size]
    random.shuffle(over_limit_files)
    random.shuffle(less_than_limit_files)
    print(less_than_limit_files[0:10])
    print(over_limit_files[0:10])

    samples = []
    content_1_list = []
    files_has_0_distance = []

    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(process_file, os.path.join(debates_path, file), max_pairs_size, max_distance) for file in less_than_limit_files]
        for future in as_completed(futures):
            file_samples, file_content_1_list, file_has_0_distance = future.result()
            samples.extend(file_samples)
            content_1_list.extend(file_content_1_list)
            files_has_0_distance.extend(file_has_0_distance)
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(process_file, os.path.join(debates_path, file), max_pairs_size, max_distance) for file in over_limit_files]
        for future in as_completed(futures):
            file_samples, file_content_1_list, file_has_0_distance = future.result()
            samples.extend(file_samples)
            content_1_list.extend(file_content_1_list)
            files_has_0_distance.extend(file_has_0_distance)








    with open("files_has_0_distance.txt", "w") as file:
        for line in files_has_0_distance:
            file.write(line + "\n")
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
    train_dataset = CustomDataset(train_data)
    dev_dataset = CustomDataset(dev_data)
    test_dataset = CustomDataset(test_data)

    # print(type(dev_data[0]))
    def collate_fn(batch):
        # 从 batch 中提取文本和标签
        texts = [example.texts for example in batch]  # 将得到一个列表的列表
        labels = torch.tensor([example.label for example in batch], dtype=torch.float32)
        
        # 处理文本：因为 texts 是列表的列表，我们需要将其展平
        flat_texts = [text for sublist in texts for text in sublist]
        encoded = tokenizer(flat_texts, padding=True, truncation=True, return_tensors="pt")

        # 由于可能有多个文本对应一个标签，确保处理后的标签与模型输出对齐
        return encoded, labels


    train_dataloader = DataLoader(CustomDataset(train_data), shuffle=True, batch_size=train_batch_size)
    dev_dataloader = DataLoader(CustomDataset(dev_data), shuffle=True, batch_size=train_batch_size)
    test_dataloader = DataLoader(CustomDataset(test_data), shuffle=True, batch_size=train_batch_size)
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
    all_files = os.listdir(debates_path)
    less_than_limit_files = [file for file in all_files if file.endswith('.csv') and len(pd.read_csv(os.path.join(debates_path, file))) < max_pairs_size]
    over_limit_files = [file for file in all_files if file.endswith('.csv') and len(pd.read_csv(os.path.join(debates_path, file))) >= max_pairs_size]


    # for file in all_files:
    #     if file.endswith('.csv'):
    #         if (len(pd.read_csv(os.path.join(debates_path, file)))) < max_pairs_size:
    #             less_than_limit_files.append(file)
    #         elif (len(pd.read_csv(os.path.join(debates_path, file)))) >= max_pairs_size:
    #             over_limit_files.append(file)
    random.shuffle(over_limit_files)
    random.shuffle(less_than_limit_files)
    csv_files_after_filter = over_limit_files + less_than_limit_files
    random.shuffle(csv_files_after_filter)

    print(less_than_limit_files[0:10])
    print(over_limit_files[0:10])

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

    def process_files_parallel(files):
        samples = []
        content_1_list = []
        with ProcessPoolExecutor() as executor:
            futures = [executor.submit(process_file, os.path.join(debates_path, file), max_pairs_size, max_distance) for file in files]
            for future in as_completed(futures):
                file_samples, file_content_1_list, file_has_0_distance = future.result()
                samples.extend(file_samples)
                content_1_list.extend(file_content_1_list)
                files_has_0_distance.extend(file_has_0_distance)
        return samples, content_1_list, files_has_0_distance
    train_samples, train_content_1_list, train_files_has_0_distance = process_files_parallel(train_files)
    dev_samples, dev_content_1_list, dev_files_has_0_distance = process_files_parallel(dev_files)
    test_samples, test_content_1_list, test_files_has_0_distance = process_files_parallel(test_files)
    print('train_content_1_list', train_content_1_list[:3], len(train_content_1_list))
    print('3 for train dev test', train_samples[:3], dev_samples[:3], test_samples[:3])

    train_data.extend(train_samples)
    dev_data.extend(dev_samples)
    test_data.extend(test_samples)
    files_has_0_distance.extend(train_files_has_0_distance)
    files_has_0_distance.extend(dev_files_has_0_distance)
    files_has_0_distance.extend(test_files_has_0_distance)
    print('train_content_1_list', len(train_content_1_list), 'dev_content_1_list', len(dev_content_1_list), 'test_content_1_list', len(test_content_1_list))
    with open("files_has_0_distance.txt", "w") as file:
        for line in files_has_0_distance:
            file.write(line + "\n")

    random.shuffle(train_data)
    random.shuffle(dev_data)
    random.shuffle(test_data)
    train_dataset = CustomDataset(train_data)
    dev_dataset = CustomDataset(dev_data)
    test_dataset = CustomDataset(test_data)


 
    







      
    # def collate_fn(batch):
    #     # 从 batch 中提取文本和标签
    #     texts = [example.texts for example in batch]  # 将得到一个列表的列表
    #     labels = torch.tensor([example.label for example in batch], dtype=torch.float32)
        
    #     # 处理文本：因为 texts 是列表的列表，我们需要将其展平
    #     flat_texts = [text for sublist in texts for text in sublist]
    #     encoded = tokenizer(flat_texts, padding=True, truncation=True, return_tensors="pt")

    #     # 由于可能有多个文本对应一个标签，确保处理后的标签与模型输出对齐
    #     return encoded, labels


    # for batch in train_dataloader:
    #     features, labels = batch
    #     outputs = model(**features)  # 使用解包字典的方式传递参数
    #     loss = train_loss(outputs, labels)

    train_dataloader = DataLoader(train_data, shuffle=True, batch_size=train_batch_size)
    dev_dataloader = DataLoader(dev_data, shuffle=True, batch_size=train_batch_size)
    test_dataloader = DataLoader(test_data, shuffle=True, batch_size=train_batch_size)
    print("Number of training examples:", len(train_dataloader.dataset))
    print("Number of dev examples:", len(dev_dataloader.dataset))
    print("Number of test examples:", len(test_dataloader.dataset))

    return train_dataloader, dev_dataloader, test_dataloader, train_data, dev_data, test_data

    # Now, you can process these files or save them as needed



def split_method_3(max_pairs_size, max_distance):
    random.seed(random_seed_num)
    torch.manual_seed(random_seed_num)
    np.random.seed(random_seed_num)
    
    all_files = os.listdir(debates_path)
    less_than_limit_files = [file for file in all_files if file.endswith('.csv') and len(pd.read_csv(os.path.join(debates_path, file))) < max_pairs_size]
    over_limit_files = [file for file in all_files if file.endswith('.csv') and len(pd.read_csv(os.path.join(debates_path, file))) >= max_pairs_size]

    random.shuffle(over_limit_files)
    random.shuffle(less_than_limit_files)
    csv_files_after_filter = over_limit_files + less_than_limit_files
    random.shuffle(csv_files_after_filter)
    
    print(less_than_limit_files[0:10])
    print(over_limit_files[0:10])

    train_ratio = 0.8
    dev_ratio = 0.1
    test_ratio = 0.1    
    assert train_ratio + dev_ratio + test_ratio == 1, "Ratios must sum up to 1."


    # max_pairs_size_train = round(max_pairs_size * train_ratio)
    # max_pairs_size_dev = round(max_pairs_size * dev_ratio)
    # max_pairs_size_test = round(max_pairs_size * test_ratio)

    train_data = []
    dev_data = []
    test_data = []
    files_has_0_distance = []

    # def equation(r, n, S):
    #     x = S / (1 + r)
    #     y = r * x
    #     lhs = n * y * (y - 1)
    #     rhs = x * (x - 1)
    #     return lhs - rhs

    # def solve_ratio(n, S):
    #     initial_guess = 1
    #     ratio = fsolve(equation, initial_guess, args=(n, S))
    #     return ratio[0]

    # def unique_pairs(lst):
    #     return [(lst[i], lst[j]) for i in range(len(lst)) for j in range(i + 1, len(lst))]

    # def process_pairs(df, pairs, data, max_distance):
    #     filtered_data = []
    #     for index_pair in pairs:
    #         selected_row = df[(df['index_1'] == index_pair[0]) & (df['index_2'] == index_pair[1])]
    #         if selected_row.empty:
    #             continue
    #         row = selected_row.iloc[0]
    #         distance = float(row['distance'])
    #         if distance == 0:
    #             files_has_0_distance.append(file_path)
    #             continue
    #         if distance > max_distance or skip_argument(row['content_1']) or skip_argument(row['content_2']):
    #             continue
    #         polarity_consistency = float(row['polarity_consistency'])
    #         polarity_1 = float(row['polarity_1'])
    #         if polarity_consistency == 1:
    #             score = 1
    #         elif polarity_consistency == -1 and polarity_1 != 0:
    #             score = 0
    #         else:
    #             continue
    #         inp_example = InputExample(texts=[row['content_1'], row['content_2']], label=score)
    #         filtered_data.append(inp_example)
    #     return filtered_data
    def process_files_parallel(files):
        train_samples = []
        dev_samples = []
        test_samples = []
        content_1_list = []
        train_argument_list = []
        dev_argument_list = []
        test_argument_list = []


        with ProcessPoolExecutor() as executor:
            futures = [executor.submit(process_file_split_method_3, os.path.join(debates_path, file), max_pairs_size, max_distance) for file in files]
            for future in as_completed(futures):
                file_train_samples, file_dev_samples, file_test_samples, file_content_1_list,train_arg_set, dev_arg_set, test_arg_set, file_has_0_distance = future.result()
                train_samples, dev_samples, test_samples, file_argument_list, train_arg_set, dev_arg_set, test_arg_set, files_has_0_distance
                train_samples.extend(file_train_samples)
                dev_samples.extend(file_dev_samples)
                test_samples.extend(file_test_samples)
                content_1_list.extend(file_content_1_list)
                train_argument_list.extend(train_arg_set)
                dev_argument_list.extend(dev_arg_set)
                test_argument_list.extend(test_arg_set)
                files_has_0_distance.extend(file_has_0_distance)
        return train_samples, dev_samples, test_samples, content_1_list,train_argument_list,dev_argument_list, test_argument_list, files_has_0_distance



    # for file in all_files:
    #     if not file.endswith('.csv'):
    #         continue
    #     print('filename', file)
    #     df = pd.read_csv(os.path.join(debates_path, file))
        
    #     argument_index_list = df['index_1'].unique().tolist()
    #     S = len(argument_index_list)
    #     n = (dev_ratio + test_ratio) / (2 * train_ratio)
    #     ratio = solve_ratio(n, S)
        
    #     train_arguments_num = int(math.ceil(S * (ratio / (1 + ratio))))
    #     dev_test_arguments_num = S - train_arguments_num
    #     dev_arguments_num = int(math.ceil(dev_test_arguments_num / 2))
    #     test_arguments_num = dev_test_arguments_num - dev_arguments_num

        # print("The train pair and dev+test pair ratio y/x for n={} and x+y={}, total list={}, arguments ratio y/x is approximately {:.4f}".format(
        #     n, train_arguments_num + dev_test_arguments_num, S, ratio))
        
        # train_argument_index_list = argument_index_list[:train_arguments_num]
        # dev_argument_index_list = argument_index_list[train_arguments_num:(train_arguments_num + dev_arguments_num)]
        # test_argument_index_list = argument_index_list[(train_arguments_num + dev_arguments_num):]
        
        # train_argument_index_list_pairs = unique_pairs(train_argument_index_list)
        # dev_argument_index_list_pairs = unique_pairs(dev_argument_index_list)
        # test_argument_index_list_pairs = unique_pairs(test_argument_index_list)

        # train_data.extend(process_pairs(df, train_argument_index_list_pairs, train_data, max_distance))
        # dev_data.extend(process_pairs(df, dev_argument_index_list_pairs, dev_data, max_distance))
        # test_data.extend(process_pairs(df, test_argument_index_list_pairs, test_data, max_distance))

    # 打印列表内容以检查
    print("Files with 0 distance:", files_has_0_distance)

    # 写入日志文件
    if files_has_0_distance:
        log_file_path = "path/to/logs/files_has_0_distance.txt"
        os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
        with open(log_file_path, 'w') as log_file:
            for file_path in files_has_0_distance:
                log_file.write(file_path + "\n")
    else:
        print("No files with 0 distance found.")

    train_samples,dev_samples, test_samples, content_1_list, train_argument_list,dev_argument_list, test_argument_list, files_has_0_distance = process_files_parallel(all_files)
    
    print('train_content_1_list', train_argument_list[:3], len(train_argument_list))
    print('3 for train dev test', train_samples[:3], dev_samples[:3], test_samples[:3])

    train_data.extend(train_samples)
    dev_data.extend(dev_samples)
    test_data.extend(test_samples)
    files_has_0_distance.extend(files_has_0_distance)

    random.shuffle(train_data)
    random.shuffle(dev_data)
    random.shuffle(test_data)
 
    # random_train_data = random.sample(train_data, min(len(train_data), max_pairs_size_train))
    # random_dev_data = random.sample(dev_data, min(len(dev_data), max_pairs_size_dev))
    # random_test_data = random.sample(test_data, min(len(test_data), max_pairs_size_test))

    # print('len(train_data)', len(train_data), 'max_pairs_size_train', max_pairs_size_train)

    train_dataloader = DataLoader(train_data, shuffle=True, batch_size=32)
    dev_dataloader = DataLoader(dev_samples, shuffle=True, batch_size=32)
    test_dataloader = DataLoader(test_samples, shuffle=True, batch_size=32)

    print('argument_size', len(content_1_list), 'train_argument_list', len(train_argument_list),'dev_argument_list', len(dev_argument_list),'test_argument_list', len(test_argument_list))
    print("Number of training examples:", len(train_dataloader.dataset))
    print("Number of dev examples:", len(dev_dataloader.dataset))
    print("Number of test examples:", len(test_dataloader.dataset))

    return train_dataloader, dev_dataloader, test_dataloader, train_data, dev_data, test_data




            # train_argument_index_list_sum = train_argument_index_list_sum + train_argument_index_list
            # dev_argument_index_list_sum = dev_argument_index_list_sum + dev_argument_index_list
            # test_argument_index_list_sum = test_argument_index_list_sum + test_argument_index_list





    # shuffled_csv_files = csv_files
    samples = []


file_name = "files_has_0_distance.txt"



train_ratio = 0.8
dev_ratio = 0.1
test_ratio = 0.1


if split_method_index == 1:
    train_dataloader, dev_dataloader, test_dataloader, train_data, dev_data, test_data = split_method_1(csv_pair_size_limit, distance_limit)

elif split_method_index == 2:
    train_dataloader, dev_dataloader, test_dataloader, train_data, dev_data, test_data = split_method_2(csv_pair_size_limit, distance_limit)
elif split_method_index == 3:
    train_dataloader, dev_dataloader, test_dataloader, train_data, dev_data, test_data = split_method_3(csv_pair_size_limit, distance_limit)
# train_loss = losses.SoftmaxLoss(model=model, sentence_embedding_dimension=model.get_sentence_embedding_dimension(), num_labels=2,loss_fct=)
train_loss = nn.BCEWithLogitsLoss()

# train_loss = losses.SoftmaxLoss(model, model.get_sentence_embedding_dimension(), num_labels=2, loss_fct= nn.BCEWithLogitsLoss())
logging.info("Read STSbenchmark dev dataset")
# evaluator = LabelAccuracyEvaluator(dev_dataloader, softmax_model=model, name='polarity-dev')

warmup_steps = math.ceil(len(train_dataloader) * num_epochs  * 0.1) #10% of train data for warm-up
logging.info("Warmup-steps: {}".format(warmup_steps))


model = model.to(device)

# 假设 input_examples 是一个包含多个 InputExample 对象的列表
dev_sentences1 = [example.texts[0] for example in dev_data]
dev_sentences2 = [example.texts[1] for example in dev_data]
dev_labels = [example.label for example in dev_data]

test_sentences1 = [example.texts[0] for example in test_data]
test_sentences2 = [example.texts[1] for example in test_data]
test_labels = [example.label for example in test_data]

print('check the devset', dev_sentences1[:3], dev_sentences2[:3], dev_labels[:3])
print('check the testset', test_sentences1[:3], test_sentences2[:3], test_labels[:3])

dev_evaluator = evaluation.BinaryClassificationEvaluator(dev_sentences1, dev_sentences2, dev_labels)
test_evaluator = evaluation.BinaryClassificationEvaluator(test_sentences1, test_sentences2, test_labels)
# model.fit(train_objectives=[(train_dataloader, train_loss)],
#           evaluator=dev_evaluator,
#           epochs=num_epochs,
#           evaluation_steps=1000,
#           warmup_steps=warmup_steps,
#           output_path=model_save_path)
# model.train()
for epoch in range(num_epochs):
    for texts1, texts2, labels in train_dataloader:
        # 计算嵌入向量
        embeddings1 = model.encode(texts1, convert_to_tensor=True)
        embeddings2 = model.encode(texts2, convert_to_tensor=True)

        # 计算损失，例如使用余弦相似度损失
        cosine_scores = util.pytorch_cos_sim(embeddings1, embeddings2)
        loss = train_loss(cosine_scores, labels)  # 你需要根据你的任务定义这个 loss_function

        # 优化过程
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Epoch: {epoch}, Loss: {loss.item()}")

        if step % 10 == 0:
            print(f"Epoch {epoch}, Step {step}, Loss: {loss.item()}")
    model.eval() 
    with torch.no_grad():
        dev_performance = dev_evaluator(model)
        print(f"Epoch {epoch}, Dev Performance: {dev_performance}")

# 可选地，在所有时期完成后在测试集上进行评估
model.eval()
with torch.no_grad():
    test_performance = test_evaluator(model)
    print(f"Final Test Performance: {test_performance}")



# model = SentenceTransformer(model_save_path)

    # 进行后续处理

    # 然后使用这些数据进行训练或评估

# test_dataloader = DataLoader(test_data, shuffle=True, batch_size=train_batch_size)
# for epoch in range(num_epochs):  # num_epochs 是你需要训练的总轮数
#     model.train()  # 将模型设置为训练模式
#     for batch in train_dataloader:
#         features, labels = batch
#         features = {k: v.to(device) for k, v in features.items()}  # 确保所有数据都移至 GPU
#         labels = labels.to(device)
        
#         optimizer.zero_grad()  # 清空过往梯度
#         outputs = model(**features)  # 前向传播
#         loss = train_loss(outputs, labels)  # 计算损失
#         loss.backward()  # 反向传播
#         optimizer.step()  # 更新参数

#     print(f'Epoch {epoch+1}, Loss: {loss.item()}')
# test_evaluator = LabelAccuracyEvaluator(test_dataloader, softmax_model=model, name='polarity-test')

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
