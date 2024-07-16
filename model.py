from sentence_transformers import SentenceTransformer,  InputExample, losses, models
from transformers import AutoTokenizer, AutoConfig
import torch
import pandas as pd
import os
from torch.utils.data import DataLoader, Dataset
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
import logging
import math
from datetime import datetime
from sklearn.model_selection import train_test_split
import random
import  re
import yaml
import sys
import json
import numpy as np
from sentence_transformers import util
import torch.nn as nn
from torch.nn import DataParallel
from concurrent.futures import ProcessPoolExecutor, as_completed
import torch.nn.functional as F
from sentence_transformers.evaluation import SimilarityFunction
from sentence_transformers.evaluation import SentenceEvaluator
from typing import Dict, Any
# print(torch.cuda.is_available())


# model_name = SentenceTransformer('all-mpnet-base-v2')
model_name = 'sentence-transformers/all-MiniLM-L12-v2'

train_batch_size = int(input("please input train_batch_size "))
num_epochs = int(input("please input num_epochs "))
negative_sample_num = int(input("please input negtive sample num "))
distance_limit = int(input("please input max distance "))
csv_pair_size_limit = int(input("please input csv pair size limit "))
split_method_index = int(input("please input split method index(1: collect all pairs and then split, 2: split csv files first then collect pairs) "))
random_seed_num = int(input("please input random seed number "))


word_embedding_model = models.Transformer(model_name)
random.seed(random_seed_num)
torch.manual_seed(random_seed_num)
np.random.seed(random_seed_num)



# Apply mean pooling to get one fixed sized sentence vector
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
                               pooling_mode_mean_tokens=True,
                               pooling_mode_cls_token=False,
                               pooling_mode_max_tokens=False)



model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DataParallel(model)
model.to(device)


logging.info("Read STSbenchmark train dataset")
# Apply mean pooling to get one fixed sized sentence vector

print('running')
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")


# # debates = load_debates_from_folder('/Users/fanzhe/Desktop/master_thesis/Data/kialo_debatetree_data/englishdebates')
# debates_path = '/Users/fanzhe/Desktop/master_thesis/Data/kialo_debatetree_data/csv_sample'

# # debates = load_debates_from_folder('/home/users0/fanze/masterarbeit/englishdebates')
debates_path = '/mount/studenten5/projects/fanze/masterarbeit_data/csv_nofilter'



# model_save_path = '/Users/fanzhe/Desktop/master_thesis/Data/model_ouput/training_stsbenchmark_'+model_name.replace("/", "-")+'-'+datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
model_save_path = '/mount/studenten5/projects/fanze/masterarbeit_data/relatedness_output/training_stsbenchmark_'+model_name.replace("/", "-")+'-'+datetime.now().strftime("%Y-%m-%d_%H-%M-%S"+'_meth'+str(split_method_index)+'_pairsize'+str(csv_pair_size_limit)+'_dis'+str(distance_limit))
# model_save_path = '/mount/studenten5/projects/fanze/masterarbeit_data/relatedness_output/training_stsbenchmark_sentence-transformers-all-MiniLM-L12-v2-2024-06-08_19-44-58_meth1_pairsize100000_dis10'


def skip_argument(text):
    pattern = r"-> See \d+(\.\d+)+\."

    return bool(re.match(pattern, text))


# 按照最大对数的限制读取并过滤文件，返回该文件中符合距离的argument对sample，不重复的content列表和0distance文件
def process_file(file_path, max_pairs_size, max_distance):
    df = pd.read_csv(file_path)
    sampled_df = df if len(df) < max_pairs_size else df.sample(n=max_pairs_size, random_state=random_seed_num)

    samples = []
    content_1_list = []
    files_has_0_distance = []

    for index, row in sampled_df.iterrows():
        if float(row['distance']) != 0 and not skip_argument(row['content_1']) and not skip_argument(row['content_2']) and float(row['distance']) <= max_distance:
            score = 1 / float(row['distance'])
            inp_example = InputExample(texts=[row['content_1'], row['content_2']], label=score)
            samples.append(inp_example)
            file_index, _ = os.path.splitext(os.path.basename(file_path))
            if row['content_1'] not in [content['content'] for content in content_1_list]:
                content_1_list.append({"file_index": str(file_index), "content": row['content_1']})
        elif float(row['distance']) == 0:
            files_has_0_distance.append(file_path)
    

    return samples, content_1_list, files_has_0_distance


# csv_files = [file for file in all_files if file.endswith('.csv') and (len(pd.read_csv(os.path.join(debates_path, file))) - 1) < 100000]
def split_method_1(max_pairs_size, max_distance):
    random.seed(random_seed_num)
    torch.manual_seed(random_seed_num)
    np.random.seed(random_seed_num)

    all_files = os.listdir(debates_path)
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
    print('number of total pairs', samples)
    print("argument number",  len(content_1_list))

    for content_1 in content_1_list:
        rest_of_contents = [f for f in content_1_list if f["file_index"] != content_1["file_index"]]
        random_negative_arguments = []
        while len(random_negative_arguments) < negative_sample_num:
            random_index_content = random.choice(rest_of_contents)
            random_content = random_index_content["content"]
            if random_content not in random_negative_arguments:
                random_negative_arguments.append(random_content)

        for negative_argument in random_negative_arguments:
            neg_inp_example = InputExample(texts=[content_1["content"], negative_argument], label=0.0)
            samples.append(neg_inp_example)

    with open("files_has_0_distance.txt", "w") as file:
        for line in files_has_0_distance:
            file.write(line + "\n")

    random.shuffle(samples)
    sample_collection = samples

    train_ratio = 0.8
    dev_ratio = 0.1
    test_ratio = 0.1

    train_data, temp_data = train_test_split(sample_collection, test_size=(1 - train_ratio), shuffle=True, random_state=random_seed_num)
    dev_data, test_data = train_test_split(temp_data, test_size=0.5, shuffle=True, random_state=random_seed_num)

    # train_dataset = CustomDataset(train_data)
    # dev_dataset = CustomDataset(dev_data)
    # test_dataset = CustomDataset(test_data)

    train_dataloader = DataLoader(train_data, shuffle=True, batch_size=train_batch_size, pin_memory=True)
    dev_dataloader = DataLoader(dev_data, shuffle=True, batch_size=train_batch_size, pin_memory=True)
    test_dataloader = DataLoader(test_data, shuffle=True, batch_size=train_batch_size, pin_memory=True)

    print("Number of training examples:", len(train_dataloader.dataset))
    print("Number of dev examples:", len(dev_dataloader.dataset))
    print("Number of test examples:", len(test_dataloader.dataset))
    
    

    return train_dataloader, dev_dataloader, test_dataloader, train_data, dev_data, test_data


# Split the csv files first method2
def split_method_2(max_pairs_size, max_distance):

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

    train_files_count = int(train_ratio * len(csv_files_after_filter))
    dev_files_count = int(dev_ratio * len(csv_files_after_filter))
    test_files_count = int(test_ratio * len(csv_files_after_filter))

    assert train_files_count + dev_files_count + test_files_count <= len(csv_files_after_filter), "File counts exceed total."

    train_files = csv_files_after_filter[:train_files_count]
    dev_files = csv_files_after_filter[train_files_count:train_files_count + dev_files_count]
    test_files = csv_files_after_filter[train_files_count + dev_files_count:]
    print("train files length", len(train_files))
    print("dev files length", len(dev_files))
    print("test files length", len(test_files))

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

    

    def add_negative_samples(content_1_list, negative_sample_num, samples):
        # print('content_1_list', content_1_list)
        for content_1 in content_1_list:
            rest_of_contents = [f for f in content_1_list if f["file_index"] != content_1["file_index"]]
            # print('rest_of_contents_len', len(rest_of_contents))
            # print('test', [t["file_index"] for t in rest_of_contents])

            if not rest_of_contents:
                print(f"Skipping {content_1} because rest_of_contents is empty")
                continue  # 如果 rest_of_contents 为空，跳过这次循环
            random_negative_arguments = []
            while len(random_negative_arguments) < negative_sample_num:
                random_index_content = random.choice(rest_of_contents)
                random_content = random_index_content["content"]
                if random_content not in random_negative_arguments:
                    random_negative_arguments.append(random_content)
            # print('negative_arguments_len', len(random_negative_arguments))


            for negative_argument in random_negative_arguments:
                neg_inp_example = InputExample(texts=[content_1["content"], negative_argument], label=0.0)
                samples.append(neg_inp_example)
    print('train_content_1_list', len(train_content_1_list), 'dev_content_1_list', len(dev_content_1_list), 'test_content_1_list', len(test_content_1_list))

    add_negative_samples(train_content_1_list, negative_sample_num, train_data)
    add_negative_samples(dev_content_1_list, negative_sample_num, dev_data)
    add_negative_samples(test_content_1_list, negative_sample_num, test_data)

    with open("files_has_0_distance.txt", "w") as file:
        for line in files_has_0_distance:
            file.write(line + "\n")

    random.shuffle(train_data)
    random.shuffle(dev_data)
    random.shuffle(test_data)

    # train_dataset = CustomDataset(train_data)
    # dev_dataset = CustomDataset(dev_data)
    # test_dataset = CustomDataset(test_data)

    train_dataloader = DataLoader(train_data, shuffle=True, batch_size=train_batch_size, pin_memory=True)
    dev_dataloader = DataLoader(dev_data, shuffle=True, batch_size=train_batch_size, pin_memory=True)
    test_dataloader = DataLoader(test_data, shuffle=True, batch_size=train_batch_size, pin_memory=True)

    print('model_name', model_name, 'train_batch_size', str(train_batch_size), 'num_epochs', str(num_epochs), 'negative_sample_num', str(negative_sample_num), 'distance_limit', str(distance_limit), 'csv_pair_size_limit', str(csv_pair_size_limit),'split_method_index', str(split_method_index), 'random_seed_num', str(random_seed_num))

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
    
   
    train_ratio = 0.8
    dev_ratio = 0.1
    test_ratio = 0.1    
    assert train_ratio + dev_ratio + test_ratio == 1, "Ratios must sum up to 1."



    train_data = []
    dev_data = []
    test_data = []
    files_has_0_distance = []

    train_contents_total = {}
    dev_contents_total = {}
    test_contents_total = {}

    def process_file_split_method_3_split(file_path, max_pairs_size, max_distance):
        df = pd.read_csv(file_path)
        sampled_df = df if len(df) < max_pairs_size else df.sample(n=max_pairs_size, random_state=random_seed_num)
        
        train_ratio = 0.8
        dev_ratio = 0.1
        test_ratio = 0.1

        

        train_contents_for_neg = []
        dev_contents_for_neg = []
        test_contents_for_neg = []

        file_argument_list = []
        train_argument_list = []
        dev_argument_list = []
        test_argument_list = []

        files_has_0_distance = []
        file_index, _ = os.path.splitext(os.path.basename(file_path))

        # print('TESTTTT', file_index, csv_files_after_filter[0])
        



        for index, row in sampled_df.iterrows():
            if float(row['distance']) != 0 and not skip_argument(row['content_1']) and not skip_argument(row['content_2']):
                if row['content_1'] not in [content['content'] for content in file_argument_list]:
                    file_argument_list.append({"file_index": str(file_index), "content": row['content_1']})
            elif float(row['distance']) == 0:
                files_has_0_distance.append(file_path)
        # print('TEST_0', len(file_argument_list))
       
        train_arg_num = max(1, int(train_ratio * len(file_argument_list)))
        dev_arg_num = max(1, int(dev_ratio * len(file_argument_list)))
        test_arg_num = max(1, int(test_ratio * len(file_argument_list)))

        # 确保总数不超过列表长度
        total_args = train_arg_num + dev_arg_num + test_arg_num
        if total_args > len(file_argument_list):
            train_arg_num -= (total_args - len(file_argument_list))

        # print('testnum',train_arg_num,dev_arg_num,test_arg_num)

        assert train_arg_num + dev_arg_num + test_arg_num <= len(file_argument_list), "Total size exceeds the list length."

        train_arg_set = file_argument_list[:train_arg_num]
        dev_arg_set = file_argument_list[train_arg_num:train_arg_num + dev_arg_num]
        test_arg_set = file_argument_list[train_arg_num + dev_arg_num:]
        # print('TEST_1', len(train_arg_set), len(dev_arg_set), len(test_arg_set))

        train_contents = {item['content'] for item in train_arg_set}
        dev_contents = {item['content'] for item in dev_arg_set}
        test_contents = {item['content'] for item in test_arg_set}

        train_contents_total[file_index] = train_contents
        dev_contents_total[file_index] = dev_contents
        test_contents_total[file_index] = test_contents

        
        return train_arg_set, dev_arg_set, test_arg_set, files_has_0_distance
        
    def process_file_split_method_3_split_filter(file_path, max_pairs_size, max_distance):
        df = pd.read_csv(file_path)
        sampled_df = df if len(df) < max_pairs_size else df.sample(n=max_pairs_size, random_state=random_seed_num)
        file_index, _ = os.path.splitext(os.path.basename(file_path))
        train_other_keys = [key for key in train_contents_total if key != file_index]
        dev_other_keys = [key for key in dev_contents_total if key != file_index]
        test_other_keys = [key for key in test_contents_total if key != file_index]
        train_neg_num_file = 0 
        dev_neg_num_file = 0 
        test_neg_num_file = 0 


        train_samples_file = []
        dev_samples_file =[]
        test_samples_file = []
        train_samples_neg_file = []
        dev_samples_neg_file =[]
        test_samples_neg_file = []

        train_argument_index_set_file = []
        dev_argument_index_set_file = []
        test_argument_index_set_file = []


        def judge_same_set(content_1, content_2, train_set, dev_set, test_set):

            if (content_1 in train_set and content_2 in train_set) or \
            (content_1 in dev_set and content_2 in dev_set) or \
            (content_1 in test_set and content_2 in test_set):
                return True
            return False

         

        print('TEST_2', file_index)
        for index, row in sampled_df.iterrows():
            
            
            if float(row['distance']) != 0 and not skip_argument(row['content_1']) and not skip_argument(row['content_2']) and float(row['distance']) <= max_distance:
                
                if judge_same_set(row['content_1'], row['content_2'], train_contents_total[file_index], dev_contents_total[file_index], test_contents_total[file_index]):
                        score = 1 / float(row['distance'])
                        inp_example = InputExample(texts=[row['content_1'], row['content_2']], label=score)

                        # print(score, row['content_1'], row['content_2'])
                        if row['content_1'] in train_contents_total[file_index] and row['content_2'] in train_contents_total[file_index]:
                            # print('TEST_2.5train', len(train_samples))
                            train_samples_file.append(inp_example)
                            random_key = random.sample(train_other_keys, negative_sample_num)
                            # 从选择的键对应的子字典中随机选取一条内容
                            if row['index_1'] not in train_argument_index_set_file:
                                for key in random_key:
                                    random_content = random.choice(list(train_contents_total[key]))
                                    neg_inp_example = InputExample(texts=[row['content_1'], random_content], label=0.0)
                                    train_samples_neg_file.append(neg_inp_example)
                                    train_neg_num_file += 1
                                    train_argument_index_set_file.append(row['index_1'])
                            if row['index_2'] not in train_argument_index_set_file:
                                for key in random_key:
                                    random_content = random.choice(list(train_contents_total[key]))
                                    neg_inp_example = InputExample(texts=[row['content_2'], random_content], label=0.0)
                                    train_samples_neg_file.append(neg_inp_example)
                                    train_neg_num_file += 1
                                    train_argument_index_set_file.append(row['index_2'])

                            
                        elif row['content_1'] in dev_contents_total[file_index] and row['content_2'] in dev_contents_total[file_index]:
                            # print('TEST_2.5dev', len(dev_samples))
                            dev_samples_file.append(inp_example)
                            random_key = random.sample(dev_other_keys, negative_sample_num)
                            # 从选择的键对应的子字典中随机选取一条内容
                            if row['index_1'] not in dev_argument_index_set_file:
                                for key in random_key:
                                    random_content = random.choice(list(dev_contents_total[key]))
                                    neg_inp_example = InputExample(texts=[row['content_1'], random_content], label=0.0)
                                    dev_samples_neg_file.append(neg_inp_example)
                                    dev_neg_num_file += 1
                                    dev_argument_index_set_file.append(row['index_1'])
                            if row['index_2'] not in dev_argument_index_set_file:
                                for key in random_key:
                                    random_content = random.choice(list(dev_contents_total[key]))
                                    neg_inp_example = InputExample(texts=[row['content_2'], random_content], label=0.0)
                                    dev_samples_neg_file.append(neg_inp_example)
                                    dev_neg_num_file += 1
                                    dev_argument_index_set_file.append(row['index_2'])
                                
                        elif row['content_1'] in test_contents_total[file_index] and row['content_2'] in test_contents_total[file_index]:
                            # print('TEST_2.5test', len(test_samples))
                            test_samples_file.append(inp_example)
                            random_key = random.sample(test_other_keys, negative_sample_num)
                            # 从选择的键对应的子字典中随机选取一条内容
                            if row['index_1'] not in test_argument_index_set_file:
                                for key in random_key:
                                    random_content = random.choice(list(test_contents_total[key]))
                                    neg_inp_example = InputExample(texts=[row['content_1'], random_content], label=0.0)
                                    test_samples_neg_file.append(neg_inp_example)
                                    test_neg_num_file += 1
                                    test_argument_index_set_file.append(row['index_1'])
                            if row['index_2'] not in test_argument_index_set_file:
                                for key in random_key:
                                    random_content = random.choice(list(test_contents_total[key]))
                                    neg_inp_example = InputExample(texts=[row['content_2'], random_content], label=0.0)
                                    test_samples_neg_file.append(neg_inp_example)
                                    test_neg_num_file += 1
                                    test_argument_index_set_file.append(row['index_2'])
        # print('TEST_3', len(train_samples), len(dev_samples), len(test_samples))
                            
                        
                                                

        return train_samples_file, dev_samples_file, test_samples_file, train_neg_num_file, dev_neg_num_file, test_neg_num_file, train_samples_neg_file, dev_samples_neg_file, test_samples_neg_file

    def process_files_parallel(files):
        train_samples_total = []
        dev_samples_total = []
        test_samples_total = []

        content_1_list = []
        train_argument_list = []
        dev_argument_list = []
        test_argument_list = []
        # train_contents_for_neg_total =[]
        # dev_contents_for_neg_total =[]
        # test_contents_for_neg_total =[]
        train_neg_num = 0
        dev_neg_num = 0
        test_neg_num = 0
        train_samples_neg_total = []
        dev_samples_neg_total = []
        test_samples_neg_total = []



        # with ProcessPoolExecutor() as executor:
        #     futures = [executor.submit(process_file_split_method_3_split, os.path.join(debates_path, file), max_pairs_size, max_distance) for file in files]
        #     for future in as_completed(futures):
        # # for file in files:
        #         train_arg_set, dev_arg_set, test_arg_set, file_has_0_distance = future.result()

        for file in files:
            result = process_file_split_method_3_split(os.path.join(debates_path, file), max_pairs_size, max_distance)
            train_arg_set, dev_arg_set, test_arg_set, file_has_0_distance= result
            # file_train_samples, file_dev_samples, file_test_samples, file_content_1_list,train_arg_set, dev_arg_set, test_arg_set, file_has_0_distance, train_contents_for_neg, dev_contents_for_neg,test_contents_for_neg = future.result()
            # file_train_samples, file_dev_samples, file_test_samples, file_content_1_list,train_arg_set, dev_arg_set, test_arg_set, file_has_0_distance, train_contents_for_neg, dev_contents_for_neg,test_contents_for_neg = process_file_split_method_3(os.path.join(debates_path, file), max_pairs_size, max_distance)

            # print('TEST', file_train_samples, file_test_samples, train_arg_set[:1])
            # print('TEST_102', len(file_train_samples), len(file_dev_samples), len(file_test_samples))


            # train_samples.extend(file_train_samples)
            # dev_samples.extend(file_dev_samples)
            # test_samples.extend(file_test_samples)
            # content_1_list.extend(file_content_1_list)
            train_argument_list.extend(train_arg_set)
            dev_argument_list.extend(dev_arg_set)
            test_argument_list.extend(test_arg_set)
            files_has_0_distance.extend(file_has_0_distance)
            # train_contents_for_neg_total.extend(train_contents_for_neg)
            # dev_contents_for_neg_total.extend(dev_contents_for_neg)
            # test_contents_for_neg_total.extend(test_contents_for_neg)

        # with ProcessPoolExecutor() as executor:
        #     futures = [executor.submit(process_file_split_method_3_split_filter, os.path.join(debates_path, file), max_pairs_size, max_distance) for file in files]
        #     for future in as_completed(futures):
        # # for file in files:
        #         file_train_samples, file_dev_samples, file_test_samples, train_neg_num_file, dev_neg_num_file, test_neg_num_file, file_train_samples_neg, file_dev_samples_neg, file_test_samples_neg = future.result()
                # file_train_samples, file_dev_samples, file_test_samples, file_content_1_list,train_arg_set, dev_arg_set, test_arg_set, file_has_0_distance, train_contents_for_neg, dev_contents_for_neg,test_contents_for_neg = future.result()
                # file_train_samples, file_dev_samples, file_test_samples, file_content_1_list,train_arg_set, dev_arg_set, test_arg_set, file_has_0_distance, train_contents_for_neg, dev_contents_for_neg,test_contents_for_neg = process_file_split_method_3(os.path.join(debates_path, file), max_pairs_size, max_distance)
        for file in files:
            result_2 = process_file_split_method_3_split_filter(os.path.join(debates_path, file), max_pairs_size, max_distance)
            file_train_samples, file_dev_samples, file_test_samples, train_neg_num_file, dev_neg_num_file, test_neg_num_file, file_train_samples_neg, file_dev_samples_neg, file_test_samples_neg= result_2
            # print('TEST', file_train_samples, file_test_samples, train_arg_set[:1])
            # print('TEST_102', len(file_train_samples), len(file_dev_samples), len(file_test_samples))


            train_samples_total.extend(file_train_samples)
            dev_samples_total.extend(file_dev_samples)
            test_samples_total.extend(file_test_samples)
            train_samples_neg_total.extend(file_train_samples_neg)
            dev_samples_neg_total.extend(file_dev_samples_neg) 
            test_samples_neg_total.extend(file_test_samples_neg)
            # content_1_list.extend(file_content_1_list)
            # train_argument_list.extend(train_arg_set)
            # dev_argument_list.extend(dev_arg_set)
            # test_argument_list.extend(test_arg_set)
            train_neg_num = train_neg_num + train_neg_num_file
            dev_neg_num = dev_neg_num + dev_neg_num_file
            test_neg_num = test_neg_num + test_neg_num_file
            # files_has_0_distance.extend(file_has_0_distance)
            # train_contents_for_neg_total.extend(train_contents_for_neg)
            # dev_contents_for_neg_total.extend(dev_contents_for_neg)
            # test_contents_for_neg_total.extend(test_contents_for_neg)
        print('negnum', train_neg_num, dev_neg_num, test_neg_num)
        print('positivesample and negative train',len(train_samples_total), len(train_samples_neg_total))
        train_samples_total.extend(train_samples_neg_total)
        dev_samples_total.extend(dev_samples_neg_total)
        test_samples_total.extend(test_samples_neg_total)

        # print('TEST_101', len(train_samples), len(dev_samples), len(test_samples), len(train_contents_for_neg_total), len(dev_contents_for_neg_total), len(test_contents_for_neg_total), len(train_argument_list), len(dev_argument_list), len(test_argument_list))
        return train_samples_total, dev_samples_total, test_samples_total, train_argument_list, dev_argument_list, test_argument_list, files_has_0_distance







    # 写入日志文件
    if files_has_0_distance:
        log_file_path = "path/to/logs/files_has_0_distance.txt"
        os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
        with open(log_file_path, 'w') as log_file:
            for file_path in files_has_0_distance:
                log_file.write(file_path + "\n")
    else:
        print("No files with 0 distance found.")

    train_samples,dev_samples, test_samples, train_argument_list, dev_argument_list, test_argument_list, files_has_0_distance = process_files_parallel(csv_files_after_filter)
    
    print('total set split train dev test', train_argument_list[:1], len(train_argument_list), len(dev_argument_list), len(test_argument_list))
    print('3 for train dev test', train_samples[:3], dev_samples[:3], test_samples[:3])

    train_data.extend(train_samples)
    dev_data.extend(dev_samples)
    test_data.extend(test_samples)
    files_has_0_distance.extend(files_has_0_distance)
    print('train sample num without neg',len(train_data),'dev sample num without neg', len(dev_data),'test sample num without neg',len(test_data),len(train_samples), len(dev_samples),len(test_samples))

    # def add_negative_samples(content_1_list, negative_sample_num, samples):
    #     # print('content_1_list', content_1_list)
    #     neg_sample_num = 0
    #     content_by_file_index = {}

    #     for content in content_1_list:
    #         if content["file_index"] not in content_by_file_index:
    #             content_by_file_index[content["file_index"]] = []
    #         content_by_file_index[content["file_index"]].append(content["content"])

    #     for content_1 in content_1_list:
    #         rest_of_contents = [content for index, contents in content_by_file_index.items() if index != content_1["file_index"] for content in contents]
    #         # print('rest_of_contents_len', len(rest_of_contents))
    #         # print('test', [t["file_index"] for t in rest_of_contents])
    #         if len(rest_of_contents) < negative_sample_num:
    #             print(f"Skipping {content_1} because there are not enough negative samples")
    #             continue 
    #         random_negative_arguments = random.sample(rest_of_contents, negative_sample_num)

    #         # random_negative_arguments = []
    #         # while len(random_negative_arguments) < negative_sample_num:
    #         #     random_index_content = random.choice(rest_of_contents)
    #         #     random_content = random_index_content["content"]
    #         #     if random_content not in random_negative_arguments:
    #         #         random_negative_arguments.append(random_content)

    #         for negative_argument in random_negative_arguments:
    #             neg_inp_example = InputExample(texts=[content_1["content"], negative_argument], label=0.0)
    #             samples.append(neg_inp_example)
    #             neg_sample_num += 1
    #     print('negative_sample', neg_sample_num)

    # add_negative_samples(train_contents_for_neg_total, negative_sample_num, train_data)
    # add_negative_samples(dev_contents_for_neg_total, negative_sample_num, dev_data)
    # add_negative_samples(test_contents_for_neg_total, negative_sample_num, test_data)

    random.shuffle(train_data)
    random.shuffle(dev_data)
    random.shuffle(test_data)
 
    # random_train_data = random.sample(train_data, min(len(train_data), max_pairs_size_train))
    # random_dev_data = random.sample(dev_data, min(len(dev_data), max_pairs_size_dev))
    # random_test_data = random.sample(test_data, min(len(test_data), max_pairs_size_test))

    # print('len(train_data)', len(train_data), 'max_pairs_size_train', max_pairs_size_train)
    # print('TEST_102', train_data[:2], dev_data[:2], test_data[:2])

    train_dataloader = DataLoader(train_data, shuffle=True, batch_size=32)
    dev_dataloader = DataLoader(dev_data, shuffle=True, batch_size=32)
    test_dataloader = DataLoader(test_data, shuffle=True, batch_size=32)

    # print('argument_size', len(content_1_list), 'train_argument_list', len(train_argument_list),'dev_argument_list', len(dev_argument_list),'test_argument_list', len(test_argument_list))
    print("Number of training examples:", len(train_dataloader.dataset))
    print("Number of dev examples:", len(dev_dataloader.dataset))
    print("Number of test examples:", len(test_dataloader.dataset))

    return train_dataloader, dev_dataloader, test_dataloader, train_data, dev_data, test_data

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


if split_method_index == 1:
    train_dataloader, dev_dataloader, test_dataloader, train_data, dev_data, test_data = split_method_1(csv_pair_size_limit, distance_limit)

elif split_method_index == 2:
    train_dataloader, dev_dataloader, test_dataloader, train_data, dev_data, test_data = split_method_2(csv_pair_size_limit, distance_limit)
elif split_method_index == 3:
    train_dataloader, dev_dataloader, test_dataloader, train_data, dev_data, test_data = split_method_3(csv_pair_size_limit, distance_limit)

train_loss = losses.CosineSimilarityLoss(model=model)
# train_loss = losses.CoSENTLoss(model=model)





logging.info("Read STSbenchmark dev dataset")
evaluator = EmbeddingSimilarityEvaluator.from_input_examples(dev_data, name='sts-dev', main_similarity=SimilarityFunction.COSINE)



warmup_steps = math.ceil(len(train_dataloader) * num_epochs  * 0.1) #10% of train data for warm-up
logging.info("Warmup-steps: {}".format(warmup_steps))

    


model.fit(train_objectives=[(train_dataloader, train_loss)],
          evaluator=evaluator,
          epochs=num_epochs,
          evaluation_steps=1000,
          warmup_steps=warmup_steps,
          output_path=model_save_path)

model = SentenceTransformer(model_save_path)
test_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(test_data, name='sts-test', main_similarity=SimilarityFunction.COSINE)

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
# TestPC

# autosavetest
# updatetest