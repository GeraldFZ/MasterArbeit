
from sentence_transformers import SentenceTransformer,  InputExample, losses, models
from transformers import AutoTokenizer
import torch
import pandas as pd
import os
from torch.utils.data import DataLoader
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
import logging
import math
from datetime import datetime
from sklearn.model_selection import train_test_split
import random
import  re
# print(torch.cuda.is_available())


# model_name = SentenceTransformer('all-mpnet-base-v2')
model_name = 'sentence-transformers/all-MiniLM-L12-v2'

train_batch_size = 64
num_epochs = 6
word_embedding_model = models.Transformer(model_name)

# Apply mean pooling to get one fixed sized sentence vector
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
                               pooling_mode_mean_tokens=True,
                               pooling_mode_cls_token=False,
                               pooling_mode_max_tokens=False)



model = SentenceTransformer(modules=[word_embedding_model, pooling_model])



logging.info("Read STSbenchmark train dataset")
# Apply mean pooling to get one fixed sized sentence vector


tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")


# # debates = load_debates_from_folder('/Users/fanzhe/Desktop/master_thesis/Data/kialo_debatetree_data/englishdebates')
# debates_path = '/Users/fanzhe/Desktop/master_thesis/Data/kialo_debatetree_data/csv_sample'

# # debates = load_debates_from_folder('/home/users0/fanze/masterarbeit/englishdebates')
debates_path = '/mount/studenten5/projects/fanze/masterarbeit_data/csv_nofilter'



# model_save_path = '/Users/fanzhe/Desktop/master_thesis/Data/model_ouput/training_stsbenchmark_'+model_name.replace("/", "-")+'-'+datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
model_save_path = '/mount/studenten5/projects/fanze/masterarbeit_data/model_output/training_stsbenchmark_'+model_name.replace("/", "-")+'-'+datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

def skip_argument(text):
    pattern = r"-> See \d+(\.\d+)+\."

    return bool(re.match(pattern, text))




# csv_files = [file for file in all_files if file.endswith('.csv') and (len(pd.read_csv(os.path.join(debates_path, file))) - 1) < 100000]
def split_method_1(debate_size_threshold, distance_threshold):

    all_files = os.listdir(debates_path)
    csv_files = [file for file in all_files if file.endswith('.csv') and (len(pd.read_csv(os.path.join(debates_path, file))) - 1) < debate_size_threshold]
    random.shuffle(csv_files)
    shuffled_csv_files = csv_files
    samples = []

    # 逐个读取CSV文件
    for file in shuffled_csv_files:
        file_path = os.path.join(debates_path, file)
        # 读取CSV文件
        df = pd.read_csv(file_path)
        # print(file_path)
        number_of_pairs = len(df) - 1

        # 按行处理数据
        files_has_0_distance = []
        for index, row in df.iterrows():
            if float(row['distance']) != 0 and not skip_argument(row['content_1']) and not skip_argument(row['content_2']) and float(row['distance']) <= distance_threshold:
                score =  1/ float(row['distance']) # Normalize score to range 0 ... 1
                inp_example = InputExample(texts=[row['content_1'], row['content_2']], label=score)
                # print(score, row['content_1'], row['content_2'])

                samples.append(inp_example)
            elif float(row['distance']) == 0:
                files_has_0_distance.append(file_path)

                print(file_path, row['distance'])
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
    print(shuffled_samples[:100])



    train_ratio = 0.8
    dev_ratio = 0.1
    test_ratio = 0.1

    train_data, temp_data = train_test_split(sample_collection, test_size=(1 - train_ratio), shuffle=True)
    dev_data, test_data = train_test_split(temp_data, test_size=0.5, shuffle=True)

# print(type(dev_data[0]))

    train_dataloader = DataLoader(train_data, shuffle=True, batch_size=train_batch_size)
    dev_dataloader = DataLoader(dev_data, shuffle=True, batch_size=train_batch_size)
    test_dataloader = DataLoader(test_data,shuffle=True, batch_size=train_batch_size)
    print("Number of training examples:", len(train_dataloader.dataset))
    print("Number of dev examples:", len(dev_dataloader.dataset))
    print("Number of test examples:", len(test_dataloader.dataset))
    return train_dataloader, dev_dataloader, test_dataloader, train_data, dev_data, test_data

def split_method_2(debate_size_threshold, distance_threshold):
    all_files = os.listdir(debates_path)
    csv_files = [file for file in all_files if file.endswith('.csv') and (
                len(pd.read_csv(os.path.join(debates_path, file))) - 1) < debate_size_threshold]
    random.shuffle(csv_files)
    shuffled_csv_files = csv_files
    train_ratio = 0.8
    dev_ratio = 0.1
    test_ratio = 0.1

    # 确保所有比例加起来等于1
    assert train_ratio + dev_ratio + test_ratio == 1, "Ratios must sum up to 1."



    # Calculate the number of files for train, dev, and test sets
    train_files_count = int(train_ratio * len(shuffled_csv_files))
    dev_files_count = int(dev_ratio * len(shuffled_csv_files))
    test_files_count = int(test_ratio * len(shuffled_csv_files))

    # Make sure the total count does not exceed the number of available files
    assert train_files_count + dev_files_count + test_files_count <= len(
        shuffled_csv_files), "File counts exceed total."

    # Split the file paths into training, development, and testing sets
    train_files = shuffled_csv_files[:train_files_count]
    dev_files = shuffled_csv_files[train_files_count:train_files_count + dev_files_count]
    test_files = shuffled_csv_files[train_files_count + dev_files_count:]
    
    train_data = []
    dev_data = []
    test_data = []
    files_has_0_distance = []

    for file in train_files:
        content_1_list = []
        file_path = os.path.join(debates_path, file)
        # 读取CSV文件
        df = pd.read_csv(file_path)
        # print(file_path)

        # 按行处理数据
        for index, row in df.iterrows():
            if float(row['distance']) != 0 and not skip_argument(row['content_1']) and not skip_argument(row['content_2']) and float(row['distance']) <= distance_threshold:
                score =  1/ float(row['distance']) # Normalize score to range 0 ... 1
                inp_example = InputExample(texts=[row['content_1'], row['content_2']], label=score)
                # print(score, row['content_1'], row['content_2'])

                train_data.append(inp_example)
                if row['content_1'] not in content_1_list:
                    content_1_list.append(row['content_1'])
            elif float(row['distance']) == 0:
                files_has_0_distance.append(file_path)

                print(file_path, row['distance'])
        for content_1 in content_1_list:
            rest_of_csv = [f for f in train_files if f != file]
            random_negative_arguments = []
            while len(random_negative_arguments)< 5:
                random_file = random.choice(rest_of_csv)
                df_random_file = pd.read_csv(random_file)
                random_value = df_random_file['content_1'].sample().iloc[0]
                if random_value not in random_negative_arguments:
                    random_negative_arguments.append(random_value)

            for negative_argument in random_negative_arguments:
                neg_inp_example = InputExample(texts=[content_1, negative_argument], label=0)
                train_data.append(neg_inp_example)


    for file in dev_files:
        file_path = os.path.join(debates_path, file)
        # 读取CSV文件
        df = pd.read_csv(file_path)
        # print(file_path)

        # 按行处理数据
        for index, row in df.iterrows():
            if float(row['distance']) != 0 and not skip_argument(row['content_1']) and not skip_argument(row['content_2']) and float(row['distance']) <= distance_threshold:
                score =  1/ float(row['distance']) # Normalize score to range 0 ... 1
                inp_example = InputExample(texts=[row['content_1'], row['content_2']], label=score)
                # print(score, row['content_1'], row['content_2'])

                dev_data.append(inp_example)
            elif float(row['distance']) == 0:
                files_has_0_distance.append(file_path)

                print(file_path, row['distance'])
    for file in test_files:
        file_path = os.path.join(debates_path, file)
        # 读取CSV文件
        df = pd.read_csv(file_path)
        # print(file_path)

        # 按行处理数据
        for index, row in df.iterrows():
            if float(row['distance']) != 0 and not skip_argument(row['content_1']) and not skip_argument(row['content_2']) and float(row['distance']) <= distance_threshold:
                score =  1/ float(row['distance']) # Normalize score to range 0 ... 1
                inp_example = InputExample(texts=[row['content_1'], row['content_2']], label=score)
                # print(score, row['content_1'], row['content_2'])

                test_data.append(inp_example)
            elif float(row['distance']) == 0:
                files_has_0_distance.append(file_path)

                print(file_path, row['distance'])


    file_name = "files_has_0_distance.txt"

    # 使用 'with' 语句打开文件进行写入，确保文件最后会被正确关闭
    with open(file_name, "w") as file:
        # 遍历列表，写入每一行
        for line in files_has_0_distance:
            file.write(line + "\n")  # "\n" 是换行符
    train_dataloader = DataLoader(train_data, shuffle=True, batch_size=train_batch_size)
    dev_dataloader = DataLoader(dev_data, shuffle=True, batch_size=train_batch_size)
    test_dataloader = DataLoader(test_data,shuffle=True, batch_size=train_batch_size)
    print("Number of training examples:", len(train_dataloader.dataset))
    print("Number of dev examples:", len(dev_dataloader.dataset))
    print("Number of test examples:", len(test_dataloader.dataset))
    return train_dataloader, dev_dataloader, test_dataloader, train_data, dev_data, test_data




    # Now, you can process these files or save them as needed

def negative_reader(dataset):
    content_1_list = [example.texts[0] for example in train_data]
    unique_content_1_list = list(set(content_1_list))  # 转换为集合去除重复，再转换回列表


train_dataloader, dev_dataloader, test_dataloader, train_data, dev_data, test_data = split_method_1(100000, 5)


train_loss = losses.CosineSimilarityLoss(model=model)

logging.info("Read STSbenchmark dev dataset")
evaluator = EmbeddingSimilarityEvaluator.from_input_examples(dev_data, name='sts-dev')

warmup_steps = math.ceil(len(train_dataloader) * num_epochs  * 0.1) #10% of train data for warm-up
logging.info("Warmup-steps: {}".format(warmup_steps))




model.fit(train_objectives=[(train_dataloader, train_loss)],
          evaluator=evaluator,
          epochs=num_epochs,
          evaluation_steps=1000,
          warmup_steps=warmup_steps,
          output_path=model_save_path)

model = SentenceTransformer(model_save_path)
test_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(test_data, name='sts-test')
test_evaluator(model, output_path=model_save_path)
#Our sentences we like to encode

# #Sentences are encoded by calling model.encode()
#
# #Print the embeddings

# print(sentences)
