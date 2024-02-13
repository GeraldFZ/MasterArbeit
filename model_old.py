from debate_class import load_debates_from_folder
import random
from sentence_transformers import SentenceTransformer, util, SentenceTransformer, SentencesDataset, InputExample, losses, models
from transformers import AutoTokenizer, AutoModel

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


# model_name = SentenceTransformer('all-mpnet-base-v2')
model_name = 'distilbert-base-uncased'
model_name = 'sentence-transformers/all-MiniLM-L12-v2'

train_batch_size = 16
num_epochs = 4
train_batch_size = 32
num_epochs = 10
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

train_samples = []

# # debates = load_debates_from_folder('/Users/fanzhe/Desktop/master_thesis/Data/kialo_debatetree_data/englishdebates')
# debates_path = '/Users/fanzhe/Desktop/master_thesis/Data/kialo_debatetree_data/csv_sample_for_model'
# debates_path = '/Users/fanzhe/Desktop/master_thesis/Data/kialo_debatetree_data/csv_sample'
debates_path = '/mount/studenten5/projects/fanze/masterarbeit_data/csv_nofilter'


# # debates = load_debates_from_folder('/home/users0/fanze/masterarbeit/englishdebates')
#
debates_path = '/mount/studenten5/projects/fanze/masterarbeit_data/csv_testmodel'
# debates_path = '/mount/studenten5/projects/fanze/masterarbeit_data/csv_nofilter'



# model_save_path = '/Users/fanzhe/Desktop/master_thesis/Data/model_ouput/training_stsbenchmark_'+model_name.replace("/", "-")+'-'+datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
model_save_path = '/mount/studenten5/projects/fanze/masterarbeit_data/model_output/training_stsbenchmark_'+model_name.replace("/", "-")+'-'+datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# 获取文件夹中的所有文件
all_files = os.listdir(debates_path)

# 筛选出CSV文件
csv_files = [file for file in all_files if file.endswith('.csv')]
samples = []
def skip_argument(text):
    # 定义正则表达式
    pattern = r"-> See \d+(\.\d+)+\."

    # 使用re.match进行匹配
    return bool(re.match(pattern, text))

# 逐个读取CSV文件
for file in csv_files:
    file_path = os.path.join(debates_path, file)
    # 读取CSV文件
    df = pd.read_csv(file_path)
    print(file_path)
    # print(file_path)

    # 按行处理数据
    files_has_0_distance = []
    for index, row in df.iterrows():
        # 假设CSV文件中有三列：name, grade, 和第三列（这里称其为column3）
        score =  1/ float(row['distance']) # Normalize score to range 0 ... 1
        inp_example = InputExample(texts=[row['content_1'], row['content_2']], label=score)
        print(score, row['content_1'], row['content_2'])
        if float(row['distance']) != 0 and not skip_argument(row['content_1']) and not skip_argument(row['content_2']) and float(row['distance']) <= 5:
            score =  1/ float(row['distance']) # Normalize score to range 0 ... 1
            inp_example = InputExample(texts=[row['content_1'], row['content_2']], label=score)
            # print(score, row['content_1'], row['content_2'])

            samples.append(inp_example)
        elif float(row['distance']) == 0:
            files_has_0_distance.append(file_path)

            print(file_path, row['distance'])
            raise ValueError(f"Invalid 'distance' value at row {index}: distance cannot be zero.")
file_name = "files_has_0_distance.txt"

# 使用 'with' 语句打开文件进行写入，确保文件最后会被正确关闭
with open(file_name, "w") as file:
    # 遍历列表，写入每一行
    for line in files_has_0_distance:
        file.write(line + "\n")  # "\n" 是换行符

# print(samples, type(samples))


random.shuffle(samples)
shuffled_samples = samples
# print(shuffled_samples, type(shuffled_samples))

train_ratio = 0.6
dev_ratio = 0.2
test_ration = 0.2

train_samples.append(inp_example)
train_data, temp_data = train_test_split(shuffled_samples, test_size=(1 - train_ratio))
dev_data, test_data = train_test_split(temp_data, test_size=0.5)

    # print(train_samples)
# print(type(dev_data[0]))

train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=train_batch_size)

train_dataloader = DataLoader(train_data, shuffle=True, batch_size=train_batch_size)
train_loss = losses.CosineSimilarityLoss(model=model)

logging.info("Read STSbenchmark dev dataset")
# evaluator = EmbeddingSimilarityEvaluator.from_input_examples(dev_samples, name='sts-dev')
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

