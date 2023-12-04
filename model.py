from debate_class import load_debates_from_folder
import random
from sentence_transformers import SentenceTransformer, util, SentenceTransformer, SentencesDataset, InputExample, losses, models
from transformers import AutoTokenizer, AutoModel
import torch
import pandas as pd
import os
from torch.utils.data import DataLoader
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
import logging
import math

# model_name = SentenceTransformer('all-mpnet-base-v2')
model_name = 'distilbert-base-uncased'

train_batch_size = 16
num_epochs = 4
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
debates_path = '/Users/fanzhe/Desktop/master_thesis/Data/kialo_debatetree_data/csv_sample_for_model'
# # debates = load_debates_from_folder('/home/users0/fanze/masterarbeit/englishdebates')
#




# 获取文件夹中的所有文件
all_files = os.listdir(debates_path)

# 筛选出CSV文件
csv_files = [file for file in all_files if file.endswith('.csv')]

# 逐个读取CSV文件
for file in csv_files:
    file_path = os.path.join(debates_path, file)
    # 读取CSV文件
    df = pd.read_csv(file_path)
    print(file_path)

    # 按行处理数据
    for index, row in df.iterrows():
        # 假设CSV文件中有三列：name, grade, 和第三列（这里称其为column3）
        score =  1/ float(row['distance']) # Normalize score to range 0 ... 1
        inp_example = InputExample(texts=[row['content_1'], row['content_2']], label=score)
        print(score, row['content_1'], row['content_2'])

        train_samples.append(inp_example)

    # print(train_samples)

train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=train_batch_size)
train_loss = losses.CosineSimilarityLoss(model=model)

logging.info("Read STSbenchmark dev dataset")
# evaluator = EmbeddingSimilarityEvaluator.from_input_examples(dev_samples, name='sts-dev')

warmup_steps = math.ceil(len(train_dataloader) * num_epochs  * 0.1) #10% of train data for warm-up
logging.info("Warmup-steps: {}".format(warmup_steps))




#Our sentences we like to encode

# #Sentences are encoded by calling model.encode()
#
# #Print the embeddings

# print(sentences)
