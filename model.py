from debate_class import Debate
from debate_class import load_debates_from_folder
import random
from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer, AutoModel
import torch

#Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask

tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

debates = load_debates_from_folder('/Users/fanzhe/Desktop/master_thesis/Data/kialo_debatetree_data/results')

random_debate = random.sample(debates, 2)
# argument = [argument for debate in random_debate for argument in debate.ar ]
sentences = []
for debate in random_debate:
    for argument in debate.arguments:
        # print(argument.index, argument.content)
        sentences.append(argument.content)
# print(sentences)

encoded_input = tokenizer(sentences, padding=True, truncation=True, max_length=128, return_tensors='pt')
with torch.no_grad():
    model_output = model(**encoded_input)

#Perform pooling. In this case, mean pooling
sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])


# embeddings = model.encode(sentences)
pairs = []
for sentence1, embeddings1 in zip(sentences, sentence_embeddings):
    for sentence2, embeddings2 in zip(sentences, sentence_embeddings):
        cosine_scores = util.cos_sim(embeddings1, embeddings2)
        for i in range(len(cosine_scores) - 1):
            for j in range(i + 1, len(cosine_scores)):
                pairs.append({'index': [i, j], 'score': cosine_scores[i][j]})
        print("pairs:", pairs)
        print("Sentence1:", sentence1)
        print("Sentence2:", sentence2)
        print("Embedding:", embeddings1, embeddings2)
        print("cosine_scores:", cosine_scores)


#Our sentences we like to encode

# #Sentences are encoded by calling model.encode()
#
# #Print the embeddings

# print(sentences)
