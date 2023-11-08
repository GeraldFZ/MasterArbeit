from debate_class import Debate
from debate_class import load_debates_from_folder
import random
from sentence_transformers import SentenceTransformer, util, SentenceTransformer, SentencesDataset, InputExample, losses
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

train_examples = []
# # debates = load_debates_from_folder('/Users/fanzhe/Desktop/master_thesis/Data/kialo_debatetree_data/englishdebates')
debates = load_debates_from_folder('/Users/fanzhe/Desktop/master_thesis/Data/kialo_debatetree_data/testsample_english')
# # debates = load_debates_from_folder('/home/users0/fanze/masterarbeit/englishdebates')
#
for debate in debates:
    print("topic:", debate.debate_topic)
    for count1, argument1 in enumerate(debate.arguments, start=1):
        for count2, argument2 in enumerate(debate.arguments, start=1):
            if count2 >= count1:
                train_examples.append(InputExample(texts=[argument1.content, argument2.content], label = next((item['relatedness'] for item in debate.Argument.distance_relatedness_compute(debate.arguments, count2) if item['index_1'] == argument1.index.split(".") and item['index_2'] == argument1.index.split(".")), None)))
print(train_examples)



random_debate = random.sample(debates, 1)
# argument = [argument for debate in random_debate for argument in debate.ar ]
for debate in random_debate:
    sentences = []
    sentences_content = []
    counter = 0

    for argument in debate.arguments:
        # print(argument.index, argument.content)
        sentences_content.append(argument.content)
        sentences.append(argument)
        counter += 1
# print(sentences)

    encoded_input = tokenizer(sentences_content, padding=True, truncation=True, max_length=128, return_tensors='pt')
    with torch.no_grad():
        model_output = model(**encoded_input)

    #Perform pooling. In this case, mean pooling
    sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])


# embeddings = model.encode(sentences)
    pairnum = 0
    for i, (sentence1, embeddings1) in enumerate(zip(sentences, sentence_embeddings),start=1):
        for j, (sentence2, embeddings2) in enumerate(zip(sentences, sentence_embeddings), start=1):
            cosine_scores = util.cos_sim(embeddings1, embeddings2)
            pairs = (sentence1, sentence2)
            if j>= i:
                pairnum += 1


                # for i in range(len(cosine_scores) - 1):
                #     for j in range(i + 1, len(cosine_scores)):
                #         pairs.append({'index': [i, j], 'score': cosine_scores[i][j]})
                # print("pairs:", pairs)
                print("Sentence1:", sentence1)
                print("Sentence2:", sentence2)
                print("Embedding:", embeddings1, embeddings2)
                print("cosine_scores:", cosine_scores)
                print("pairnum:",j, pairnum)
                print('counter:', counter)
                print(len(sentences), len(sentence_embeddings))

#Our sentences we like to encode

# #Sentences are encoded by calling model.encode()
#
# #Print the embeddings

# print(sentences)
