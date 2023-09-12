from debate_class import Debate
from debate_class import load_debates_from_folder
import random
from sentence_transformers import SentenceTransformer


debates = load_debates_from_folder('/Users/fanzhe/Desktop/master_thesis/Data/kialo_debatetree_data/results')

random_debate = random.sample(debates, 2)
# argument = [argument for debate in random_debate for argument in debate.ar ]
sentences = []
for debate in random_debate:
    for argument in debate.arguments:
        # print(argument.index, argument.content)
        sentences.append(argument.content)
print(sentences)




model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(sentences)
for sentence, embedding in zip(sentences, embeddings):
    print("Sentence:", sentence)
    print("Embedding:", embedding)
    print("")


#Our sentences we like to encode

# #Sentences are encoded by calling model.encode()
#
# #Print the embeddings

# print(sentences)
