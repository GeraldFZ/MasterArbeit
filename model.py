from debate_class import Debate
from debate_class import load_debates_from_folder
import random


debates = load_debates_from_folder('/Users/fanzhe/Desktop/master_thesis/Data/kialo_debatetree_data/results')

random_debate = random.sample(debates, 100)
random_argument = [random.sample(debate.arguments, 5) for debate in random_debate]


print(random_argument)
    # for argument in random_debate:

