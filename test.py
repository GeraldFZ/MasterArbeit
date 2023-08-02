import nltk
import os

def is_english(text):
    # 使用NLTK的句子分割器检测文本是否为英文
    try:
        sentences = nltk.sent_tokenize(text)
        for sentence in sentences:
            words = nltk.word_tokenize(sentence)
            for word in words:
                if not word.isalpha():
                    return False
    except LookupError:
        nltk.download('punkt')
    return True

def num_english_files(files_path):
    num = 0
    with open(files_path, 'r', encoding='utf-8') as file:
        text = file.read()
    if is_english(text):
        num += 1
    return num

folder_path = '/Users/fanzhe/Desktop/master_thesis/Data/kialo_debatetree_data/testsample'
txtfiles = [f for f in os.listdir(folder_path) if f.endswith('.txt')]

num_english = 0
for file_name in txtfiles:
    file_path = os.path.join(folder_path, file_name)
    print(file_path)
    num_english += num_english_files(file_path)
    print(file_path,num_english_files(file_path))

print("Number of English files:", num_english)
