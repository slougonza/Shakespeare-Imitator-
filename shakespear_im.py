import nltk
nltk_data_path = nltk.download('shakespeare')
from nltk.lm.preprocessing import pad_both_ends
from sklearn.feature_extraction.text import CountVectorizer
from nltk.lm import MLE
from string import punctuation
my_punctuation = punctuation.replace("'", "")


def get_data():
    """
    Load text data and produce a list of token lists
    """
    
    sentences = []

    a_file = open("THE_SONNETS.txt", "r")

    list_of_lists = []
    for line in a_file:
        stripped_line = line.strip().lower()
        no_punc = stripped_line.translate(str.maketrans("", "", my_punctuation))
        line_list = no_punc.split()
        list_of_lists.append(line_list)
        sentences = [x for x in list_of_lists if len(x)>=2]
        

    a_file.close()
    
    return sentences

def build_vocab(sentences):
    """
    Take a list of sentences and return a vocab
    """
    
    flat_list = [item for sublist in sentences for item in sublist]
    set_words = set(flat_list)
    list_words = list(set_words)
    list_words.append('<s>')
    list_words.append('</s>')
    return list_words 



def build_ngrams(n, sentences):
    """
    Take a list of unpadded sentences and create all n-grams as specified by the argument "n" for each sentence
    """
    all_ngrams = []
    for line in sentences:
        sent = list(pad_both_ends(line, n))
        all_ngrams.append(list(nltk.everygrams(sent, min_len=n, max_len=n)))
    
    return all_ngrams


def train_ngram_lm(n):
    """
    Train a n-gram language model as specified by the argument "n"
    """
    lm = MLE(n)
    lm.fit(build_ngrams(n, get_data()), build_vocab(get_data()))    
    
    return lm


# Every time it runs, a different sonnet is written. 
n = 2
num_lines = 14
num_words_per_line = 8
text_seed = ["<s>"] * (n - 1)

lm = train_ngram_lm(n)

sonnet = []
while len(sonnet) < num_lines:
    while True:  # keep generating a line until success
        try:
            line = lm.generate(num_words_per_line, text_seed=text_seed)
        except ValueError:  # Ccapture exceptions
            continue
        else:
            line = [x for x in line if x not in ["<s>", "</s>"]]
            sonnet.append(" ".join(line))
            break


print("\n".join(sonnet))

