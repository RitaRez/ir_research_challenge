import json, os, math, subprocess, re, nltk, time, logging

from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
nltk.download('punkt')


def preprocesser(text: str) -> list:
    """
    Does stemming and removes stopwords and punctuation
    """
    
    snow_stemmer = SnowballStemmer(language='english')
    
    text = re.sub(r'\n|\r', ' ', text)       #Removes breaklines
    text = re.sub(r'[^\w\s]', ' ', text)       #Removes punctuation
    words = word_tokenize(text.lower())       #Tokenizes the text

    filtered_sentence = []
    for w in words:
        if w not in stop_words:
            filtered_sentence.append(snow_stemmer.stem(w))

    return filtered_sentence


def get_word_frequency(cleaned_text: list) -> dict:
    """
    Iterates through the text and returns a dictionary with the word frequency
    """
    
    word_freq = dict()
    for word in cleaned_text:
        if word not in word_freq:
            word_freq[word] = 0 
        word_freq[word] += 1

    return word_freq


def breaks_corpus(corpus_path: str, index_path: str, number_of_divisions: int) -> str:
    """
    Will read the jsonl file and break it into smaller chunks
    """

    # number_of_documents = 4641784
    number_of_documents = int(str(subprocess.check_output(f"wc -l {corpus_path}", shell=True), 'utf-8').split(" ")[0])
    size_of_division = math.ceil(number_of_documents / number_of_divisions)
    
    if not os.path.isdir(index_path + "shards/"):
        os.mkdir(index_path + "shards/") 

    subprocess.Popen(f'split --lines={size_of_division} --numeric-suffixes --suffix-length=2 {corpus_path} {index_path + "shards/"}t', shell=True)

    return index_path + "shards/"
          

def get_term_lexicon(corpus_path: str, index_path):
    """
    Will read the jsonl file and break it into smaller chunks
    """

    term_lexicon = dict()
    number_of_words = 0

    with open(corpus_path, 'r') as f:
        for line in f:
            doc = json.loads(line)
            cleaned_text = preprocesser(doc['text'])
            for word in cleaned_text:
                if word not in term_lexicon:
                    number_of_words += 1
                    term_lexicon[word] = number_of_words


    with open(index_path + "term_lexicon.json", 'w') as f:
        for word, id in term_lexicon.items():
            f.write(f"{word}:{id}")

    return term_lexicon, number_of_words
    
