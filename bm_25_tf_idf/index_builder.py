import json, os, time, logging, math, subprocess

from preprocesser import preprocesser, get_word_frequency
from collections import OrderedDict
from multiprocessing import  Pool


def encode_to_string(word: int, doc_freqs: list[int, int]) -> str:
    """
    Encodes a word and its document frequencies to a string
    """

    line = "{\"word\":\"" + str(word) + "\",\"docs\": ["
    for doc, freq in doc_freqs:
        line += "{\"id\":\"" + str(doc) + "\",\"freq\":\"" + str(freq) + "\"},"
    line = line[:-1]
    line += "]}\n" 

    return line


def download_inverted_index(index: OrderedDict, index_path: str, file_name: str):
    """
    Downloads the index of a shard to a file
    """

    seconds = time.time()
    with open(index_path + "inverted_indexes/" + file_name, 'w') as fp:
        for word in sorted(index):

            # line = encode_to_binary(word, index[word])
            line = encode_to_string(word, index[word])
            fp.write(line)

    logging.info(f"Time to save inverted index of shard {file_name}: {time.time() - seconds} seconds")


def download_doc_index(index: OrderedDict, index_path: str, file_name: str):
    """
    Downloads the doc index of a shard to a file
    """

    seconds = time.time()
    with open(index_path + "doc_indexes/" + file_name, 'w') as dindex, open(index_path + "doc_lens/" + file_name, 'w') as dstats:
        for doc in sorted(index):
            index[doc] = " ".join(index[doc])
            dindex.write(json.dumps({"id": doc, "text": index[doc]}) + "\n")
            dstats.write(json.dumps({"id": doc, "len": len(index[doc])}) + "\n")
            

        # json.dump(index, fp)



    logging.info(f"Time to save doc index of shard {file_name}: {time.time() - seconds} seconds")


def index_shard(file_name: str, index_path: str) -> None:
    """
    Indexes a shard of the corpus
    """

    seconds = time.time()
    inverted_index = OrderedDict()
    doc_index = OrderedDict()
    
    with open(index_path + "shards/" + file_name, 'r') as f:
        
        for line in f:
            doc = json.loads(line, strict=False)
            text_to_be_processed = doc['text'] + " " +  doc["title"] + " "+ " ".join(doc['keywords'])
            cleaned_text = preprocesser(text_to_be_processed)
            doc_index[doc['id']] = cleaned_text
            word_freq = get_word_frequency(cleaned_text)
            
            for word, freq in word_freq.items():	
                if word not in inverted_index:
                    inverted_index[word] = []
                inverted_index[word].append((doc['id'],freq))


    logging.info(f"Time to create index of shard {file_name}: {time.time() - seconds} seconds")
   
    download_inverted_index(inverted_index, index_path, file_name)
    download_doc_index(doc_index, index_path, file_name)


def run_indexer_thread_pool(index_path: str, number_of_threads: int, number_of_shards: int):
    """
    Creates a thread pool with the number of threads we want to use
    """
    
    notice_files = os.listdir(index_path + "shards/")
    count = 0
    while len(notice_files) < number_of_shards:
        notice_files = os.listdir(index_path + "shards/")
        time.sleep(1)
        count += 1

            
    logging.info(f"Found {len(notice_files)} shards")

    if not os.path.isdir(index_path + "inverted_indexes/"):
        os.mkdir(index_path + "inverted_indexes/") 

    if not os.path.isdir(index_path + "doc_indexes/"):
        os.mkdir(index_path + "doc_indexes/") 

    if not os.path.isdir(index_path + "doc_lens/"):
        os.mkdir(index_path + "doc_lens/") 

    with Pool(number_of_threads) as pool:
        pool.starmap(index_shard, [(file_name, index_path) for file_name in notice_files])