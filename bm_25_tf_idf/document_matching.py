import os, time, json, math, logging, argparse, re

from queue import PriorityQueue
from multiprocessing import Lock

def decode_from_binary(byte_array: bytes):
    """
    Decodes a byte array into a word id and a list of documents
    """
    
    word_id = int.from_bytes(byte_array[:4], "big")
    number_of_docs = int.from_bytes(byte_array[4:8], "big")
    docs = []
    for i in range(number_of_docs):
        doc_id = int.from_bytes(byte_array[8 + i*6:12 + i*6], "big")
        freq = int.from_bytes(byte_array[12 + i*6:14 + i*6], "big")
        docs.append({"id": doc_id, "freq": freq})
    
    return word_id, docs


def retrieve_word_postings(index_path: str, byte_start: int, byte_end: int) -> dict:
    """
    Retrieves the inverted list for a given word id
    """

    with open(index_path, 'rb') as index_file:
        
        index_file.seek(byte_start)
        byte_array = index_file.read(byte_end - byte_start)
        
        retrieved_word_id, retrieved_docs = decode_from_binary(byte_array)

        return  retrieved_docs

def get_term_byte_start(term_lexicon_path: str, term: str) -> (str, str, str, str, str):
    """
    Binary search for word in lexicon file.
    """    
    
    with open(term_lexicon_path, 'r') as lexicon_file:
        lexicon = lexicon_file.read().splitlines()
        low = 0; high = len(lexicon) - 1

        while (low <= high):
            mid = math.floor((low + high)/2)
            current_term = lexicon[mid].split(" ")[0]
            
            if (term == current_term):
                return lexicon[mid].split(" ")
            
            elif (term > current_term):
                low = mid + 1
            
            else:
                high = mid - 1

    return [None] * 5


def get_inverted_list(index_path: str, term: str, index_lock: Lock, lexicon_lock: Lock) -> list:
    """
    Returns the inverted list for a given term
    """

    inverted_index_path = os.path.join(index_path, 'inverted_index')
    term_lexicon_path = os.path.join(index_path, 'term_lexicon.txt')

    with lexicon_lock:
        term, term_id, byte_start, byte_end, tfc = get_term_byte_start(term_lexicon_path, term)
        if (term == None):
            # print("Term not found in lexicon")
            return [], 1



    with index_lock:
        word_postings = retrieve_word_postings(inverted_index_path, int(byte_start), int(byte_end))

    return word_postings, tfc
    

def get_valid_documents(postings_for_all_tokens: list[list], matching: str) -> list[(str, int)]:
    """
    Returns a list of documents that contain all the terms in the query
    """

    cleaned_postings = []
    for posting in postings_for_all_tokens:
        cleaned_postings.append([int(doc["id"]) for doc in posting])

    if matching == "conjuntive_daat":
        return list(set.intersection(*map(set, cleaned_postings)))

    elif matching == "disjunctive_daat":
        return list(set.union(*map(set, cleaned_postings)))

    else:
        return []


def skip_forward_to_document(doc_id: str, term_posting: list[dict]) -> int:
    """
    Skips forward in the inverted list to the document with the given id
    """

    low = 0; high = len(term_posting) - 1
    
    while (low <= high):
        mid = math.floor((low + high)/2)
        current_term = term_posting[mid]['id']
        
        if (doc_id == current_term):
            return int(term_posting[mid]['freq'])
        
        elif (doc_id > current_term):
            low = mid + 1
        
        else:
            high = mid - 1

    return 0

        
def bm25(term_idf: float, term_frequency: float, doc_len: float, avg_doc_len: float, b = 0.75, k1 = 1.2) -> float:
    """
    This function implements the BM25 scoring function
    """

    return term_idf * ((term_frequency * (k1 + 1))/(term_frequency + k1 * (1 - b + b * (doc_len/avg_doc_len))))


def tf_idf(term_frequency: float, term_idf: float) -> float:
    """
    This function implements the TF-IDF scoring function
    """

    return term_idf * term_frequency


def daat(index_path: str, query: list[str], doc_lens: dict[int, int], k: int, matching: str, ranker: str, index_lock: Lock, lexicon_lock: Lock) -> PriorityQueue:
    """
    This function implements the DAAT algorithm
    """

    l = []
    idfs = {}
    r = PriorityQueue()
    avg_lens = sum(doc_lens.values())/len(doc_lens)

    for term in query:                                                            # For each term in que query
        term_postings, tfc = get_inverted_list(
            index_path, 
            term,
            index_lock, 
            lexicon_lock    
        )                  
        l.append(term_postings)
        idfs[term] = math.log10(len(doc_lens)/int(tfc))                           # We calculate its idf

    valid_documents = get_valid_documents(l, matching)                                      # So to be conjuctive we pick only the docs that have all query terms
    for doc_id in valid_documents:                                                # For each valid document
        
        sd = 0
        for idx, term_postings in enumerate(l):                                   # We go through each inverted list
            term = query[idx]                                                     # Get the term
            weight = skip_forward_to_document(doc_id, term_postings)              # Get number of times the term appears in the current document

            # Then calculate the score for the current document and current term
            current_score = bm25(idfs[term], weight, doc_lens[doc_id], avg_lens) if ranker == 'BM25' else tf_idf(weight, idfs[term])
            sd += current_score
            
        r.put((sd, doc_id))

        if r.qsize() == k:
            r.get()


    return r


def invert_pq_results(r: PriorityQueue()) -> list[(int, float)]:
    """
    Inverts the priority queue to return results in descending order
    """
   
    results = []
    while not r.empty():
        score, doc_id = r.get()
        results.append((doc_id, score))


    return results[::-1]


def save_query_ans(output_path: str, query_idx: str, r: PriorityQueue) -> None:
    """
    Saves the query results to a file
    """

    with open(output_path + query_idx, 'w') as f:       
        for doc_id, score in invert_pq_results(r):
            f.write(f"{query_idx},{doc_id},{score}\n")
            

def process_query(index_path: str, query: (str, [str]), doc_lens: dict[int, int], matching: str, ranker: str, cut: int, index_lock: Lock, lexicon_lock: Lock) -> None:
    """
    Process individual query
    """
    query_idx = query[0]
    query = query[1]

    
    r = daat(index_path, query, doc_lens, cut, matching, ranker, index_lock, lexicon_lock)
    save_query_ans(f"results/{ranker}_output/", query_idx, r)
