import sys, resource, argparse, os, logging, json, os, math, subprocess, time

from collections import OrderedDict 
from multiprocessing import Pool, set_start_method, Lock, Manager
from preprocesser import preprocesser

set_start_method("spawn")

from document_matching import process_query

import sys
sys.modules['__main__'].__file__ = 'ipython'

def load_queries(queries_path: str) -> [int, [str]]:
    """
    Loads the queries from the queries.txt file
    """
    queries = []
    with open(queries_path, 'r') as f:
        for line_idx, query in enumerate(f):
            if line_idx == 0:
                continue
            query_idx, query_text = query.split(",")
            queries.append((query_idx, preprocesser(query_text)))

    return queries


def load_docs(index_path: str) -> dict[str, str]:
    """
    Loads the document index from the document_index file
    """

    doc_index = {}
    with open(index_path + "document_index", "r") as f:
        for line in f:
            documents = json.loads(line)
            id = int(documents["id"])
            text = documents["text"]
            doc_index[id] = text


    return doc_index



def load_doc_lens(index_path: str) -> dict[int, int]:
    """
    Loads the document lengths from the document_lengths file
    """

    doc_lens = {}
    with open(index_path + "document_lens", "r") as f:
        for line in f:
            doc_len = json.loads(line)
            id = int(doc_len["id"])
            length = int(doc_len["len"])
            doc_lens[id] = length

    return doc_lens


def main(index_path: str, queries_path: str, ranker: str, matching: str, number_of_threads: int, cut_number: int):
    """
    Your main calls should be added here
    """

    queries = load_queries(queries_path)
    doc_lens = load_doc_lens(index_path)

    results_folder = f"./results/{ranker}_output/"
    
    if not os.path.isdir(results_folder):
        os.mkdir(results_folder) 

    with Manager() as manager:

        index_lock = manager.Lock()
        lexicon_lock = manager.Lock()
        
        with Pool(number_of_threads) as pool:
            pool.starmap(process_query, [(index_path, query, doc_lens, matching, ranker, cut_number, index_lock, lexicon_lock) for query in queries])


    queries_path_without_extension = queries_path.split(".")[0]

    print(f"Saving results on results/{matching}_{ranker}_scores.csv")
    subprocess.Popen(f"cat {results_folder}* > results/{matching}_{ranker}_scores.csv", shell=True)

    time.sleep(10)

    for file in os.listdir(results_folder):
        os.remove(os.path.join(results_folder, file))
    os.rmdir(results_folder)


# $ python3 processor.py -i <INDEX> -q <QUERIES> -r <RANKER>

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')

    parser.add_argument('-i',dest='index_path',action='store',required=True,type=str)
    parser.add_argument('-q',dest='queries_path',action='store',required=True,type=str)
    parser.add_argument('-r',dest='ranker',action='store',required=True,type=str,default="BM25")
    parser.add_argument('-m',dest='matching',action='store',required=False,type=str,default="conjuntive_daat")
    parser.add_argument('-t',dest='number_of_threads',action='store',required=False,type=int,default=12)
    parser.add_argument('-k',dest='cut_number',action='store',required=False,type=int,default=100000)

    args = parser.parse_args()

    main(args.index_path, args.queries_path, args.ranker, args.matching, args.number_of_threads, args.cut_number)

