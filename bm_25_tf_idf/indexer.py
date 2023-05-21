import sys, argparse, os, logging, json, os, math, subprocess, time
from preprocesser import breaks_corpus, get_term_lexicon
from index_builder import index_shard, run_indexer_thread_pool
from index_merger import run_merger_thread_pool
from collections import OrderedDict 

logging.basicConfig(filename=f'main.log', level=logging.INFO)

#number of documents, number of tokens == number of inverted lists
# distribution of the number of postings per inverted list.

def print_indexer_statistics(index_path: str, corpus_path: str, full_time: int):
    """
    Prints the statistics of the indexer
    """
    
    with open(index_path + "index_statistics.txt", 'r+') as fp:
        stats = json.load(fp)
        stats["Index Size"] = '{0:.0f}'.format(int(str(subprocess.check_output(f"du -s {index_path}inverted_index", shell=True), 'utf-8').split("\t")[0])/1000)
        stats["Elapsed Time"] = '{0:.0f}'.format(full_time)
        stats["Number of Documents in Corpus"] = '{0:.0f}'.format(int(str(subprocess.check_output(f"wc -l {corpus_path}", shell=True), 'utf-8').split(" ")[0]))
    
        print(stats)

    with open(index_path + "index_statistics.txt", 'w') as fp:
        json.dump(stats, fp)


def main(corpus_path: str, index_path: str, verbose: bool, number_of_threads: int):
    """
    Your main calls should be added here
    """

    if not index_path.endswith("/"):
        index_path += '/'
    
    if not os.path.isdir(index_path):
        os.mkdir(index_path) 

    
    full_time = time.time()
    # Amount of pieces we can manage to process for each thread due to memory limit
    number_of_divisions = 64 
    
    logging.info(f"Indexing corpus {corpus_path} of size to an indexer in {index_path} with {number_of_threads} threads")
    logging.info(f"Were divinding the corpus in  {str(number_of_divisions)} pieces")

    seconds = time.time()
    shards_path = breaks_corpus(corpus_path, index_path, number_of_divisions)
    logging.info(f"Time to split the corpus: {time.time() - seconds} seconds")

    time.sleep(3)

    seconds = time.time()    
    run_indexer_thread_pool(index_path, number_of_threads, number_of_divisions)
    logging.info(f"Time to index all shards: {time.time() - seconds} seconds")

    seconds = time.time()    
    run_merger_thread_pool(index_path, number_of_threads)
    logging.info(f"Time to merge all indexes: {time.time() - seconds} seconds")


    full_time = time.time() - full_time
    logging.info(f"Total time to index corpus {corpus_path} of size {file_size} to an indexer in {index_path} with {number_of_threads} threads and {memory_limit} MB of memory: {full_time} seconds")
    logging.info(f"------------------------------------------------------------------------------------------------------------------------------------")


    print_indexer_statistics(index_path, corpus_path, full_time)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')

    parser.add_argument('-c',dest='corpus_path',action='store',required=True,type=str)
    parser.add_argument('-i',dest='index_path',action='store',required=True,type=str)
    parser.add_argument('-v',dest='verbose',action='store',required=False,type=bool,default=False)
    parser.add_argument('-t',dest='number_of_threads',action='store',required=False,type=int,default=8)

    args = parser.parse_args()
    main(args.corpus_path, args.index_path, args.verbose, args.number_of_threads)


