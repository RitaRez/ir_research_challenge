
import json, os, time, logging, math, subprocess

from collections import OrderedDict
from multiprocessing import Pool

logging.basicConfig(filename='main.log', level=logging.INFO)



def encode_to_binary(word_id: int, docs: list[int, int]) -> bytearray:
    """Encodes a word and its document frequencies to binary"""
    
    bin_array = bytearray()
    bin_array += word_id.to_bytes(4, "big")
    bin_array += len(docs).to_bytes(4, "big")

    for doc in docs:
        try:
            bin_array += int(doc["id"]).to_bytes(4, "big")
            bin_array += int(doc["freq"]).to_bytes(2, "big")
        except TypeError:
            print(f"Error encoding {word_id, doc} to binary")

    return bin_array, len(bin_array)


def save_index_statistics(number_of_words: int, summed_n_docs: int, index_path: str):
    """Saves the index statistics to a file"""

    stats = {
        "Number of Lists: ": str(number_of_words),
        "Average List Size: ": '{0:.2f}'.format(summed_n_docs/number_of_words)
    }

    with open(index_path + "index_statistics.txt", 'w') as fp:        
        json.dump(stats, fp)


def builds_term_lexicon(word: str, lexicon, number_of_words: int, number_of_postings: int, byte_start: int, byte_end: int) -> int:
    """
    Builds the term lexicon
    """
    
    lexicon.write(word + " " + str(number_of_words) + " " + str(byte_start) + " " + str(byte_end) + " " + str(number_of_postings) + "\n")
    number_of_words += 1
    
    return number_of_words


def builds_last_index(output_file, doc: list, lexicon, number_of_words: int, summed_n_docs: int, byte_offset: int) -> (int, int, int):
    """
    Builds the last index
    """
    
    bin_array, doc_len = encode_to_binary(number_of_words, doc["docs"])
    output_file.write(bin_array)                                      
    
    number_of_postings = len(doc["docs"])

    summed_n_docs += number_of_postings

    number_of_words = builds_term_lexicon(
        word = doc['word'], 
        lexicon = lexicon, 
        number_of_words = number_of_words, 
        number_of_postings = number_of_postings,
        byte_start = byte_offset, 
        byte_end = byte_offset + doc_len
    )       
    
    return byte_offset + doc_len, number_of_words, summed_n_docs


def finishes_reading_index(f, output_file, lexicon, last_merge: bool, 
    number_of_words: int, summed_n_docs: int, byte_offset: int) -> (int, int, int):
    """
    Finishes reading the index and writes the rest of the lines to the output file
    """    

    for line in f.readlines():            
        doc = json.loads(line)
       
        if last_merge:
            byte_offset, number_of_words, summed_n_docs = builds_last_index(
                output_file, doc, lexicon, number_of_words, summed_n_docs, byte_offset
            )
        else:
            output_file.write(json.dumps(doc) + "\n")
                                                                   
    return number_of_words, summed_n_docs, byte_offset


def merger(index_path: str, first_file: str, second_file: str, last_merge=False):
    """
    Merges two files into one
    """    
    
    number_of_words = 0; byte_offset = 0; summed_n_docs = 0
    output_file_path = index_path + "inverted_index" if last_merge else index_path + "inverted_indexes/" + first_file + second_file
    
    logging.info(f"Merging {index_path}inverted_indexes/{first_file} and {index_path}inverted_indexes/{second_file} into {output_file_path}")

    if last_merge:
        output_file = open(output_file_path, 'wb')
    else :
        output_file = open(output_file_path, 'w')
        
    with open(index_path + "inverted_indexes/" + first_file, 'r') as ff, open(index_path + "inverted_indexes/" + second_file, 'r') as fs, open(index_path + "term_lexicon.txt", 'w') as lexicon:    
        
        ff_line = ff.readline()
        fs_line = fs.readline()
        while True:  
            if ff_line == "":                                                                         
                number_of_words, summed_n_docs, byte_offset = finishes_reading_index(fs, output_file, lexicon, last_merge, number_of_words, summed_n_docs, byte_offset)    
                break
            
            elif fs_line == "":                                                                    
                number_of_words, summed_n_docs, byte_offset = finishes_reading_index(ff, output_file, lexicon, last_merge, number_of_words, summed_n_docs,byte_offset)     
                break
            
            elif ff_line == "" and fs_line == "":                                                      
                break
            
            else:                                                                                   
                ff_doc = json.loads(ff_line); fs_doc = json.loads(fs_line)                           
                if  ff_doc['word'] == fs_doc['word']:     

                    ff_doc['docs'] += fs_doc['docs']

                    if last_merge:
                        byte_offset, number_of_words, summed_n_docs = builds_last_index(output_file, ff_doc, lexicon, number_of_words, summed_n_docs, byte_offset)
                    else:
                        output_file.write(json.dumps(ff_doc) + "\n")                                            
                                                                          
                    ff_line = ff.readline(); fs_line = fs.readline()                              
                
                elif ff_doc['word'] < fs_doc['word']:                                                

                    if last_merge:
                        byte_offset, number_of_words, summed_n_docs = builds_last_index(output_file, ff_doc, lexicon, number_of_words, summed_n_docs, byte_offset)
                    else:
                        output_file.write(json.dumps(ff_doc) + "\n")       

                    ff_line = ff.readline()                                                         
                
                else:                                                                                  

                    if last_merge:
                        byte_offset, number_of_words, summed_n_docs = builds_last_index(output_file, fs_doc, lexicon, number_of_words, summed_n_docs, byte_offset)
                    else:
                        output_file.write(json.dumps(fs_doc) + "\n")     

                    fs_line = fs.readline()                                                         
                
    output_file.close()

    os.remove(os.path.join(index_path + "inverted_indexes/", first_file))                                     
    os.remove(os.path.join(index_path + "inverted_indexes/", second_file))

                                                        
    if last_merge:
        save_index_statistics(number_of_words, summed_n_docs, index_path)


    time.sleep(1)


def clear_output_folders(index_path: str):
    """
    Deletes all files in the output folders
    """

    for file in os.listdir(index_path + "inverted_indexes/"):
        os.remove(os.path.join(index_path + "inverted_indexes/", file))

    for file in os.listdir(index_path + "shards/"):
        os.remove(os.path.join(index_path + "shards/", file))

    for file in os.listdir(index_path + "doc_indexes/"):
        os.remove(os.path.join(index_path + "doc_indexes/", file))

    for file in os.listdir(index_path + "doc_lens/"):
        os.remove(os.path.join(index_path + "doc_lens/", file))


    os.rmdir(index_path + "shards/")
    os.rmdir(index_path + "inverted_indexes/")
    os.rmdir(index_path + "doc_indexes/")
    os.rmdir(index_path + "doc_lens/")


def run_merger_thread_pool(index_path: str, number_of_threads: int):
    """
    Creates a thread pool for merging indexes
    """

    seconds = time.time()

    while True:
    
        notice_files = os.listdir(index_path + "inverted_indexes/")
        number_of_files = math.floor(len(notice_files)/2) * 2

        if len(notice_files) == 2:
            merger(index_path, notice_files[0], notice_files[1], True)
            break

        with Pool(number_of_threads) as pool:
            pool.starmap(merger, [(index_path, notice_files[i], notice_files[i+1], False) for i in range(0, number_of_files, 2)]) 


    subprocess.Popen(f"cat {index_path}doc_indexes/* > {index_path}document_index", shell=True)
    time.sleep(15)
    logging.info(f"Time to merge all files: {time.time() - seconds} seconds")
    subprocess.Popen(f"cat {index_path}doc_lens/* > {index_path}document_lens", shell=True)
    time.sleep(15)
    logging.info(f"Time to merge lens: {time.time() - seconds} seconds")

  
    clear_output_folders(index_path)

