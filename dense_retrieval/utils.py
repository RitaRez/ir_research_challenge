import json, torch, time, torch
import pandas as pd

from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer

def read_corpus(corpus_path: str) -> dict[str]:
    """
    Read the corpus from file_path
    """
    
    films_data = {}
    with open(corpus_path, 'r') as f:
        for line in f:    
            film = json.loads(line)            
            films_data[film['id']] =  film['text'] # film['title'] + ' ' + film['text'] + ' ' + film['genres'] + ' ' + film['startYear']

    return films_data


def read_queries(queries_path: str) -> dict[str]:
    """
    Read the queries from file_path
    """

    queries = {}
    with open(queries_path, 'r') as f:
        for line in f:    
            query = json.loads(line)            
            queries[query['id']] = query['title'] + ' ' + query['description']

    return queries


def split_qrels_train_test(qrels_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split the corpus into train and test
    """
    
    df = pd.read_csv(qrels_path, sep="\s+", names=["query_id", "_", "movie_id", "label"]) # For header names
    df.drop('_', axis=1, inplace=True)

    train, test = train_test_split(df, test_size=0.2)

    return train, test


def get_integer_ids_to_films(el: dict[str]) -> (dict[int,str], dict[str,int], list[int]):
    """
    HNSBWL needs integer ids so well convert our str ones to it
    """

    str_to_int_converter = {}
    int_to_str_converter = {}
    numerical_id_list = []

    for int_idx, (movie_id, movie) in enumerate(el.items()):
        str_to_int_converter[int_idx] = movie_id
        int_to_str_converter[movie_id] = int_idx
        numerical_id_list.append(int_idx)

    return str_to_int_converter, int_to_str_converter, numerical_id_list