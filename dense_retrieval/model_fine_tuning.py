import json, torch, time, torch, math, pickle
import pandas as pd

from sentence_transformers import SentenceTransformer, losses
from sentence_transformers.readers import InputExample

from torch.utils.data import DataLoader


def prepare_train_set(corpus: dict, train: pd.DataFrame, queries: dict, qrels_path: str) -> list[InputExample]:
    """
    Prepare the train set
    """

    count = 0
    train_data = []; test_data = []
    
    for index, row in train.iterrows():  
        query_id = row['query_id']; movie_id = row['movie_id']; label = float(row['label'])
        if query_id not in queries or movie_id not in corpus:
            count += 1
            continue
        else:
            query = queries[query_id]
            film_text = corpus[movie_id]
            train_data.append(InputExample(texts=[query, film_text], label=label))
    

    print(f"Had to skip {count} pairs")
    
    return train_data


def train_model(train_examples: list[InputExample], model_path: str, qrels_path: str):
    """
    Train the model
    """

    model = SentenceTransformer('all-MiniLM-L6-v2', device='cuda')

    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=64)
    train_loss = losses.CosineSimilarityLoss(model)

    model.fit(train_objectives=[(train_dataloader, train_loss)],
          epochs = 2,
          evaluation_steps = 1000,
          warmup_steps = math.ceil(len(train_dataloader) * 4 * 0.1))
    
    model.save(model_path)
    
    return model


def encode_test_set(model, corpus: dict, queries: dict, test: pd.DataFrame, test_path: str):
    """
    With the finetuned model we will encode the test set and store it
    """
    count = 0
    test_data = []
    test_labels = []
    for index, row in test.iterrows():  
        query_id = row['query_id']; movie_id = row['movie_id']; label = float(row['label'])
        if query_id not in queries or movie_id not in corpus:
            count += 1
            continue
        else:
            query = queries[query_id]
            test_data.append(query)
            test_labels.append(movie_id)

    pool = model.start_multi_process_pool()
    test_embeddings = model.encode_multi_process(test_data, pool)

    with open(test_path, "wb") as fOut:
        pickle.dump({'sentences': test_data, 'embeddings': test_embeddings, 'labels': test_labels}, fOut, protocol=pickle.HIGHEST_PROTOCOL)

    return test_embeddings    


def encode_corpus(corpus: dict, model_path: str, embedings_path: str):
    """
    Encode the corpus using SentenceTransformer
    """

    model = SentenceTransformer(model_path)
    pool = model.start_multi_process_pool()
    embeddings = model.encode_multi_process(corpus.values(), pool)

    with open(embedings_path, "wb") as fOut:
        pickle.dump({'sentences': corpus, 'embeddings': embeddings}, fOut, protocol=pickle.HIGHEST_PROTOCOL)


    return embeddings


def load_embeddings(embedings_path: str):
    """
    Load pre trained embeddings
    """
    
    with open(embedings_path, 'rb') as pkl:
        data = pickle.load(pkl)
        sentences = data['sentences']
        embeddings = data['embeddings']
        if 'labels' in data:
          labels = data['labels']
        else:
          labels = []

    return embeddings, sentences, labels