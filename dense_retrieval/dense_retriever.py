import json, torch, time, torch, hnswlib
import pandas as pd
import numpy as np

from sentence_transformers import SentenceTransformer
from torch.utils.data import DataLoader


def build_index(embeddings, ids):
    """
    Build the index for the embeddings with distance being cosine
    """

    dim = embeddings.shape[1]
    num_elements = embeddings.shape[0]

    data = embeddings.copy()

    # Declaring index
    p = hnswlib.Index(space = 'cosine', dim = dim)

    # Initializing index - the maximum number of elements should be known beforehand
    p.init_index(max_elements = num_elements, ef_construction = 200, M = 16)

    # Element insertion (can be called several times):
    p.add_items(data)

    # Controlling the recall by setting ef:
    p.set_ef(50) # ef should always be > k

    return p



def retrieve_from_index(p, query_embeddings: np.ndarray, k: int) -> list[list[int]]:
    """
    Retrieve the k nearest neighbours from the index
    """

    return p.knn_query(query_embeddings, k = k)

