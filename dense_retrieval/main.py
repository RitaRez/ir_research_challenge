import os
from utils import *
from model_fine_tuning import *
from dense_retriever import *

path = "../../"
qrels_path = path + "Movies/qrels.txt"
corpus_path = path + "Movies/documents.json"
queries_path = path + "Movies/queries.json"

test_embeddings_path = path + 'Embeddings/test_embeddings.pkl'
model_path = path + 'Models/'
corpus_embeddings_path = path + 'Embeddings/corpus_embeddings.pkl'



def main():

    corpus = read_corpus(corpus_path)
    queries = read_queries(queries_path)

    if not os.path.isfile(path + 'Models/pytorch_model.bin'):

        print("Fine Tuning Model")
        train, test = split_qrels_train_test(qrels_path)
        train_examples = prepare_train_set(corpus, train, queries, qrels_path)    

        model = train_model(
            train_examples = train_examples, 
            model_path = model_path,
            qrels_path = qrels_path
        )

        test_embeddings = encode_test_set(model, corpus, queries, test, test_embeddings_path)

    else:
        print("Loading Model")
        model = SentenceTransformer(model_path)


    if not os.path.isfile(corpus_embeddings_path):
        print("Generating Embeddings...")
        corpus_embeddings = encode_corpus(
            corpus = corpus, 
            model_path = model_path,
            embedings_path = corpus_embeddings_path
        ) 
        test_embeddings, test_sentences, test_labels = load_embeddings(test_embeddings_path)
    else:
        print("Loading Embeddings...")
        corpus_embeddings, corpus_sentences, labels = load_embeddings(corpus_embeddings_path)
        test_embeddings, test_sentences, test_labels = load_embeddings(test_embeddings_path)


    p = build_index(corpus_embeddings, numerical_id_list)
    labels, distances = retrieve_from_index(p, test_embeddings, 10)

main()