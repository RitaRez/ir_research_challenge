all:
    echo "usage: make [install|clean|run]"

install:
    pip3 install -r requirements.txt

clean:
    rm -rf data/index.*

clean-tmp:
    rm -rf data/index.*.busy
    rm -rf data/index.*.tmp

run:
    python3 indexer.py -m 2024 -c data/corpus.jsonl -i data/index.se

test:
    python3.11 bm_25_tf_idf/indexer.py  -c data/corpus.jsonl -i indexer -v True -t 12

load:
    cp -r data-backup/* data/

get:
    cat data/corpus.jsonl | grep '^{"id": "${DOC_ID}"'