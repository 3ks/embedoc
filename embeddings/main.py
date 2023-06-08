import os
import re
from pathlib import Path

import numpy as np
import tensorflow_hub as hub
from sklearn.neighbors import NearestNeighbors

recommender = None


def preprocess(text):
    text = text.replace('\n', ' ')
    text = re.sub('\s+', ' ', text)
    return text


def read_markdown_file(file_path):
    with open(file_path, "r", encoding="utf-8", errors='ignore') as file:
        text = file.read()
    return text


def text_to_chunks(texts, word_length=150):
    text_toks = [t.split(' ') for t in texts]
    chunks = []

    for idx, words in enumerate(text_toks):
        for i in range(0, len(words), word_length):
            chunk = words[i: i + word_length]
            if (
                    (i + word_length) > len(words)
                    and (len(chunk) < word_length)
                    and (len(text_toks) != (idx + 1))
            ):
                text_toks[idx + 1] = chunk + text_toks[idx + 1]
                continue
            chunk = ' '.join(chunk).strip()
            chunks.append(chunk)
    return chunks


class SemanticSearch:
    def __init__(self):
        self.use = hub.load('./use')
        self.fitted = False

    def fit(self, data, batch=1000, n_neighbors=5):
        self.data = data
        self.embeddings = self.get_text_embedding(data, batch=batch)
        n_neighbors = min(n_neighbors, len(self.embeddings))
        self.nn = NearestNeighbors(n_neighbors=n_neighbors)
        self.nn.fit(self.embeddings)
        self.fitted = True

    def __call__(self, text, return_data=True):
        inp_emb = self.use([text])
        neighbors = self.nn.kneighbors(inp_emb, return_distance=False)[0]

        if return_data:
            return [self.data[i] for i in neighbors]
        else:
            return neighbors

    def get_text_embedding(self, texts, batch=1000):
        embeddings = []
        for i in range(0, len(texts), batch):
            text_batch = texts[i: (i + batch)]
            emb_batch = self.use(text_batch)
            embeddings.append(emb_batch)
        embeddings = np.vstack(embeddings)
        return embeddings


def load_recommender(directory, save_embeddings_path):
    global recommender
    if recommender is None:
        recommender = SemanticSearch()

    paths = [p for p in Path(directory).rglob("*.md") if p.is_file()]
    all_texts = []

    for path in paths:
        text = read_markdown_file(path)
        text = preprocess(text)
        all_texts.append(text)

    chunks = text_to_chunks(all_texts)
    recommender.fit(chunks)

    # Save embeddings to file
    np.save(save_embeddings_path, recommender.embeddings)

    return 'Corpus Loaded.'


# Sample usage
directory = "/app/docs"
save_embeddings_path = "/app/embeddings.npy"
load_recommender(directory, save_embeddings_path)