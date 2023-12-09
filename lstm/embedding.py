import torch
import gensim.downloader as api
from gensim.models import KeyedVectors

def create_embedding_matrix(vocab, embedding_dim):
    WORD_VEC = "word2vec-google-news-300" # using Google's pre-trained Word2Vec
    vocab_size = len(vocab)  # Size of your vocabulary
    
    assert WORD_VEC in api.info()['models'].keys(), "Invalid word vector."
    assert embedding_dim == 300, "Embedding dimension must be 300."

    print(f"We initialize our word vector to {WORD_VEC}")

    word_vectors = api.load(WORD_VEC)
    embedding_matrix = torch.zeros((vocab_size, embedding_dim))
    for word, idx in vocab.get_stoi().items():
        try:
            embedding_matrix[idx] = torch.from_numpy(word_vectors[word].copy())
        except KeyError:
            pass  # For words not in the pre-trained model, embeddings remain zero