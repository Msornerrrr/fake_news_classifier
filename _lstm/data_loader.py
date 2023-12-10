import torch
from torch.utils.data import Dataset, DataLoader
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

from util.data_loader import load_data, split_data

def tokenize(text):
    """
    Tokenizes the text into words.
    """
    tokenizer = get_tokenizer("basic_english")
    return tokenizer(text.lower())

def build_vocab(X, max_size=10000):
    """
    Builds a vocabulary from the given text data.
    """
    tokenizer = get_tokenizer("basic_english")
    def yield_tokens(data_iter):
        for title, text in data_iter:
            yield tokenizer(title.lower())
            yield tokenizer(text.lower())

    return build_vocab_from_iterator(yield_tokens(X), specials=['<unk>', '<pad>'], max_tokens=max_size)

class NewsDataset(Dataset):
    def __init__(self, X, y, vocab, seq_length=500):
        self.X = X
        self.y = y
        self.vocab = vocab
        self.seq_length = seq_length
        self.tokenizer = get_tokenizer("basic_english")
    
    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        title_text_pair = self.X[idx]
        combined_text = ' '.join(title_text_pair)
        tokens = self.tokenizer(combined_text.lower())
        stoi = self.vocab.get_stoi()  # Get string-to-index mapping
        unk_idx = stoi['<unk>']  # Index for unknown token
        vectorized_text = [stoi.get(token, unk_idx) for token in tokens]  # Use get with default value for unknown tokens
        padded_text = vectorized_text[:self.seq_length] + [stoi['<pad>']] * max(0, self.seq_length - len(vectorized_text))
        return torch.tensor(padded_text, dtype=torch.long), torch.tensor(self.y[idx], dtype=torch.long)
    
def get_data_loader(batch_size, dataset=2):
    # Load the data
    X, y = load_data(dataset=dataset)
    X_train, y_train, X_dev, y_dev, X_test, y_test = split_data(X, y, train_percent=70, dev_percent=10, shuffle=False)

    # Build vocabulary
    vocab = build_vocab(X_train)

    # Create datasets
    train_dataset = NewsDataset(X_train, y_train, vocab)
    dev_dataset = NewsDataset(X_dev, y_dev, vocab)
    test_dataset = NewsDataset(X_test, y_test, vocab)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    print(f"Finished loading dataset {dataset} & data preprocessing")
    return train_loader, dev_loader, test_loader, vocab