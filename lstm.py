import argparse
import time
import datetime
import json
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import gensim.downloader as api
from gensim.models import KeyedVectors

from data_loader import load_data, split_data

OPT = None
JSON_PATH = "models/models_hyperparameters.json"


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

class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, bidirectional, dropout, embedding_matrix=None):
        super(LSTMClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        if embedding_matrix is not None:
            self.embedding.weight.data.copy_(embedding_matrix)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=n_layers, bidirectional=bidirectional, dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text):
        embedded = self.dropout(self.embedding(text))
        output, (hidden, cell) = self.lstm(embedded)
        hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1))
        return self.fc(hidden.squeeze(0))

def train(model, iterator, optimizer, criterion, device):
    model.train()
    epoch_loss = 0
    for batch in iterator:
        text, labels = batch
        text, labels = text.to(device), labels.to(device)  # Move data to the device
        
        optimizer.zero_grad()
        predictions = model(text).squeeze(1)
        loss = criterion(predictions, labels.float())
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss / len(iterator)

def evaluate(model, iterator, criterion, device):
    model.eval()
    epoch_loss = 0
    all_preds = []
    all_y = []
    with torch.no_grad():
        for batch in iterator:
            text, labels = batch
            text, labels = text.to(device), labels.to(device)  # Move data to the device

            predictions = model(text).squeeze(1)
            loss = criterion(predictions, labels.float())
            epoch_loss += loss.item()
            all_preds.extend(torch.sigmoid(predictions).round().cpu().numpy())
            all_y.extend(labels.cpu().numpy())
    accuracy = accuracy_score(all_y, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_y, all_preds, average='binary')
    return epoch_loss / len(iterator), accuracy, precision, recall, f1


def load_hyperparameters(model_id, json_path):
    """
    Load hyperparameters for a given model ID from a JSON file.
    """
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"No hyperparameters file found at {json_path}")

    with open(json_path, 'r') as file:
        data = json.load(file)
        if model_id not in data:
            raise ValueError(f"No hyperparameters found for model ID {model_id}")

    return data[model_id]

def save_hyperparameters(model_id, hyperparameters, json_path):
    # Check if the JSON file already exists
    if os.path.exists(json_path):
        # Read the existing data
        with open(json_path, 'r') as file:
            data = json.load(file)
    else:
        data = {}

    # Update with new model's hyperparameters
    data[model_id] = hyperparameters

    # Write back to the JSON file
    with open(json_path, 'w') as file:
        json.dump(data, file, indent=4)


def main():
    start_time = time.time()

    # Load the data
    X, y = load_data(dataset=OPT.dataset)
    X_train, y_train, X_dev, y_dev, X_test, y_test = split_data(X, y, train_percent=70, dev_percent=10, shuffle=False)

    # Build vocabulary
    vocab = build_vocab(X_train)

    # Create datasets
    train_dataset = NewsDataset(X_train, y_train, vocab)
    dev_dataset = NewsDataset(X_dev, y_dev, vocab)
    test_dataset = NewsDataset(X_test, y_test, vocab)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=OPT.batch_size, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=OPT.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=OPT.batch_size, shuffle=False)
    print(f"Finished loading data & data preprocessing")

    # model hyperparameters
    vocab_size = len(vocab)  # Size of your vocabulary
    embedding_dim = 300  # Size of each embedding vector
    output_dim = 1  # Binary classification (fake or real)
    n_layers = 2  # Number of LSTM layers
    bidirectional = True  # Use a bidirectional LSTM

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    criterion = nn.BCEWithLogitsLoss()

    train_dev_performance = [] # List of dictionaries containing training and development performance

    # if we didn't specify any existing model to use, we train our model
    if not OPT.model:
        embedding_matrix = None
        # if we use exiting word_vectors, initialize embedding matrix
        if OPT.use_pretrained_embeddings:
            WORD_VEC = "word2vec-google-news-300" # using Google's pre-trained Word2Vec
            print(f"We initialize our word vector to {WORD_VEC}")

            word_vectors = api.load(WORD_VEC)
            embedding_matrix = torch.zeros((vocab_size, embedding_dim))
            for word, idx in vocab.get_stoi().items():
                try:
                    embedding_matrix[idx] = torch.from_numpy(word_vectors[word].copy())
                except KeyError:
                    pass  # For words not in the pre-trained model, embeddings remain zero

        model = LSTMClassifier(vocab_size, embedding_dim, OPT.hidden_dim, output_dim, n_layers, bidirectional, OPT.dropout_prob, embedding_matrix)
        model = model.to(device)

        print(f"Start training the model for {OPT.num_epochs} episodes")
        
        optimizer = optim.Adam(model.parameters(), lr=OPT.learning_rate)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

        for epoch in range(OPT.num_epochs):
            train_loss = train(model, train_loader, optimizer, criterion, device)
            dev_loss, dev_acc, dev_prec, dev_rec, dev_f1 = evaluate(model, dev_loader, criterion, device)
            train_dev_performance.append({
                'epoch': epoch+1,
                'train_loss': train_loss,
                'dev_loss': dev_loss,
                'dev_acc': dev_acc,
                'dev_prec': dev_prec,
                'dev_rec': dev_rec,
                'dev_f1': dev_f1
            })
            print(f'Epoch: {epoch+1}, Train Loss: {train_loss:.4f}, Val. Loss: {dev_loss:.4f}, Val. Acc: {dev_acc:.4f}, Precision: {dev_prec:.4f}, Recall: {dev_rec:.4f}, F1: {dev_f1:.4f}')
            
            # scheduler.step()
    # if we use an existing model that has been trained before
    else:
        model_id = OPT.model
        hyperparams = load_hyperparameters(model_id, JSON_PATH)

        # Use hyperparameters to configure the model
        embedding_dim = hyperparams['embedding_dim']
        hidden_dim = hyperparams['hidden_dim']
        n_layers = hyperparams.get('n_layers', 2)  # Default to 2 if not specified
        bidirectional = hyperparams.get('bidirectional', True)  # Default to True
        dropout = hyperparams.get('dropout_prob', 0.3)  # Default to 0.3

        model = LSTMClassifier(vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, bidirectional, dropout)
        model = model.to(device)

        PATH = f"models/{OPT.model}.pth"
        model.load_state_dict(torch.load(PATH, map_location=device))
        print(f"Using saved model at {PATH}")

    # Evaluate on test set
    test_loss, test_acc, test_prec, test_rec, test_f1 = evaluate(model, test_loader, criterion, device)
    test_performance = {
        'test_loss': test_loss,
        'test_acc': test_acc,
        'test_prec': test_prec,
        'test_rec': test_rec,
        'test_f1': test_f1
    }
    print(f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}, Precision: {test_prec:.4f}, Recall: {test_rec:.4f}, F1: {test_f1:.4f}')

    # if we decide to save our model
    if OPT.save:
        # create model unique id from current time
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        model_id = f"LSTM_{timestamp}"

        PATH = f"models/{model_id}.pth"
        torch.save(model.state_dict(), PATH)
        print(f"Model saved to {PATH}")

        hyperparameters = {
            'dataset': OPT.dataset,
            'learning_rate': OPT.learning_rate,
            'num_epochs': OPT.num_epochs,
            'batch_size': OPT.batch_size,
            'hidden_dim': OPT.hidden_dim,
            'dropout_prob': OPT.dropout_prob,
            'use_pretrained_embeddings': OPT.use_pretrained_embeddings,
            'embedding_dim': embedding_dim,
            'performance': {
                'train_dev': train_dev_performance,
                'test': test_performance
            }
        }
        
        save_hyperparameters(model_id, hyperparameters, JSON_PATH)
        print(f"Model hyperparameters saved to {JSON_PATH}")

    print(f"Total time taken: {time.time() - start_time:.2f} seconds")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='LSTM Classifier for Fake News Detection')
    parser.add_argument('--dataset', '-d', help='Choose dataset to use', choices=[1, 2, 3], type=int, default=3)
    parser.add_argument('--learning_rate', '-r', type=float, default=1e-3)
    parser.add_argument('--num_epochs', '-T', help='Choose number of episode', type=int, default=5)
    parser.add_argument('--batch_size', '-b', help='Choose number of batches', type=int, default=32)
    parser.add_argument('--hidden_dim', '-i', type=int, default=256)
    parser.add_argument('--dropout_prob', '-p', type=float, default=0.3)
    parser.add_argument('--use_pretrained_embeddings', action='store_true', help='Whether we use pretrained word vector')
    parser.add_argument('--model', '-m', help='Select path to existing model', type=str, default=None)
    parser.add_argument('--save', action='store_true', help='Whether we save our model')
    OPT = parser.parse_args()

    main()