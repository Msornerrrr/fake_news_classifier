import argparse
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from data_loader import load_data, split_data

OPT = None


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
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, bidirectional, dropout):
        super(LSTMClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
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

            predictions = model(batch[0]).squeeze(1)
            loss = criterion(predictions, labels.float())
            epoch_loss += loss.item()
            all_preds.extend(torch.sigmoid(predictions).round().cpu().numpy())
            all_y.extend(batch[1].cpu().numpy())
    accuracy = accuracy_score(all_y, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_y, all_preds, average='binary')
    return epoch_loss / len(iterator), accuracy, precision, recall, f1

def main():
    start_time = time.time()

    # Load the data
    X, y = load_data(dataset=OPT.dataset)
    X_train, y_train, X_dev, y_dev, X_test, y_test = split_data(X, y, train_percent=70, dev_percent=10)

    # Build vocabulary
    vocab = build_vocab(X_train)

    # Create datasets
    train_dataset = NewsDataset(X_train, y_train, vocab)
    dev_dataset = NewsDataset(X_dev, y_dev, vocab)
    test_dataset = NewsDataset(X_test, y_test, vocab)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    print(f"Finished loading data & data preprocessing")

    vocab_size = len(vocab)  # Size of your vocabulary
    embedding_dim = 100  # Size of each embedding vector
    hidden_dim = 256  # Number of features in the hidden state of the LSTM
    output_dim = 1  # Binary classification (fake or real)
    n_layers = 2  # Number of LSTM layers
    bidirectional = True  # Use a bidirectional LSTM
    dropout = 0.5  # Dropout for regularization

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = LSTMClassifier(vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, bidirectional, dropout)
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    criterion = nn.BCEWithLogitsLoss()
    print(f"Finished creating model, optimizer, and criterion")

    num_epochs = 5
    for epoch in range(num_epochs):
        train_loss = train(model, train_loader, optimizer, criterion, device)
        dev_loss, dev_acc, dev_prec, dev_rec, dev_f1 = evaluate(model, dev_loader, criterion, device)
        print(f'Epoch: {epoch+1}, Train Loss: {train_loss:.4f}, Val. Loss: {dev_loss:.4f}, Val. Acc: {dev_acc:.4f}, Precision: {dev_prec:.4f}, Recall: {dev_rec:.4f}, F1: {dev_f1:.4f}')

        # scheduler.step()

    # Evaluate on test set
    test_loss, test_acc, test_prec, test_rec, test_f1 = evaluate(model, test_loader, criterion, device)
    print(f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}, Precision: {test_prec:.4f}, Recall: {test_rec:.4f}, F1: {test_f1:.4f}')

    print(f"Total time taken: {time.time() - start_time:.2f} seconds")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='LSTM Classifier for Fake News Detection')
    parser.add_argument('--dataset', '-d', help='Choose dataset to use', choices=[1, 2, 3], type=int, default=3)
    OPT = parser.parse_args()

    main()