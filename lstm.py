import argparse
import time
import datetime

import torch
import torch.optim as optim
from torch.nn import BCEWithLogitsLoss
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from util.json_io import load_json_with_key, add_json_with_key
from _lstm.embedding import create_embedding_matrix
from _lstm.plot import plot_model_info
from _lstm.data_loader import get_data_loader
from _lstm.model import LSTMClassifier

# Global variables
model_info_path = "models/models_info.json"

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


def train_model(OPT, device):
    # Load the data
    train_loader, dev_loader, test_loader, vocab = get_data_loader(OPT.batch_size, dataset=OPT.dataset)

    # model hyperparameters
    vocab_size = len(vocab)  # Size of your vocabulary
    embedding_dim = 300  # Size of each embedding vector
    hidden_dim = OPT.hidden_dim  # Size of the hidden layer output
    n_layers = 2  # Number of LSTM layers
    bidirectional = True  # Use a bidirectional LSTM
    dropout_prob = OPT.dropout_prob  # Dropout probability

    # initialize model
    model = LSTMClassifier(
        vocab_size, 
        embedding_dim, 
        hidden_dim,
        n_layers, 
        bidirectional,
        dropout_prob,
        create_embedding_matrix(vocab, embedding_dim) if OPT.use_pretrained_embeddings else None
    )
    model_info = {}
    criterion = BCEWithLogitsLoss()
    model = model.to(device)

    # train model
    print(f"Start training the model for {OPT.num_epochs} episodes")
    optimizer = optim.Adam(model.parameters(), lr=OPT.learning_rate)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    
    train_dev_performance = [] # List of dictionaries containing training and development performance

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

    # save model info
    model_info = {
        'dataset': OPT.dataset,
        'hyperparameters': {
            'batch_size': OPT.batch_size,
            'learning_rate': OPT.learning_rate,
            'num_epochs': OPT.num_epochs,
            'hidden_dim': OPT.hidden_dim,
            'dropout_prob': OPT.dropout_prob,
            'use_pretrained_embeddings': OPT.use_pretrained_embeddings,
            'embedding_dim': embedding_dim,
            'n_layers': n_layers,
            'bidirectional': bidirectional
        },
        'performance': {
            'train_dev': train_dev_performance,
            'test': []
        }
    }

    if not OPT.notest:
        # Evaluate on test set
        test_loss, test_acc, test_prec, test_rec, test_f1 = evaluate(model, test_loader, criterion, device)
        test_performance = {
            'test_datetime': datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
            'test_dataset': OPT.dataset,
            'test_loss': test_loss,
            'test_acc': test_acc,
            'test_prec': test_prec,
            'test_rec': test_rec,
            'test_f1': test_f1
        }
        model_info['performance']['test'].append(test_performance)
        print(f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}, Precision: {test_prec:.4f}, Recall: {test_rec:.4f}, F1: {test_f1:.4f}')

    # if we decide to save our model
    if OPT.save:
        # create model unique id from current time
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        model_id = f"LSTM_{timestamp}"

        # save model
        PATH = f"models/{model_id}.pth"
        torch.save(model.state_dict(), PATH)
        print(f"Model saved to {PATH}")
        
        add_json_with_key(model_info, model_info_path, model_id)
        print(f"Model info saved to {model_info_path}")

    # if we decide to plot our model performance
    if OPT.plot:
        plot_model_info(model_info)


def run_model(OPT, device):
    # Load model info & hyperparameters
    model_id = OPT.model
    model_info = load_json_with_key(model_info_path, model_id)
    print(f"Loaded model info for {model_id}")

    if not OPT.notest:
        criterion = BCEWithLogitsLoss() # Loss function
        _, _, test_loader, vocab = get_data_loader(OPT.batch_size, dataset=OPT.dataset)

        vocab_size = len(vocab)
        embedding_dim = model_info['hyperparameters']['embedding_dim']
        hidden_dim = model_info['hyperparameters']['hidden_dim']
        n_layers = model_info['hyperparameters']['n_layers']
        bidirectional = model_info['hyperparameters']['bidirectional']
        dropout = model_info['hyperparameters']['dropout_prob']

        # Use hyperparameters to create the model
        model = LSTMClassifier(vocab_size, embedding_dim, hidden_dim, n_layers, bidirectional, dropout)
        model = model.to(device)

        PATH = f"models/{OPT.model}.pth"
        model.load_state_dict(torch.load(PATH, map_location=device))
        print(f"Using saved model at {PATH}")

        # Evaluate on test set
        test_loss, test_acc, test_prec, test_rec, test_f1 = evaluate(model, test_loader, criterion, device)
        test_performance = {
            'test_datetime': datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
            'test_dataset': OPT.dataset,
            'test_loss': test_loss,
            'test_acc': test_acc,
            'test_prec': test_prec,
            'test_rec': test_rec,
            'test_f1': test_f1
        }
        model_info['performance']['test'].append(test_performance)
        print(f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}, Precision: {test_prec:.4f}, Recall: {test_rec:.4f}, F1: {test_f1:.4f}')

        model_id = OPT.model
        add_json_with_key(model_info, model_info_path, model_id)
        print(f"Model info saved to {model_info_path}")

    # if we decide to plot our model performance
    if OPT.plot:
        plot_model_info(model_info)

def main():
    # device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Parent parser for shared arguments
    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument('--dataset', '-d', help='Choose dataset to use', choices=[1, 2, 3], type=int, default=2)
    parent_parser.add_argument('--batch_size', '-b', help='Choose number of batches', type=int, default=32)
    parent_parser.add_argument('--notest', '-nt', action='store_true', help='Whether we don\'t evaluate on test set')
    parent_parser.add_argument('--plot', '-p', action='store_true', help='Whether we plot the model performance')

    # Main parser
    parser = argparse.ArgumentParser(description='LSTM Classifier for Fake News Detection')
    subparsers = parser.add_subparsers(dest='mode', help='Modes: train or run')

    # Subparser for training
    train_parser = subparsers.add_parser('train', parents=[parent_parser], help='Train a new model')
    train_parser.add_argument('--learning_rate', '-r', type=float, default=1e-3)
    train_parser.add_argument('--num_epochs', '-T', help='Choose number of episode', type=int, default=5)
    train_parser.add_argument('--hidden_dim', '-hd', type=int, default=256)
    train_parser.add_argument('--dropout_prob', '-dp', type=float, default=0.3)
    train_parser.add_argument('--use_pretrained_embeddings', action='store_true', help='Whether we use pretrained word vector')
    train_parser.add_argument('--save', '-s', action='store_true', help='Whether we save our model')

    # Subparser for running existing model
    run_parser = subparsers.add_parser('run', parents=[parent_parser], help='Run an existing model')
    run_parser.add_argument('--model', '-m', required=True, help='Path to the existing model')
    
    OPT = parser.parse_args()

    start_time = time.time()
    if OPT.mode == 'train':
        train_model(OPT, device)
    elif OPT.mode == 'run':
        run_model(OPT, device)
    else:
        print("Invalid mode")
        parser.print_help()

    print(f"Total time taken: {time.time() - start_time:.2f} seconds")

if __name__ == '__main__':
    main()