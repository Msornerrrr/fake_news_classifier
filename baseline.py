from collections import Counter
from tracemalloc import start
import numpy as np
import math
import argparse
import time

from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from data_loader import load_data, split_data

OPT = None


class BaseModel:
    def train(self, X, y) -> None:
        pass

    def predict(self, X):
        pass


class MostCommonWords(BaseModel):
    def __init__(self, top_n=10, title_weight=0.7, text_weight=0.3) -> None:
        self.top_n = top_n
        self.title_weight = title_weight
        self.text_weight = text_weight

        self.true_common_words_title = set()
        self.fake_common_words_title = set()
        self.true_common_words_text = set()
        self.fake_common_words_text = set()

    def train(self, X, y):
        # Extract titles and texts from X
        titles = X[:, 0]
        texts = X[:, 1]

        # Tokenize titles and texts
        tokenized_titles = [title.lower().split() for title in titles]
        tokenized_texts = [text.lower().split() for text in texts]

        # Separate titles and texts based on labels
        true_titles = [title for title, label in zip(tokenized_titles, y) if label == 1]
        fake_titles = [title for title, label in zip(tokenized_titles, y) if label == 0]
        true_texts = [text for text, label in zip(tokenized_texts, y) if label == 1]
        fake_texts = [text for text, label in zip(tokenized_texts, y) if label == 0]

        # Count occurrences for titles and texts
        self._count_occurrences(true_titles, self.true_common_words_title)
        self._count_occurrences(fake_titles, self.fake_common_words_title)
        self._count_occurrences(true_texts, self.true_common_words_text)
        self._count_occurrences(fake_texts, self.fake_common_words_text)

    def _count_occurrences(self, tokenized_data, common_word_set):
        word_counts = Counter(word for data in tokenized_data for word in data)
        common_word_set.update([word for word, _ in word_counts.most_common(self.top_n)])

    def predict(self, X):
        titles = X[:, 0]
        texts = X[:, 1]
        tokenized_titles = [title.lower().split() for title in titles]
        tokenized_texts = [text.lower().split() for text in texts]

        predictions = []
        for title, text in zip(tokenized_titles, tokenized_texts):
            title_true_count = sum(word in self.true_common_words_title for word in title)
            title_fake_count = sum(word in self.fake_common_words_title for word in title)
            text_true_count = sum(word in self.true_common_words_text for word in text)
            text_fake_count = sum(word in self.fake_common_words_text for word in text)

            # Combine evaluations using a weighted sum
            title_score = title_true_count - title_fake_count
            text_score = text_true_count - text_fake_count
            score = self.title_weight * title_score + self.text_weight * text_score
            predictions.append(1 if score > 0 else 0)
        return np.array(predictions)


class NumPunctuation(BaseModel):
    def __init__(self, punctuations=['!', '?', '#']) -> None:
        """
        Initialize the ExclamationPoints classifier.
        This classifier predicts news authenticity based on the frequency of exclamation points and question marks in the title.
        """
        self.decision_boundary = -1
        self.punctuations = punctuations

    def train(self, X, y):
        """
        Train the classifier using the provided data.

        Parameters:
        - X (array-like): Features matrix, where the first column contains the titles of the news articles.
        - y (array-like): True labels for the data.

        Note:
        The classifier calculates the average frequency of exclamation points and question marks in genuine and fake articles.
        It then sets a decision boundary based on these averages.
        """
        
        # Extract titles from X
        titles = X[:, 0]

        # Calculate exclamation point and question mark percentages for each title
        punctuation_percentages = [sum(title.count(punc) for punc in self.punctuations) / len(title) for title in titles]

        # Calculate average punctuation percentages for true and fake news
        true_percent_punct = sum(punct_percent for punct_percent, label in zip(punctuation_percentages, y) if label == 1) / y.sum()
        false_percent_punct = sum(punct_percent for punct_percent, label in zip(punctuation_percentages, y) if label == 0) / (len(y) - y.sum())

        # Set decision boundary
        self.decision_boundary = math.sqrt(true_percent_punct * false_percent_punct)

    def predict(self, X):
        """
        Predict the authenticity of news articles based on their titles.

        Parameters:
        - X (array-like): Features matrix, where the first column contains the titles of the news articles.

        Returns:
        - list: Predicted labels for the data.
        """
        
        assert 0 <= self.decision_boundary <= 1

        titles = X[:, 0]
        punctuation_percentages = [sum(title.count(punc) for punc in self.punctuations) / len(title) for title in titles]
        predictions = [(punct_percent < self.decision_boundary) for punct_percent in punctuation_percentages]
        return predictions


class NumCaps(BaseModel):
    """
    A naive classifier that predicts news authenticity based on the capitalization percentage in the title.
    """
    def __init__(self) -> None:
        """
        Initialize the NumCaps classifier.
        """
        self.decision_boundary = -1

    def train(self, X, y) -> None:
        """
        Train the NumCaps classifier.

        Parameters:
        - X: A 2D array where each row represents a news article and columns represent features (title, text, etc.).
        - y: A 1D array of labels (1 for true news, 0 for fake news).
        """
        # Extract titles from X
        titles = X[:, 0]

        # Calculate capitalization percentages for each title
        capitalization_percentages = [sum(1 for c in title if c.isupper()) / len(title) for title in titles]

        # Calculate average capitalization percentages for true and fake news
        true_percent_caps = sum(cap_percent for cap_percent, label in zip(capitalization_percentages, y) if label == 1) / y.sum()
        false_percent_caps = sum(cap_percent for cap_percent, label in zip(capitalization_percentages, y) if label == 0) / (len(y) - y.sum())

        # Set decision boundary
        self.decision_boundary = math.sqrt(true_percent_caps * false_percent_caps)

    def predict(self, X):
        """
        Predict the authenticity of news articles.

        Parameters:
        - X: A 2D array where each row represents a news article and columns represent features (title, text, etc.).

        Returns:
        - predictions: A list of predictions (1 for true news, 0 for fake news).
        """
        assert 0 <= self.decision_boundary <= 1

        titles = X[:, 0]
        capitalization_percentages = [sum(1 for c in title if c.isupper()) / len(title) for title in titles]
        predictions = [(cap_percent < self.decision_boundary) for cap_percent in capitalization_percentages]
        return predictions


def run_most_common_words(X_train, y_train, X_dev, y_dev, plot=True):
    """
    Run the MostCommonWords classifier.
    """
    best_settings = {
        'top_n': 0,
        'title_weight': 0,
        'text_weight': 0,
        'model': None
    }
    dev_results = []
    best_accuracy = 0
    for top_n in [10, 100, 1000]:
        for title_weight in range(1, 10, 1):
            model = MostCommonWords(top_n, title_weight, text_weight=10-title_weight)
            model.train(X_train, y_train)
            y_pred = model.predict(X_dev)
            accuracy = accuracy_score(y_dev, y_pred)

            dev_results.append({
                'top_n': top_n,
                'title_weight': title_weight,
                'text_weight': 10-title_weight,
                'accuracy': accuracy
            })
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_settings['top_n'] = top_n
                best_settings['title_weight'] = title_weight
                best_settings['text_weight'] = 10-title_weight
                best_settings['model'] = model

    if plot:
        # Plot for MostCommonWords
        df = pd.DataFrame(dev_results)
        plt.figure(figsize=(12, 6))
        ax1 = sns.barplot(x='top_n', y='accuracy', hue='title_weight', data=df)
        plt.ylim(0.5, 1.0)  # Set y-axis limits
        plt.title('Accuracy for MostCommonWords')
        plt.show()
        
    return best_settings


def run_num_punctuation(X_train, y_train, X_dev, y_dev, plot=True):
    """
    Run the NumPunctuation classifier.
    """
    best_settings = {
        'punctuations': [],
        'model': None
    }
    dev_results = []
    best_accuracy = 0
    for punc in [['!', '?', '#'], ['!', '?'], ['!'], ['?']]:
        model = NumPunctuation(punctuations=punc)
        model.train(X_train, y_train)
        y_pred = model.predict(X_dev)
        accuracy = accuracy_score(y_dev, y_pred)

        dev_results.append({
            'punctuations': ''.join(punc),
            'accuracy': accuracy
        })

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_settings['punctuations'] = punc
            best_settings['model'] = model

    if plot:
        # Plot for NumPunctuation
        df = pd.DataFrame(dev_results)
        plt.figure(figsize=(12, 6))
        ax2 = sns.barplot(x='punctuations', y='accuracy', data=df)
        plt.ylim(0.3, 0.7)  # Set y-axis limits
        plt.title('Accuracy for NumPunctuation')
    
        # Annotate each bar with its value
        for p in ax2.patches:
            ax2.annotate(f"{p.get_height():.2f}", 
                        (p.get_x() + p.get_width() / 2., p.get_height()), 
                        ha='center', va='center', 
                        xytext=(0, 9), 
                        textcoords='offset points')
        plt.show()
    
    return best_settings


def run_num_caps(X_train, y_train, X_dev, y_dev):
    """
    Run the NumCaps classifier.
    """
    model = NumCaps()
    model.train(X_train, y_train)
    y_pred = model.predict(X_dev)
    accuracy = accuracy_score(y_dev, y_pred)
    print(f"Dev Accuracy for NumCaps: {accuracy}")
    
    return {
        'model': model,
    }


def main():
    """
    Main function to load data, train the classifiers, and evaluate their performance.
    """
    start_time = time.time()

    X, y = load_data(dataset=OPT.dataset)    
    X_train, y_train, X_dev, y_dev, X_test, y_test = split_data(X, y, 70, 10)
    best_settings = None

    if OPT.method == 'MostCommonWords':
        best_settings = run_most_common_words(X_train, y_train, X_dev, y_dev)
    elif OPT.method == 'NumPunctuation':
        best_settings = run_num_punctuation(X_train, y_train, X_dev, y_dev)
    elif OPT.method == 'NumCaps':
        best_settings = run_num_caps(X_train, y_train, X_dev, y_dev)
    else:
        raise ValueError(f"Invalid method: {OPT.method}")

    best_model = best_settings['model']
    del best_settings['model']

    # Evaluate on test set
    y_pred_test = best_model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred_test)
    test_result = {
        'model': OPT.method,
        'test_accuracy': test_accuracy,
        **best_settings
    }

    # Print results
    df = pd.DataFrame([test_result])
    print(f"For model {OPT.method}: the test accuracy and best hyperparameters are:")
    print(df)

    print(f"Total time taken: {time.time() - start_time:.2f} seconds")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Baseline Classifiers for Fake News Detection')
    parser.add_argument('--method', '-m', help='Choose baseline method to run', choices=['MostCommonWords', 'NumPunctuation', 'NumCaps'])
    parser.add_argument('--dataset', '-d', help='Choose dataset to use', choices=[1, 2, 3], type=int, default=3)
    OPT = parser.parse_args()

    main()
    