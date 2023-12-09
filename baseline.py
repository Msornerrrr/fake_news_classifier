import argparse
import time

from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from util.data_loader import load_data, split_data
from _baseline.MostCommonWords import MostCommonWords
from _baseline.NumPunctuation import NumPunctuation
from _baseline.NumCaps import NumCaps

OPT = None

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
    parser.add_argument('--method', '-m', help='Choose baseline method to run', choices=['MostCommonWords', 'NumPunctuation', 'NumCaps'], required=True)
    parser.add_argument('--dataset', '-d', help='Choose dataset to use', choices=[1, 2, 3], type=int, default=2)
    OPT = parser.parse_args()

    main()
    