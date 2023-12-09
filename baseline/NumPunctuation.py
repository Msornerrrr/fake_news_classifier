import math

from baseline.BaseModel import BaseModel

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