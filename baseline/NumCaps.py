import math

from baseline.BaseModel import BaseModel

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