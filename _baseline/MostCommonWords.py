from collections import Counter
import numpy as np

from _baseline.BaseModel import BaseModel

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