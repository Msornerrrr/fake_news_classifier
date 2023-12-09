import argparse
import time

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from util.data_loader import load_data, split_data

OPT = None
default_alpha_values = [1000,100,10,1,0.1,1e-2,1e-3,1e-4,1e-5,1e-6,1e-7]

def main():
    start_time = time.time()
    
    # Load the data
    X, y = load_data(dataset=OPT.dataset)
    X_train, y_train, X_dev, y_dev, X_test, y_test = split_data(X, y, train_percent=70, dev_percent=10)

    # Convert the 'title' and 'text' fields from training data into a matrix of token counts
    print("Vectorizing the data...")
    vectorizer = CountVectorizer(stop_words='english')
    X_train_counts = vectorizer.fit_transform(X_train[:, 0] + ' ' + X_train[:, 1])
    X_dev_counts = vectorizer.transform(X_dev[:, 0] + ' ' + X_dev[:, 1])
    X_test_counts = vectorizer.transform(X_test[:, 0] + ' ' + X_test[:, 1])
    print("Done vectorizing.\n")

    stores = []
    for alpha in OPT.alpha:
        # Train the Naive Bayes classifier
        print(f"Training Naive Bayes classifier with alpha={alpha}...")
        clf = MultinomialNB(alpha=alpha)
        clf.fit(X_train_counts, y_train)
        print("Done training.")

        # Predict on the development set
        y_pred = clf.predict(X_dev_counts)

        # Evaluate the model
        accuracy = accuracy_score(y_dev, y_pred)
        stores.append((accuracy, clf, alpha))

        print(f"Accuracy on the development set: {accuracy:.4f}\n")

    best_models, best_models_alpha = max(stores, key=lambda x: x[0])[1:]
    print(f"Best model has alpha={best_models_alpha}")

    print("Predicting on the test set...")
    y_test_pred = best_models.predict(X_test_counts)
    accuracy = accuracy_score(y_test, y_test_pred)
    print(f"Accuracy on the test set: {accuracy:.4f}")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_test_pred))

    print(f"Total time taken: {time.time() - start_time:.2f} seconds")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Naive Bayes Classifier for Fake News Detection')
    parser.add_argument('--alpha', nargs='+', type=float, help='List of alpha values for hyperparameter tuning', default=default_alpha_values)
    parser.add_argument('--dataset', '-d', help='Choose dataset to use', choices=[1, 2, 3], type=int, default=2)
    OPT = parser.parse_args()
    
    main()
