import numpy as np
import pandas as pd

def prepare_old_data():
    # Load the datasets
    true_data = pd.read_csv('data/True.csv')
    fake_data = pd.read_csv('data/Fake.csv')

    # Label the datasets
    true_data['label'] = 1
    fake_data['label'] = 0

    # Combine the datasets
    df = pd.concat([true_data, fake_data], axis=0).reset_index(drop=True)
    return df

def prepare_WEL_data():
    df = pd.read_csv('data/WELFake_Dataset.csv')
    df.dropna(subset = ['text', 'title'], inplace = True)
    return df

def load_data(dataset=1) -> tuple:
    df = None
    if dataset == 1:
        df = prepare_old_data()
    elif dataset == 2:
        df = prepare_WEL_data()
    elif dataset == 3:
        df = prepare_old_data()
        df2 = prepare_WEL_data()
        df = pd.concat([df, df2], axis=0).reset_index(drop=True)
    else:
        raise ValueError("Invalid option.")

    X = df[['title', 'text']].values
    y = df['label'].values

    assert X.shape[0] == y.shape[0], "The number of samples and labels must be equal."
    assert X.shape[1] == 2, "The number of features must be 2."

    return X, y

def split_data(X, y, train_percent: int, dev_percent: int, shuffle=True) -> tuple:
    """
    Split the data into training, development, and testing sets.

    Parameters:
    - X (np.array): Features matrix.
    - y (np.array): Labels vector.
    - train_percent (int): Percentage of the dataset to be used for training.
    - dev_percent (int): Percentage of the dataset to be used for development/validation.

    Returns:
    - tuple: Contains training, development, and testing data in the form (X_train, y_train, X_dev, y_dev, X_test, y_test).
    """

    # Ensure that the provided percentages sum up to 100 or less
    assert 0 <= train_percent <= 100, "Training percentage must be between 0 and 100."
    assert 0 <= dev_percent <= 100, "Development percentage must be between 0 and 100."
    assert train_percent + dev_percent <= 100, "The sum of the training and development percentages must be 100 or less."

    # Shuffle and split the data
    indices = np.arange(X.shape[0])

    if shuffle:
        np.random.shuffle(indices)

    train_end = int(train_percent / 100 * X.shape[0])
    dev_end = train_end + int(dev_percent / 100 * X.shape[0])

    X_train, y_train = X[indices[:train_end]], y[indices[:train_end]]
    X_dev, y_dev = X[indices[train_end:dev_end]], y[indices[train_end:dev_end]]
    X_test, y_test = X[indices[dev_end:]], y[indices[dev_end:]]

    return X_train, y_train, X_dev, y_dev, X_test, y_test
