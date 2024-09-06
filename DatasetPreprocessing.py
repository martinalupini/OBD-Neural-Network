import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Splits the dataset in training, validation and testing set and standardizes the data
def preprocess(dataset: pd.DataFrame, label_name: str):
    preprocessed_dataset = dataset
    # removing lines with null values
    preprocessed_dataset = preprocessed_dataset.dropna(axis=0)

    # shuffling the dataset
    preprocessed_dataset = preprocessed_dataset.sample(frac=1).reset_index(drop=True)

    # removing the label column from dataset and saving it in another array
    labels_column = preprocessed_dataset[label_name].values
    preprocessed_dataset = preprocessed_dataset.drop(label_name, axis=1)

    # Standardizing the dataset
    scaler = StandardScaler()
    standardized_matrix = scaler.fit_transform(preprocessed_dataset)

    # creating a matrix with the rows of the dataset
    samples_matrix = standardized_matrix
    samples_number: int = samples_matrix.shape[0]
    train_set_samples = int(samples_number * 0.6)
    other_set_samples = int(samples_number * 0.2)
    valid_ind = train_set_samples + other_set_samples

    # dividing the dataset
    X_train, Y_train = samples_matrix[0:train_set_samples, :], labels_column[0:train_set_samples]
    X_train = X_train.reshape(train_set_samples, -1).T
    Y_train = Y_train.reshape(-1, train_set_samples)
    X_valid, Y_valid = samples_matrix[train_set_samples:valid_ind, :], labels_column[train_set_samples:valid_ind]
    X_valid = X_valid.reshape(other_set_samples, -1).T
    Y_valid = Y_valid.reshape(-1, other_set_samples)
    X_test, Y_test = samples_matrix[valid_ind: valid_ind + other_set_samples, :], labels_column[valid_ind: valid_ind + other_set_samples]
    X_test = X_test.reshape(other_set_samples, -1).T
    Y_test = Y_test.reshape(-1, other_set_samples)

    return X_train, Y_train, X_valid, Y_valid, X_test, Y_test


# Balances the dataset in order to have the same number of positive and negative instances
def balance_dataset(dataset: pd.DataFrame, label_name: str, percent):
    dataset_0 = dataset[dataset[label_name] == 0]
    dataset_1 = dataset[dataset[label_name] == 1]

    dataset_0 = dataset_0.sample(frac=percent, random_state=42)
    dataset_1 = dataset_1.sample(frac=percent, random_state=42)

    balanced_dataset = pd.concat([dataset_0, dataset_1])

    balanced_dataset = balanced_dataset.sample(frac=1, random_state=42).reset_index(drop=True)

    return balanced_dataset
