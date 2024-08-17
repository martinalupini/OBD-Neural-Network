import numpy as np
import pandas as pd
import h5py

from sklearn.preprocessing import StandardScaler
from CrossValidation import *


def main():
    np.random.seed(123456)
    np.set_printoptions(suppress=True)

    dataset = pd.read_csv("./datasets/alzheimers_disease_data.csv")

    X_train, Y_train, X_valid, Y_valid, X_test, Y_test = preprocess(dataset, "Diagnosis")

    # set up layers dimensions
    first = X_train.shape[0]
    #layers_dims_list = [[first, 10, 1], [first, 10, 10, 1], [first, 20, 1], [first, 20, 20, 1]]
    #lambda_list = [0, 0.1, 10, 50]
    layers_dims_list = [[first, 10, 1]]
    lambda_list = [10]

    parameters = cross_validation(X_train, Y_train, X_valid, Y_valid, layers_dims_list, lambda_list, hidden_layers_activation_fn="sigmoid", with_momentum=True)

    # print the test accuracy
    print("The test accuracy rate: ", accuracy(X_test, parameters, Y_test))



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
    train_set_samples = int(samples_number * 0.8)
    other_set_samples = int(samples_number * 0.1)
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


if __name__ == "__main__":
    main()