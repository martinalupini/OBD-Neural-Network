import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from PlotUtils import *


# Splits the dataset in training, validation and testing set and standardizes the data
def preprocess(dataset: pd.DataFrame, label_name: str, dataset_name):
    preprocessed_dataset = dataset
    # removing lines with null values
    preprocessed_dataset = preprocessed_dataset.dropna(axis=0)

    # shuffling the dataset
    #preprocessed_dataset = preprocessed_dataset.sample(frac=1).reset_index(drop=True)

    # removing the label column from dataset and saving it in another array
    labels_column = preprocessed_dataset[label_name].values
    preprocessed_dataset = preprocessed_dataset.drop(label_name, axis=1)

    # Standardizing the dataset
    scaler = StandardScaler()
    standardized_matrix = scaler.fit_transform(preprocessed_dataset)

    X_train, X_temp, Y_train, Y_temp = train_test_split(
        standardized_matrix, labels_column,
        test_size=0.4, stratify=labels_column, random_state=42
    )

    # Ora dividiamo il set rimanente in validazione e test
    X_valid, X_test, Y_valid, Y_test = train_test_split(
        X_temp, Y_temp,
        test_size=0.5, stratify=Y_temp, random_state=42
    )

    # Riorganizzazione delle matrici per la forma richiesta
    X_train = X_train.T
    Y_train = Y_train.reshape(1, -1)
    X_valid = X_valid.T
    Y_valid = Y_valid.reshape(1, -1)
    X_test = X_test.T
    Y_test = Y_test.reshape(1, -1)

    exploratory_data_analysis(Y_train, dataset_name + "/train.png")
    exploratory_data_analysis(Y_valid, dataset_name + "/valid.png")
    exploratory_data_analysis(Y_test, dataset_name + "/test.png")

    return X_train, Y_train, X_valid, Y_valid, X_test, Y_test


# Balances the dataset in order to have the same number of positive and negative instances
def resize_dataset(dataset: pd.DataFrame, label_name: str, percent):
    dataset_0 = dataset[dataset[label_name] == 0]
    dataset_1 = dataset[dataset[label_name] == 1]

    dataset_0 = dataset_0.sample(frac=percent, random_state=42)
    dataset_1 = dataset_1.sample(frac=percent, random_state=42)

    balanced_dataset = pd.concat([dataset_0, dataset_1])

    balanced_dataset = balanced_dataset.sample(frac=1, random_state=42).reset_index(drop=True)

    return balanced_dataset


def exploratory_data_analysis_entire_dataset():

    dataset = pd.read_csv("./datasets/mushroom.csv")
    label_name = "class"
    count = dataset[label_name].value_counts()
    plotClasses(count, "mushroom.png")

    dataset = pd.read_csv("./datasets/bank_churn.csv")
    label_name = "Exited"
    count = dataset[label_name].value_counts()
    plotClasses(count, "bank.png")

    dataset = pd.read_csv("./datasets/alzheimers_disease_data.csv")
    label_name = "Diagnosis"
    count = dataset[label_name].value_counts()
    plotClasses(count, "alzheimer.png")


#exploratory_data_analysis_entire_dataset()

