import numpy as np
import pandas as pd
import h5py
import time

from CrossValidation import *
from DatasetPreprocessing import *
from datetime import datetime


def main():
    np.random.seed(123456)
    np.set_printoptions(suppress=True)

    clear_folder("./plots/accuracy")
    clear_folder("./plots/error")
    clear_folder("./csv_files")

    dataset = menu(
        "Che dataset vuoi usare?\n[1] mushroom classification (10.000 instances)\n[2] bank churn (17.000)\n[3] alzheimers disease (2000 instances)",
        ["1", "2", "3"]
    )

    activation_function = menu(
        "Che funzione di attivazione vuoi usare?\n[1] relu\n[2] sigmoid\n[3] tanh",
        ["1", "2", "3"]
    )
    if activation_function == "1":
        activation_function = "relu"
    elif activation_function == "2":
        activation_function = "sigmoid"
    elif activation_function == "3":
        activation_function = "tanh"

    if(dataset == "1"):
        dataset = pd.read_csv("./datasets/mushroom.csv")
        label_name = "class"
        dataset = balance_dataset(dataset, label_name, 0.2)
    elif dataset == "2":
        dataset = pd.read_csv("./datasets/bank_churn.csv")
        label_name = "Exited"
        dataset = balance_dataset(dataset, label_name, 0.2)
    else:
        dataset = pd.read_csv("./datasets/alzheimers_disease_data.csv")
        label_name = "Diagnosis"


    # dataset preprocessing
    X_train, Y_train, X_valid, Y_valid, X_test, Y_test = preprocess(dataset, label_name)

    # set up layers dimensions
    first = X_train.shape[0]

    if X_train.shape[1] < 3000:
        num_epoch = 50
    else:
        num_epoch = 70

    if first < 10:
        lambda_l1_list = [1e-4, 1e-3]
        lambda_l2_list = [0.01, 0.1, 0.5]
    else:
        lambda_l1_list = [0.5, 1, 1.2]
        lambda_l2_list = [0.1, 0.5, 4.5]
    layers_dims_list = [[first, 16, 8, 1], [first, 16, 16, 1], [first, 32, 16, 1], [first, 32, 32, 1], [first, 64, 32, 1], [first, 64, 64, 1], [first, 128, 64, 1], [first, 128, 128, 1]]

    print("Starting cross validation at ", datetime.now().time())
    parameters = cross_validation(X_train, Y_train, X_valid, Y_valid, layers_dims_list, lambda_l1_list, lambda_l2_list, hidden_layers_activation_fn=activation_function, with_momentum=True, num_epochs=80)

    # print the test accuracy
    print("The test accuracy rate: ", accuracy(X_test, parameters, Y_test, activation_function), "%")



def menu(messaggio, opzioni):
    while True:
        print(messaggio)
        scelta = input(f"Seleziona un numero tra {opzioni}: ").strip().lower()
        if scelta in opzioni:
            return scelta
        else:
            print(f"Scelta non valida.\n")



if __name__ == "__main__":
    main()
