import numpy as np
import pandas as pd
import h5py

from CrossValidation import *
from DatasetPreprocessing import *


def main():
    np.random.seed(123456)
    np.set_printoptions(suppress=True)

    model_type = menu(
        "Vuoi fare una classificazione o una regressione?",
        ["classificazione", "regressione"]
    )

    reg_type = menu(
        "Vuoi usare la regolarizzazione L1, L2 o nessuna?",
        ["l1", "l2", "nessuna"]
    )

    activation_function = menu(
        "Che funzione di attivazione vuoi usare?",
        ["relu", "sigmoid", "tanh"]
    )

    if(model_type == "classificazione"):
        dataset = pd.read_csv("./datasets/alzheimers_disease_data.csv")
        label_name = "Diagnosis"
    else:
        dataset = pd.read_csv("./datasets/alzheimers_disease_data.csv")
        label_name = "Diagnosis"


    # dataset preprocessing
    X_train, Y_train, X_valid, Y_valid, X_test, Y_test = preprocess(dataset, label_name)

    # set up layers dimensions
    first = X_train.shape[0]
    layers_dims_list = [[first, 64, 128, 1], [first, 128, 256, 1], [first, 256, 256, 1]]
    lambda_list = [0.01, 0.5, 5]

    parameters = cross_validation(X_train, Y_train, X_valid, Y_valid, layers_dims_list, lambda_list, hidden_layers_activation_fn=activation_function, with_momentum=True, num_epochs=6000, reg_type=reg_type)

    # print the test accuracy
    print("The test accuracy rate: ", accuracy(X_test, parameters, Y_test))



def menu(messaggio, opzioni):
    while True:
        print(messaggio)
        scelta = input(f"Seleziona un'opzione tra {opzioni}: ").strip().lower()
        if scelta in opzioni:
            return scelta
        else:
            print(f"Scelta non valida. Scegli una delle seguenti opzioni: {opzioni}\n")



if __name__ == "__main__":
    main()