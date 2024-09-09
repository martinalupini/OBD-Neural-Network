import numpy as np
import pandas as pd

from CrossValidation import *
from DatasetPreprocessing import *


def main():
    np.random.seed(123456)
    np.set_printoptions(suppress=True)

    dataset = menu(
        "What dataset do you want to use?\n[1] mushroom classification (6000 instances)\n[2] credit card fraud (20000 instances)\n[3] alzheimers disease (2000 instances)",
        ["1", "2", "3"]
    )

    activation_function = menu(
        "What activation function do you want to use?\n[1] relu\n[2] tanh",
        ["1", "2"]
    )
    if activation_function == "1":
        activation_function = "relu"
    elif activation_function == "2":
        activation_function = "tanh"

    if dataset == "1":
        dataset = pd.read_csv("./datasets/mushroom.csv")
        label_name = "class"
        dir = "mushroom"
        dataset = resize_dataset(dataset, label_name, 0.2)
        num_epoch = 50
        mini_batch_size = 64
    elif dataset == "2":
        dataset = pd.read_csv("./datasets/fraud.csv")
        label_name = "Class"
        dir = "fraud"
        dataset = resize_dataset(dataset, label_name, 0.02)
        num_epoch = 50
        mini_batch_size = 64
    else:
        dataset = pd.read_csv("./datasets/alzheimers_disease_data.csv")
        label_name = "Diagnosis"
        dir = "alzheimer"
        num_epoch = 50
        mini_batch_size = 64

    # clearing previous plots and files
    clear_folder("./plots/" + dir + "/" + activation_function)
    if os.path.exists("./plots/" + dir + "/results.csv"):
        os.remove("./plots/" + dir + "/results.csv")

    # dataset preprocessing
    X_train, Y_train, X_valid, Y_valid, X_test, Y_test = preprocess(dataset, label_name, dir)

    # set up layers dimensions
    first = X_train.shape[0]

    if first < 30:
        lambda_l1_list = [1e-3, 5e-3]
        lambda_l2_list = [0.01, 0.1, 0.5]
    else:
        lambda_l1_list = [0.5, 1]
        lambda_l2_list = [0.1, 0.5, 4.5]
    layers_dims_list = [[first, 64, 64, 1], [first, 128, 64, 1], [first, 128, 128, 1], [first, 256, 128, 1], [first, 256, 256, 1]]

    parameters = cross_validation(X_train, Y_train, X_valid, Y_valid, layers_dims_list, lambda_l1_list, lambda_l2_list,
                                  dir, hidden_layers_activation_fn=activation_function, with_momentum=True,
                                  num_epochs=num_epoch, print_debug=False, mini_batch_size=mini_batch_size)

    # print the test accuracy
    accuracy, precision, recall, f1 = evaluate_model(X_test, parameters, Y_test, activation_function)
    text = "The test accuracy rate: " + str(accuracy) + "%\nThe precision is " + str(precision) + "%\nThe recall is " + str(recall) + "%\nThe F1 score is " + str(f1)
    print(text)
    with open('plots/' + dir + '/' + activation_function + '/final_result', "a") as file:
        file.write(text + "\n")



def menu(message, options):
    while True:
        print(message)
        choice = input(f"Select a number between {options}: ").strip().lower()
        if choice in options:
            return choice
        else:
            print(f"Non valid choice. Try again.\n")



if __name__ == "__main__":
    main()
