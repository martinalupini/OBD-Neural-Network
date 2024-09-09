import numpy as np
import time

from NeuralNetwork import *
from PlotUtils import *
from FileUtils import *
import concurrent.futures


def cross_validation(X_train, Y_train, X_valid, Y_valid, layers_dims_list, lambda_l1_list, lambda_l2_list, dir, hidden_layers_activation_fn="relu", with_momentum=True, learning_rate=0.01, num_epochs=50, print_debug=True, mini_batch_size=64):

    best_parameters = None
    best_accuracy: float = 0.0
    best_dim = None
    best_lambda = None
    error_list_final_model = None
    accuracy_list_final_model = None
    reg_type = None

    start = time.time()
    print("Starting cross validation. This might take time...")

    def evaluate_model_CV(layers_dims, lambd, reg_type):
        if reg_type == "none":
            parameters, error, error_list, accuracy_list = model_with_regularization(
                X_train, Y_train, layers_dims, dir, learning_rate, num_epochs, hidden_layers_activation_fn,
                0, with_momentum, mini_batch_size=mini_batch_size)
        elif reg_type == "L1":
            parameters, error, error_list, accuracy_list = model_with_regularization(
                X_train, Y_train, layers_dims, dir, learning_rate, num_epochs, hidden_layers_activation_fn,
                lambd, with_momentum, False, mini_batch_size=mini_batch_size)
        else:  # L2
            parameters, error, error_list, accuracy_list = model_with_regularization(
                X_train, Y_train, layers_dims, dir, learning_rate, num_epochs, hidden_layers_activation_fn,
                lambd, with_momentum, True, mini_batch_size=mini_batch_size)

        validation_accuracy, precision, recall, f1 = evaluate_model(X_valid, parameters, Y_valid, hidden_layers_activation_fn)
        training_accuracy, precision, recall, f1 = evaluate_model(X_train, parameters, Y_train, hidden_layers_activation_fn)

        if print_debug:
            print(f"The training accuracy rate for model {layers_dims} with {reg_type} regularization and lambda {lambd}: ", training_accuracy)
            print(f"The validation accuracy rate for model {layers_dims} with {reg_type} regularization and lambda {lambd}: ", validation_accuracy)

        return {
            'parameters': parameters,
            'validation_accuracy': validation_accuracy,
            'training_accuracy': training_accuracy,
            'layers_dims': layers_dims,
            'lambda': lambd,
            'error_list': error_list,
            'accuracy_list': accuracy_list,
            'reg_type': reg_type
        }

    def update_best_model(result):
        nonlocal best_accuracy, best_parameters, best_dim, best_lambda, error_list_final_model, accuracy_list_final_model, reg_type
        if result['validation_accuracy'] > best_accuracy:
            best_accuracy = result['validation_accuracy']
            best_parameters = result['parameters']
            best_dim = result['layers_dims']
            best_lambda = result['lambda']
            error_list_final_model = result['error_list']
            accuracy_list_final_model = result['accuracy_list']
            reg_type = result['reg_type']

    results = []

    # Parallel processing using ThreadPoolExecutor
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        for layers_dims in layers_dims_list:
            futures.append(executor.submit(evaluate_model_CV, layers_dims, None, "none"))
            for lambd in lambda_l1_list:
                futures.append(executor.submit(evaluate_model_CV, layers_dims, lambd, "L1"))
            for lambd in lambda_l2_list:
                futures.append(executor.submit(evaluate_model_CV, layers_dims, lambd, "L2"))

        # Aspetta che tutti i thread siano completati
        concurrent.futures.wait(futures)

        # Raccolta dei risultati da tutti i thread completati
        for future in futures:
            result = future.result()  # Estrai il risultato da ciascun future
            results.append(result)

    for result in results:
        add_csv_line(result['layers_dims'], result['reg_type'], result['lambda'], result['training_accuracy'], result['validation_accuracy'], dir, hidden_layers_activation_fn)
        update_best_model(result)

    end = time.time()
    min, sec = divmod(end - start, 60)
    print(f"End cross validation. Time spent for cross validation is {int(min)}:{sec:.2f} min")

    if reg_type == "none":
        text = f"Best configuration is {best_dim} using no regularization"
    else:
        text = f"Best configuration is {best_dim} using {reg_type} with lambda {best_lambda}"

    print(text)
    with open(f'plots/{dir}/{hidden_layers_activation_fn}/final_result', "w") as file:
        file.write(f"Time spent for cross validation is {int(min)}:{sec:.2f} min\n\n")
        file.write(text + "\n\n")

    plotError(error_list_final_model, len(error_list_final_model), dir, activation_fn=hidden_layers_activation_fn)
    plotAccuracy(accuracy_list_final_model, len(accuracy_list_final_model), dir, activation_fn=hidden_layers_activation_fn)

    return best_parameters
