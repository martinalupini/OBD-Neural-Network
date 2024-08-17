import numpy as np
from NeuralNetwork import *


def cross_validation(X_train, Y_train, X_valid, Y_valid, layers_dims_list, lambda_list, hidden_layers_activation_fn="relu", with_momentum=True, learning_rate=0.01,  num_epochs=3000,
                     print_cost=False):

    best_parameters = None
    best_accuracy: float = 0.0
    min_error = float('inf')
    best_dim = None
    best_lambda = None
    error_list_final_model = None

    for lambd in lambda_list:
        for layers_dims in layers_dims_list:

            print("***********************************************************************************************")
            print("                         MODEL: ", layers_dims, " lambda: ", lambd, "                        ")
            print("***********************************************************************************************")

            if lambd == 0:
                parameters, error, error_list = model_with_regularization(X_train, Y_train, layers_dims,
                                                                          learning_rate, num_epochs, print_cost, hidden_layers_activation_fn,
                                                                          lambd, with_momentum)

                validation_accuracy = accuracy(X_valid, parameters, Y_valid)
                print("The training accuracy rate with no regularization: ", accuracy(X_train, parameters, Y_train))
                print("The training error with no regularization: ", error)
                print("The validation accuracy rate with no regularization: ", validation_accuracy)
                print("-----------------------------------------------------------------------------------------------")

                # Different scenarios: accuracy and cost are both better, accuracy or error is significantly better
                if (validation_accuracy > best_accuracy and error < min_error) or \
                        ((best_accuracy - validation_accuracy) < 5 and error < min_error) or \
                        (validation_accuracy - best_accuracy) > 10:
                    best_accuracy = validation_accuracy
                    min_error = error
                    best_parameters = parameters
                    best_dim = layers_dims
                    best_lambda = lambd
                    error_list_final_model = error_list

                continue

            for is_L2 in [True, False]:

                parameters, error, error_list = model_with_regularization(X_train, Y_train, layers_dims,
                                                                          learning_rate, num_epochs, print_cost, hidden_layers_activation_fn,
                                                                          lambd, with_momentum, is_L2)

                validation_accuracy = accuracy(X_valid, parameters, Y_valid)
                if is_L2: reg = "L2"
                else: reg = "L1"
                print("The training accuracy rate with ", reg, " regularization: ", accuracy(X_train, parameters, Y_train))
                print("The training error with ", reg, " regularization: ", error)
                print("The validation accuracy rate with ", reg, " regularization: ", validation_accuracy)
                print("-----------------------------------------------------------------------------------------------")

                # Different scenarios: accuracy and cost are both better, accuracy or error is significantly better
                if (validation_accuracy > best_accuracy and error < min_error) or \
                    ((best_accuracy - validation_accuracy) < 5 and error < min_error) or \
                    (validation_accuracy - best_accuracy) > 10:
                    best_accuracy = validation_accuracy
                    min_error = error
                    best_parameters = parameters
                    best_dim = layers_dims
                    best_lambda = lambd
                    error_list_final_model = error_list

    print("Best configuration is ", best_dim, " with lambda", best_lambda)
    plotError(error_list_final_model, num_epochs, hidden_layers_activation_fn, with_momentum)
    return best_parameters
