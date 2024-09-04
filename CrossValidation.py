import numpy as np
from NeuralNetwork import *


def cross_validation(X_train, Y_train, X_valid, Y_valid, layers_dims_list, lambda_list, hidden_layers_activation_fn="relu", with_momentum=True, learning_rate=0.01,  num_epochs=3000,
                     print_cost=False, reg_type="none"):

    best_parameters = None
    best_accuracy: float = 0.0
    min_error = float('inf')
    best_dim = None
    best_lambda = None
    error_list_final_model = None

    if reg_type == "nessuna":
        for layers_dims in layers_dims_list:

            print("***********************************************************************************************")
            print("                         MODEL: ", layers_dims, " NO REGULARIZATION                            ")
            print("***********************************************************************************************")

            parameters, error, error_list = model_with_regularization(X_train, Y_train, layers_dims,
                                                                          learning_rate, num_epochs, print_cost, hidden_layers_activation_fn,
                                                                          0, with_momentum)

            validation_accuracy = accuracy(X_valid, parameters, Y_valid, hidden_layers_activation_fn)
            print("The training accuracy rate with no regularization: ", accuracy(X_train, parameters, Y_train, hidden_layers_activation_fn))
            print("The validation accuracy rate with no regularization: ", validation_accuracy)
            print("-----------------------------------------------------------------------------------------------")

            # Different scenarios: accuracy and cost are both better, accuracy or error is significantly better
            if validation_accuracy > best_accuracy :
                best_accuracy = validation_accuracy
                best_parameters = parameters
                best_dim = layers_dims
                best_lambda = 0
                error_list_final_model = error_list

    elif reg_type == "l1":
        for lambd in lambda_list:
            for layers_dims in layers_dims_list:

                print("***********************************************************************************************")
                print("                 MODEL: ", layers_dims, " L1 regularization lambda: ", lambd, "                ")
                print("***********************************************************************************************")

                parameters, error, error_list = model_with_regularization(X_train, Y_train, layers_dims,
                                                                          learning_rate, num_epochs, print_cost, hidden_layers_activation_fn,
                                                                          lambd, with_momentum, False)

                validation_accuracy = accuracy(X_valid, parameters, Y_valid, hidden_layers_activation_fn)
                print("The training accuracy rate with L1 regularization: ", accuracy(X_train, parameters, Y_train, hidden_layers_activation_fn))
                print("The validation accuracy rate with L1 regularization: ", validation_accuracy)
                print("-----------------------------------------------------------------------------------------------")

                # Different scenarios: accuracy and cost are both better, accuracy or error is significantly better
                if validation_accuracy > best_accuracy :
                    best_accuracy = validation_accuracy
                    best_parameters = parameters
                    best_dim = layers_dims
                    best_lambda = lambd
                    error_list_final_model = error_list


    elif reg_type == "l2":
        for lambd in lambda_list:
            for layers_dims in layers_dims_list:

                print("***********************************************************************************************")
                print("                 MODEL: ", layers_dims, " L2 regularization lambda: ", lambd, "                ")
                print("***********************************************************************************************")

                parameters, error, error_list = model_with_regularization(X_train, Y_train, layers_dims,
                                                                          learning_rate, num_epochs, print_cost, hidden_layers_activation_fn,
                                                                          lambd, with_momentum, True)

                validation_accuracy = accuracy(X_valid, parameters, Y_valid, hidden_layers_activation_fn)
                print("The training accuracy rate with L2 regularization: ", accuracy(X_train, parameters, Y_train, hidden_layers_activation_fn))
                print("The validation accuracy rate with L2 regularization: ", validation_accuracy)
                print("-----------------------------------------------------------------------------------------------")

                # Different scenarios: accuracy and cost are both better, accuracy or error is significantly better
                if validation_accuracy > best_accuracy :
                    best_accuracy = validation_accuracy
                    best_parameters = parameters
                    best_dim = layers_dims
                    best_lambda = lambd
                    error_list_final_model = error_list


    print("Best configuration with " + reg_type + " is ", best_dim, " with lambda", best_lambda)
    plotError(error_list_final_model, num_epochs, hidden_layers_activation_fn, with_momentum)
    return best_parameters
