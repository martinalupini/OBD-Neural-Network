import numpy as np
import time

from NeuralNetwork import *
from PlotUtils import *
from FileUtils import *
from datetime import datetime


def cross_validation(X_train, Y_train, X_valid, Y_valid, layers_dims_list, lambda_l1_list, lambda_l2_list, dir, hidden_layers_activation_fn="relu", with_momentum=True, learning_rate=0.01,  num_epochs=50, print_debug=True, mini_batch_size=64):

    best_parameters = None
    best_accuracy: float = 0.0
    best_dim = None
    best_lambda = None
    error_list_final_model = None
    accuracy_list_final_model = None

    start = time.time()

    for layers_dims in layers_dims_list:

        if print_debug:
            print("***********************************************************************************************")
            print("                         MODEL: ", layers_dims, " NO REGULARIZATION                            ")
            print("***********************************************************************************************")

        parameters, error, error_list, accuracy_list = model_with_regularization(X_train, Y_train, layers_dims, dir,
                                                                    learning_rate, num_epochs, hidden_layers_activation_fn,
                                                                  0, with_momentum, mini_batch_size=mini_batch_size)

        validation_accuracy = accuracy(X_valid, parameters, Y_valid, hidden_layers_activation_fn)
        training_accuracy = accuracy(X_train, parameters, Y_train, hidden_layers_activation_fn)

        if print_debug:
            print("The training accuracy rate with no regularization: ", training_accuracy)
            print("The validation accuracy rate with no regularization: ", validation_accuracy)
        add_csv_line(layers_dims, "none", "none", training_accuracy, validation_accuracy, dir, hidden_layers_activation_fn,)

        # Different scenarios: accuracy and cost are both better, accuracy or error is significantly better
        if validation_accuracy > best_accuracy :
            best_accuracy = validation_accuracy
            best_parameters = parameters
            best_dim = layers_dims
            best_lambda = 0
            error_list_final_model = error_list
            accuracy_list_final_model = accuracy_list
            reg_type = "no regularization"
        model_name = str(layers_dims)+"_noREG.png"
        #plotError(error_list, len(error_list), dir, model_name, hidden_layers_activation_fn)
        #plotAccuracy(accuracy_list, len(accuracy_list), dir, model_name, hidden_layers_activation_fn)

        for lambd in lambda_l1_list:

            if print_debug:
                print("***********************************************************************************************")
                print("                 MODEL: ", layers_dims, " L1 regularization lambda: ", lambd, "                ")
                print("***********************************************************************************************")

            parameters, error, error_list, accuracy_list = model_with_regularization(X_train, Y_train, layers_dims, dir,
                                                                          learning_rate, num_epochs, hidden_layers_activation_fn,
                                                                          lambd, with_momentum, False, mini_batch_size=mini_batch_size)

            validation_accuracy = accuracy(X_valid, parameters, Y_valid, hidden_layers_activation_fn)
            training_accuracy = accuracy(X_train, parameters, Y_train, hidden_layers_activation_fn)

            if print_debug:
                print("The training accuracy rate with L1 regularization: ", training_accuracy)
                print("The validation accuracy rate with L1 regularization: ", validation_accuracy)
            add_csv_line(layers_dims, "L1", lambd, training_accuracy, validation_accuracy, dir, hidden_layers_activation_fn)

            # Different scenarios: accuracy and cost are both better, accuracy or error is significantly better
            if validation_accuracy > best_accuracy :
                best_accuracy = validation_accuracy
                best_parameters = parameters
                best_dim = layers_dims
                best_lambda = lambd
                error_list_final_model = error_list
                accuracy_list_final_model = accuracy_list
                reg_type = "L1 regularization"

            model_name = str(layers_dims)+"_L1_" + str(lambd)+ ".png"
            #plotError(error_list, len(error_list), dir, model_name, hidden_layers_activation_fn)
            #plotAccuracy(accuracy_list, len(accuracy_list), dir, model_name, hidden_layers_activation_fn)

        for lambd in lambda_l2_list:

            if print_debug:
                print("***********************************************************************************************")
                print("                 MODEL: ", layers_dims, " L2 regularization lambda: ", lambd, "                ")
                print("***********************************************************************************************")

            parameters, error, error_list, accuracy_list = model_with_regularization(X_train, Y_train, layers_dims, dir,
                                                                          learning_rate, num_epochs, hidden_layers_activation_fn,
                                                                          lambd, with_momentum, True, mini_batch_size=mini_batch_size)

            validation_accuracy = accuracy(X_valid, parameters, Y_valid, hidden_layers_activation_fn)
            training_accuracy = accuracy(X_train, parameters, Y_train, hidden_layers_activation_fn)

            if print_debug:
                print("The training accuracy rate with L2 regularization: ", training_accuracy)
                print("The validation accuracy rate with L2 regularization: ", validation_accuracy)
            add_csv_line(layers_dims, "L2", lambd, training_accuracy, validation_accuracy, dir, hidden_layers_activation_fn)

            # Different scenarios: accuracy and cost are both better, accuracy or error is significantly better
            if validation_accuracy > best_accuracy :
                best_accuracy = validation_accuracy
                best_parameters = parameters
                best_dim = layers_dims
                best_lambda = lambd
                error_list_final_model = error_list
                accuracy_list_final_model = accuracy_list
                reg_type = "L2 regularization"

            model_name = str(layers_dims)+"_L2_" + str(lambd)+ ".png"
            #plotError(error_list, len(error_list), dir, model_name, hidden_layers_activation_fn)
            #plotAccuracy(accuracy_list, len(accuracy_list), dir, model_name, hidden_layers_activation_fn)

    end = time.time()
    min, sec = divmod(end - start, 60)

    print(f"Time spent for cross validation is {int(min)}:{sec:.2f} min")

    if reg_type == "no regularization":
        text = "Best configuration is " + str(best_dim) + " using " + reg_type
    else:
        text = "Best configuration is " + str(best_dim) + " using " + reg_type + " with lambda " + str(best_lambda)

    print(text)
    with open('plots/' + dir + '/' + hidden_layers_activation_fn + '/final_result', "w") as file:
        file.write("Time spent for cross validation is " + str(int(min)) + ":" + str(sec) + " min\n\n")
        file.write(text + "\n\n")

    plotError(error_list_final_model, len(error_list_final_model), dir, activation_fn=hidden_layers_activation_fn)
    plotAccuracy(accuracy_list_final_model, len(accuracy_list_final_model), dir, activation_fn=hidden_layers_activation_fn)
    return best_parameters
