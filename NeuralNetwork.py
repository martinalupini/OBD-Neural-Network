import sys
import matplotlib.pyplot as plt
import numpy as np

from UtilsFunctions import *

def compute_cost_reg(AL, y, parameters, lambd=0, is_L2=True):

    # number of examples
    m = y.shape[1]

    # compute traditional cross entropy cost
    cross_entropy_cost = compute_cost(AL, y)

    # convert parameters dictionary to vector
    parameters_vector = dictionary_to_vector(parameters)

    # compute the regularization penalty
    if is_L2:
        regularization_penalty = (lambd / (2 * m)) * np.sum(np.square(parameters_vector))
    elif not is_L2:
        regularization_penalty = (lambd / (2 * m)) * np.sum(np.abs(parameters_vector))
    else:
        regularization_penalty = 0

    # compute the total cost
    cost = cross_entropy_cost + regularization_penalty

    return cost


def linear_backward_reg(dZ, cache, lambd=0, is_L2=True):

    A_prev, W, b = cache
    m = A_prev.shape[1]

    if(is_L2):
        dW = (1 / m) * np.dot(dZ, A_prev.T) + (lambd / m) * W
    else:
        dW = (1 / m) * np.dot(dZ, A_prev.T) + (lambd / m) * np.sign(W)
    db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)
    dA_prev = np.dot(W.T, dZ)

    assert (dA_prev.shape == A_prev.shape)
    assert (dW.shape == W.shape)
    assert (db.shape == b.shape)

    return dA_prev, dW, db


def linear_activation_backward_reg(dA, cache, activation_fn="relu", lambd=0, is_L2=True):

    linear_cache, activation_cache = cache

    if activation_fn == "sigmoid":
        dZ = sigmoid_gradient(dA, activation_cache)
        dA_prev, dW, db = linear_backward_reg(dZ, linear_cache, lambd, is_L2)

    elif activation_fn == "tanh":
        dZ = tanh_gradient(dA, activation_cache)
        dA_prev, dW, db = linear_backward_reg(dZ, linear_cache, lambd, is_L2)

    elif activation_fn == "relu":
        dZ = relu_gradient(dA, activation_cache)
        dA_prev, dW, db = linear_backward_reg(dZ, linear_cache, lambd, is_L2)

    return dA_prev, dW, db


def L_model_backward_reg(AL, y, caches, hidden_layers_activation_fn="relu",
                         lambd=0, is_L2=True):

    y = y.reshape(AL.shape)
    L = len(caches)
    grads = {}

    epsilon = 1e-8
    dAL = np.divide(AL - y, np.multiply(AL, 1 - AL) + epsilon)

    grads["dA" + str(L - 1)], grads["dW" + str(L)], grads["db" + str(L)] = \
        linear_activation_backward_reg(dAL, caches[L - 1], "sigmoid", lambd, is_L2)

    for l in range(L - 1, 0, -1):
        current_cache = caches[l - 1]
        grads["dA" + str(l - 1)], grads["dW" + str(l)], grads["db" + str(l)] = \
            linear_activation_backward_reg(
                grads["dA" + str(l)], current_cache,
                hidden_layers_activation_fn, lambd, is_L2)

    return grads


def model_with_regularization(
        X, y, layers_dims, dir, learning_rate=0.01, num_epochs=50, hidden_layers_activation_fn="relu", lambd=0, with_momentum=True, is_L2=True):

    # get number of examples
    m = X.shape[1]

    # to get consistents output
    np.random.seed(1)

    # initialize parameters
    parameters, previous_parameters = initialize_parameters(layers_dims)

    # intialize cost list
    cost_list = []
    grad_list = []
    accuracy_list = []

    decay = 0

    for i in range(num_epochs):

        diminishing_stepsize = learning_rate / (1 + decay * i)

        # Creare mini-batch
        mini_batches = create_mini_batches(X, y, 64)

        for mini_batch in mini_batches:
            (mini_batch_X, mini_batch_y) = mini_batch

            # Forward propagation
            AL, caches = L_model_forward(mini_batch_X, parameters, hidden_layers_activation_fn)

            # Calcolare il costo regolarizzato
            reg_cost = compute_cost_reg(AL, mini_batch_y, parameters, lambd, is_L2)

            # Backward propagation
            grads = L_model_backward_reg(AL, mini_batch_y, caches, hidden_layers_activation_fn, lambd, is_L2)

            # Aggiornare i parametri con o senza momentum
            parameters, previous_parameters = update_parameters(parameters, grads, diminishing_stepsize, previous_parameters, with_momentum)


        AL, caches = L_model_forward(X, parameters, hidden_layers_activation_fn)
        reg_cost = compute_cost_reg(AL, y, parameters, lambd, is_L2)
        accuracy_epoch = accuracy(X, parameters, y, hidden_layers_activation_fn)
        # append cost
        cost_list.append(reg_cost)
        accuracy_list.append(accuracy_epoch)

    return parameters, reg_cost, cost_list, accuracy_list

# Initialize parameters with he inizialization
def initialize_parameters(layers_dims):

    np.random.seed(1)
    parameters = {}
    previous_parameters = {}
    L = len(layers_dims)

    for l in range(1, L):
        parameters["W" + str(l)] = np.random.randn(
            layers_dims[l],
            layers_dims[l - 1]) * np.sqrt(2 / layers_dims[l - 1])
        parameters["b" + str(l)] = np.zeros((layers_dims[l], 1))

        previous_parameters["W" + str(l)] = np.zeros((layers_dims[l], layers_dims[l - 1]))
        previous_parameters["b" + str(l)] = np.zeros((layers_dims[l], 1))

        assert parameters["W" + str(l)].shape == (
            layers_dims[l], layers_dims[l - 1])
        assert parameters["b" + str(l)].shape == (layers_dims[l], 1)

    return parameters, previous_parameters

# Define helper functions that will be used in L-model forward prop
def linear_forward(A_prev, W, b):

    Z = np.dot(W, A_prev) + b
    cache = (A_prev, W, b)

    return Z, cache


def linear_activation_forward(A_prev, W, b, activation_fn):

    assert activation_fn == "sigmoid" or activation_fn == "tanh" or \
           activation_fn == "relu"

    if activation_fn == "sigmoid":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = sigmoid(Z)

    elif activation_fn == "tanh":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = tanh(Z)

    elif activation_fn == "relu":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = relu(Z)

    assert A.shape == (W.shape[0], A_prev.shape[1])

    cache = (linear_cache, activation_cache)

    return A, cache


def L_model_forward(X, parameters, hidden_layers_activation_fn="relu"):

    A = X
    caches = []
    L = len(parameters) // 2

    for l in range(1, L):
        A_prev = A
        A, cache = linear_activation_forward(
            A_prev, parameters["W" + str(l)], parameters["b" + str(l)],
            activation_fn=hidden_layers_activation_fn)
        caches.append(cache)

    AL, cache = linear_activation_forward(
        A, parameters["W" + str(L)], parameters["b" + str(L)],
        activation_fn="sigmoid")
    caches.append(cache)

    assert AL.shape == (1, X.shape[1])

    return AL, caches


# define the function to update both weight matrices and bias vectors
def update_parameters(parameters, grads, learning_rate, previous_parameters, with_momentum=True, momentum=0.9):

    L = len(parameters) // 2
    prev_parameters = parameters

    for l in range(1, L + 1):
        if with_momentum:
            parameters["W" + str(l)] = parameters[
                                       "W" + str(l)] - learning_rate * grads["dW" + str(l)] + momentum * (parameters["W" + str(l)] - previous_parameters["W" + str(l)])
            parameters["b" + str(l)] = parameters[
                                       "b" + str(l)] - learning_rate * grads["db" + str(l)] + momentum * (parameters["b" + str(l)] - previous_parameters["b" + str(l)])
        else:
            parameters["W" + str(l)] = parameters[
                                           "W" + str(l)] - learning_rate * grads["dW" + str(l)]
            parameters["b" + str(l)] = parameters[
                                           "b" + str(l)] - learning_rate * grads["db" + str(l)]


    return parameters, prev_parameters


def accuracy(X, parameters, y, activation_fn):

    probs, caches = L_model_forward(X, parameters, activation_fn)
    labels = (probs >= 0.5) * 1
    accuracy = np.mean(labels == y) * 100

    return accuracy


def create_mini_batches(X, y, mini_batch_size):

    m = X.shape[1]
    mini_batches = []

    # shuffling the dataset
    permutation = np.random.permutation(m)
    shuffled_X = X[:, permutation]
    shuffled_y = y[:, permutation]

    # creation of mini batches
    num_complete_minibatches = m // mini_batch_size
    for k in range(num_complete_minibatches):
        mini_batch_X = shuffled_X[:, k * mini_batch_size:(k + 1) * mini_batch_size]
        mini_batch_y = shuffled_y[:, k * mini_batch_size:(k + 1) * mini_batch_size]
        mini_batches.append((mini_batch_X, mini_batch_y))

    # handle the case where there are not enough examples remaining for a full batch.
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[:, num_complete_minibatches * mini_batch_size:]
        mini_batch_y = shuffled_y[:, num_complete_minibatches * mini_batch_size:]
        mini_batches.append((mini_batch_X, mini_batch_y))

    return mini_batches
