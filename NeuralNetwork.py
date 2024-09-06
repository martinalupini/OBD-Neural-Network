import sys
import matplotlib.pyplot as plt
import numpy as np

from UtilsFunctions import *

def compute_cost_reg(AL, y, parameters, lambd=0, is_L2=True):
    """
    Computes the Cross-Entropy cost function with L2 regularization.

    Arguments
    ---------
    AL : 2d-array
        probability vector of shape 1 x training_examples.
    y : 2d-array
        true "label" vector.
    parameters : dict
        contains all the weight matrices and bias vectors for all layers.
    lambd : float
        regularization hyperparameter.

    Returns
    -------
    cost : float
        binary cross-entropy cost.
    """
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
    """
    Computes the gradient of the output w.r.t weight, bias, & post-activation
    output of (l - 1) layers at layer l.

    Arguments
    ---------
    dZ : 2d-array
        gradient of the cost w.r.t. the linear output (of current layer l).
    cache : tuple
        values of (A_prev, W, b) coming from the forward propagation in the
        current layer.
    lambd : float
        regularization hyperparameter.

    Returns
    -------
    dA_prev : 2d-array
        gradient of the cost w.r.t. the activation (of the previous layer l-1).
    dW : 2d-array
        gradient of the cost w.r.t. W (current layer l).
    db : 2d-array
        gradient of the cost w.r.t. b (current layer l).
    """
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
    """
    Arguments
    ---------
    dA : 2d-array
        post-activation gradient for current layer l.
    cache : tuple
        values of (linear_cache, activation_cache).
    activation : str
        activation used in this layer: "sigmoid", "tanh", or "relu".
    lambd : float
        regularization hyperparameter.

    Returns
    -------
    dA_prev : 2d-array
        gradient of the cost w.r.t. the activation (of previous layer l-1),
        same shape as A_prev.
    dW : 2d-array
        gradient of the cost w.r.t. W (current layer l), same shape as W.
    db : 2d-array
        gradient of the cost w.r.t. b (current layer l), same shape as b.
    """
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
    """
    Computes the gradient of output layer w.r.t weights, biases, etc. starting
    on the output layer in reverse topological order.

    Arguments
    ---------
    AL : 2d-array
        probability vector, output of the forward propagation
        (L_model_forward()).
    y : 2d-array
        true "label" vector (containing 0 if non-cat, 1 if cat).
    caches : list
        list of caches for all layers.
    hidden_layers_activation_fn :
        activation function used on hidden layers: "tanh", "relu".
    lambd : float
        regularization hyperparameter.

    Returns
    -------
    grads : dict
        gradients.
    """
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
        X, y, layers_dims, learning_rate=0.01,  num_epochs=3000,
        print_cost=False, hidden_layers_activation_fn="relu", lambd=0, with_momentum=True, is_L2=True):
    """
    Implements L-Layer neural network.

    Arguments
    ---------
    X : 2d-array
        cat, shape: number of examples x num_px * num_px * 3.
    y : 2d-array
        true "label" vector, shape: 1 x number of examples.
    layers_dims : list
        input size and size of each layer, length: number of layers + 1.
    learning_rate : float
        learning rate of the gradient descent update rule.
     num_epochs : int
        number of times to over the training cat.
    print_cost : bool
        if True, it prints the cost every 100 steps.
    hidden_layers_activation_fn : str
        activation function to be used on hidden layers: "tanh", "relu".
    lambd : float
        regularization hyperparameter.

    Returns
    -------
    parameters : dict
        parameters learnt by the model. They can then be used to predict test
        examples.
    """
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

# Initialize parameters
def initialize_parameters(layers_dims):
    """
    Initialize parameters dictionary.

    Weight matrices will be initialized to random values from uniform normal
    distribution.
    bias vectors will be initialized to zeros.

    Arguments
    ---------
    layers_dims : list or array-like
        dimensions of each layer in the network.

    Returns
    -------
    parameters : dict
        weight matrix and the bias vector for each layer.
    """
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
    """
    Computes affine transformation of the input.

    Arguments
    ---------
    A_prev : 2d-array
        activations output from previous layer.
    W : 2d-array
        weight matrix, shape: size of current layer x size of previuos layer.
    b : 2d-array
        bias vector, shape: size of current layer x 1.

    Returns
    -------
    Z : 2d-array
        affine transformation output.
    cache : tuple
        stores A_prev, W, b to be used in backpropagation.
    """
    Z = np.dot(W, A_prev) + b
    cache = (A_prev, W, b)

    return Z, cache


def linear_activation_forward(A_prev, W, b, activation_fn):
    """
    Computes post-activation output using non-linear activation function.

    Arguments
    ---------
    A_prev : 2d-array
        activations output from previous layer.
    W : 2d-array
        weight matrix, shape: size of current layer x size of previuos layer.
    b : 2d-array
        bias vector, shape: size of current layer x 1.
    activation_fn : str
        non-linear activation function to be used: "sigmoid", "tanh", "relu".

    Returns
    -------
    A : 2d-array
        output of the activation function.
    cache : tuple
        stores linear_cache and activation_cache. ((A_prev, W, b), Z) to be used in backpropagation.
    """
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
    """
    Computes the output layer through looping over all units in topological
    order.

    Arguments
    ---------
    X : 2d-array
        input matrix of shape input_size x training_examples.
    parameters : dict
        contains all the weight matrices and bias vectors for all layers.
    hidden_layers_activation_fn : str
        activation function to be used on hidden layers: "tanh", "relu".

    Returns
    -------
    AL : 2d-array
        probability vector of shape 1 x training_examples.
    caches : list
        that contains L tuples where each layer has: A_prev, W, b, Z.
    """
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
    """
    Update the parameters' values using gradient descent rule.

    Arguments
    ---------
    parameters : dict
        contains all the weight matrices and bias vectors for all layers.
    grads : dict
        stores all gradients (output of L_model_backward).

    Returns
    -------
    parameters : dict
        updated parameters.
    """
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
    """
    Computes the average accuracy rate.

    Arguments
    ---------
    X : 2d-array
        cat, shape: number of examples x num_px * num_px * 3.
    parameters : dict
        learnt parameters.
    y : 2d-array
        true "label" vector, shape: 1 x number of examples.
    activation_fn : str
        activation function to be used on hidden layers: "tanh", "relu".

    Returns
    -------
    accuracy : float
        accuracy rate after applying parameters on the input cat
    """
    probs, caches = L_model_forward(X, parameters, activation_fn)
    labels = (probs >= 0.5) * 1
    accuracy = np.mean(labels == y) * 100

    return accuracy


def create_mini_batches(X, y, mini_batch_size):
    """
    Suddivide il set di dati in mini-batch.

    Arguments
    ---------
    X : 2d-array
        Input di dimensione (input_size, numero di esempi).
    y : 2d-array
        Etichette di dimensione (1, numero di esempi).
    mini_batch_size : int
        Dimensione di ciascun mini-batch.

    Returns
    -------
    mini_batches : list
        Una lista di tuple (mini_batch_X, mini_batch_y).
    """
    m = X.shape[1]  # numero di esempi
    mini_batches = []

    # Mischiare il set di dati
    permutation = np.random.permutation(m)
    shuffled_X = X[:, permutation]
    shuffled_y = y[:, permutation]

    # Creare i mini-batch
    num_complete_minibatches = m // mini_batch_size  # Numero di batch completi
    for k in range(num_complete_minibatches):
        mini_batch_X = shuffled_X[:, k * mini_batch_size:(k + 1) * mini_batch_size]
        mini_batch_y = shuffled_y[:, k * mini_batch_size:(k + 1) * mini_batch_size]
        mini_batches.append((mini_batch_X, mini_batch_y))

    # Gestire il caso in cui rimangano esempi non sufficienti per un batch completo
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[:, num_complete_minibatches * mini_batch_size:]
        mini_batch_y = shuffled_y[:, num_complete_minibatches * mini_batch_size:]
        mini_batches.append((mini_batch_X, mini_batch_y))

    return mini_batches
