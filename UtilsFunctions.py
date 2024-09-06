import numpy as np

# Define activation functions that will be used in forward propagation
def sigmoid(Z):

    A = 1 / (1 + np.exp(-Z))

    return A, Z


def tanh(Z):

    A = np.tanh(Z)

    return A, Z


def relu(Z):

    A = np.maximum(0, Z)

    return A, Z


def leaky_relu(Z):

    A = np.maximum(0.1 * Z, Z)

    return A, Z

# Compute cross-entropy cost
def compute_cost(AL, y):

    m = y.shape[1]
    cost = - (1 / m) * np.sum(
        np.multiply(y, np.log(AL + 1e-16)) + np.multiply(1 - y, np.log(1 - AL + 1e-16)))

    return cost

# Define derivative of activation functions w.r.t z that will be used in back-propagation
def sigmoid_gradient(dA, Z):

    A, Z = sigmoid(Z)
    dZ = dA * A * (1 - A)

    return dZ


def tanh_gradient(dA, Z):

    A, Z = tanh(Z)
    dZ = dA * (1 - np.square(A))

    return dZ


def relu_gradient(dA, Z):

    A, Z = relu(Z)
    dZ = np.multiply(dA, np.int64(A > 0))

    return dZ


# define helper functions that will be used in L-model back-prop
def linear_backward(dZ, cache):

    A_prev, W, b = cache
    m = A_prev.shape[1]

    dW = (1 / m) * np.dot(dZ, A_prev.T)
    db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)
    dA_prev = np.dot(W.T, dZ)

    assert dA_prev.shape == A_prev.shape
    assert dW.shape == W.shape
    assert db.shape == b.shape

    return dA_prev, dW, db

# Roll a dictionary into a single vector.
def dictionary_to_vector(params_dict):

    count = 0
    for key in params_dict.keys():
        new_vector = np.reshape(params_dict[key], (-1, 1))
        if count == 0:
            theta_vector = new_vector
        else:
            theta_vector = np.concatenate((theta_vector, new_vector))
        count += 1

    return theta_vector

