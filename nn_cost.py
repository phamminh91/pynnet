import numpy as np
from nn_helpers import sigmoid, sigmoidGradient

def nnGradient(nn_params, *args):
    input_size, hidden_size, n_labels, X, y, lmbda = args

    W1_size = hidden_size * (input_size + 1)
    W1 = np.reshape(nn_params[0 : W1_size], (hidden_size, input_size + 1), order = 'F')
    W2 = np.reshape(nn_params[W1_size : ], (n_labels, hidden_size + 1), order = 'F')

    ###############################################################################
    # Part 1: Feed forward the neural network and return the cost in the
    #         variable J.
    ###############################################################################
    n = X.shape[0]
    a1 = np.hstack((np.ones((n, 1)), X))
    z2 = a1.dot(np.transpose(W1))
    a2 = sigmoid(z2)
    a2 = np.hstack((np.ones((a2.shape[0], 1)), a2))
    z3 = a2.dot(np.transpose(W2))
    a3 = sigmoid(z3)

    Y = np.eye(n_labels)[y, :]

    W1_nobias = W1[:, 1:]
    W2_nobias = W2[:, 1:]

    # calculate J
    delta_3 = a3 - Y
    delta_2 = np.dot(delta_3, W2) * np.hstack((np.ones((z2.shape[0], 1)), sigmoidGradient(z2)))
    delta_2 = delta_2[:, 1:]

    # accumulate gradients
    D2 = np.dot(np.transpose(delta_3), a2)
    D1 = np.dot(np.transpose(delta_2), a1)

    # calculate regularized gradient
    g2 = lmbda * 1.0 / n * np.hstack((np.zeros((W2.shape[0], 1)), W2_nobias))
    g1 = lmbda * 1.0 / n * np.hstack((np.zeros((W1.shape[0], 1)), W1_nobias))
    W1_grad = D1 / n + g1
    W2_grad = D2 / n + g2
    grad = np.hstack((W1_grad.T.ravel(), W2_grad.T.ravel()))

    return grad

def nnCost(nn_params, *args):
    input_size, hidden_size, n_labels, X, y, lmbda = args

    W1_size = hidden_size * (input_size + 1)
    W1 = np.reshape(nn_params[0 : W1_size], (hidden_size, input_size + 1), order = 'F')
    W2 = np.reshape(nn_params[W1_size : ], (n_labels, hidden_size + 1), order = 'F')

    ###############################################################################
    # Part 1: Feed forward the neural network and return the cost in the
    #         variable J.
    ###############################################################################
    n = X.shape[0]
    a1 = np.hstack((np.ones((n, 1)), X))
    z2 = a1.dot(np.transpose(W1))
    a2 = sigmoid(z2)
    a2 = np.hstack((np.ones((a2.shape[0], 1)), a2))
    z3 = a2.dot(np.transpose(W2))
    a3 = sigmoid(z3)

    Y = np.eye(n_labels)[y, :]

    W1_nobias = W1[:, 1:]
    W2_nobias = W2[:, 1:]

    J = -1.0 / n * np.sum((Y * 1.0 * np.log(a3) + (1.0 - Y) * np.log(1.0 - a3)).T.ravel())
    reg = lmbda * 1.0 / (2 * n) * (np.linalg.norm(W1_nobias.T.ravel())**2
                                 + np.linalg.norm(W2_nobias.T.ravel())**2)
    J = J + reg
    return J

def nnCostFunction(nn_params, input_size, hidden_size, n_labels, X, y, lmbda):
    """
    Implements the neural network cost function for a two layer neural network
    which performs classification
    Computes the cost and gradient of the neural network. The parameters for
    the neural network are "unrolled" into the vector nn_params and need to be
    converted back into the weight matrices.

    The returned parameter grad should be an "unrolled" vector of the partial
    derivatives of the neural network.

    @param
    @return J cost
    @return grad gradient
    """

    # X: n x m
    # w1: h x (m+1)
    # w2: o x (h+1)

    # Reshaping the input into W1 and W2
    print('\tReshaping ...')
    W1_size = hidden_size * (input_size + 1)
    W1 = np.reshape(nn_params[0 : W1_size], (hidden_size, input_size + 1), order = 'F')

    """
    print '\nW1 =================================='
    print 'w1.shape: ', W1.shape
    print 'w1.size:  ', W1_size
    print 'w1        ', W1[0, 0:5]
    print '\n'
    """

    W2 = np.reshape(nn_params[W1_size : ], (n_labels, hidden_size + 1), order = 'F')

    ###############################################################################
    # Part 1: Feed forward the neural network and return the cost in the
    #         variable J.
    ###############################################################################
    n = X.shape[0]

    print('\tComputing hidden layer input ...')
    a1 = np.hstack((np.ones((n, 1)), X))
    z2 = a1.dot(np.transpose(W1))
    print('\tComputing output layer ...')
    a2 = sigmoid(z2)
    a2 = np.hstack((np.ones((a2.shape[0], 1)), a2))
    z3 = a2.dot(np.transpose(W2))
    a3 = sigmoid(z3)

    Y = np.eye(n_labels)[y, :]

    W1_nobias = W1[:, 1:]
    W2_nobias = W2[:, 1:]

    # calculate J
    print('\tCalculating cost ...')
    J = -1.0 / n * np.sum((Y * 1.0 * np.log(a3) + (1.0 - Y) * np.log(1.0 - a3)).T.ravel())
    reg = lmbda * 1.0 / (2 * n) * (np.linalg.norm(W1_nobias.T.ravel())**2
                                 + np.linalg.norm(W2_nobias.T.ravel())**2)
    J = J + reg

    ###############################################################################
    # Part 2: Implement the backpropagation algorithm to compute the gradients
    #         W1_grad and W2_grad. You should return the partial derivative of
    #         the cost function with respect to W1 and W2 in W1_grad and W2_grad,
    #         respectively.
    #
    #         Note: The vector y passed into the function is a vector of labels
    #               containing values from 1..K. You need to map this vector into a
    #               binary vector of 1's and 0's to be used with the neural
    #               network cost function.

    ###############################################################################

    # error at output layer

    # delta_3   n_input x n_labels
    # W2        n_hidden x n_labels
    # delta_2   n_input x n_hidden
    # W1        n_input x n_hidden

    delta_3 = a3 - Y
    delta_2 = np.dot(delta_3, W2) * np.hstack((np.ones((z2.shape[0], 1)), sigmoidGradient(z2)))
    delta_2 = delta_2[:, 1:]

    # accumulate gradients
    # a2        n_input x n_hidden
    # delta_3   n_input x n_labels
    D2 = np.dot(np.transpose(delta_3), a2)
    D1 = np.dot(np.transpose(delta_2), a1)

    # calculate regularized gradient
    g2 = lmbda * 1.0 / n * np.hstack((np.zeros((W2.shape[0], 1)), W2_nobias))
    g1 = lmbda * 1.0 / n * np.hstack((np.zeros((W1.shape[0], 1)), W1_nobias))
    W1_grad = D1 / n + g1
    W2_grad = D2 / n + g2
    grad = np.hstack((W1_grad.T.ravel(), W2_grad.T.ravel()))

    return (J, grad)