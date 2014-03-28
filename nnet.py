import numpy as np
from nn_helpers import *
from scipy.optimize import fmin_cg
from nnCostFunction import nnCost, nnGradient

class NeuralNet:
    """
    A Neural Network with
    1 input layer
    1 hidden layer
    1 output layer
    """

    def __init__(self, n_features, n_hiddens, n_labels, lmbda = 1.0, maxiter = 50):
        """
        Initialize the neural network with weights W1 and W2
        @param n_features dimensions of the input
        @param n_hiddens number of hidden units
        @param n_labels number of classes
        """
        self._n_features = n_features
        self._n_hiddens = n_hiddens
        self._n_labels = n_labels
        self._W1 = np.array([])
        self._W2 = np.array([])
        self._lmbda = float(lmbda)
        self._maxiter = int(maxiter)

    def fit(self, X, y):
        """
        Train the neural network on input X and corresponding classes y
        @param X training data
        @param y classes of training data
        """
        n_instances = X.shape[0]
        assert (y.shape == (n_instances, )), 'the number of instances in X and y do not match'

        initial_nn_params = self.init_weights()

        print("\nTraining the network ...")
        xopt, fopt, func_calls, grad_calls, warnflag = fmin_cg(
                                                        nnCost,
                                                        initial_nn_params,
                                                        fprime = nnGradient,
                                                        args = (self._n_features, self._n_hiddens, self._n_labels, X, y, self._lmbda),
                                                        maxiter = self._maxiter,
                                                        full_output = True)

        W1_size = self._n_hiddens * (self._n_features + 1)

        self._W1 = np.reshape(xopt[0 : W1_size],
                             (self._n_hiddens, self._n_features + 1),
                             order = 'F')

        self._W2 = np.reshape(xopt[W1_size : ],
                             (self._n_labels, self._n_hiddens + 1),
                             order = 'F')

    def predict(self, X):
        """
        Predict the label

        @param X input needs to be classified
        @return p Predicted label of X given the trained weights
        of a neural network (W1, W2)
        """
        # X: n x m
        # w1: h x (m+1)
        # w2: o x (h+1)

        n = X.shape[0]
        a1 = np.hstack((np.ones((n, 1)), X))
        z2 = a1.dot(np.transpose(self._W1))
        a2 = sigmoid(z2)
        a2 = np.hstack((np.ones((a2.shape[0], 1)), a2))
        z3 = a2.dot(np.transpose(self._W2))
        a3 = sigmoid(z3)
        p = a3.argmax(axis = 1)

        return p

    def init_weights(self):
        """
        Initialize the weights of the neural network
        """
        self._W1 = randInitWeights(self._n_features, self._n_hiddens)
        self._W2 = randInitWeights(self._n_hiddens, self._n_labels)
        initial_nn_params = np.hstack((self._W1.T.ravel(), self._W2.T.ravel()))

        return initial_nn_params

    def __str__(self):
        return "The configuration of the neural network is:\n\tInput Layer:  %d\n\tHidden Layer: %d\n\tOutput Layer: %d\n\tlmbda: %f\n\tmaxiter: %d" \
                % (self._n_features, self._n_hiddens, self._n_labels, self._lmbda, self._maxiter)

if __name__ == '__main__':
    nn = NeuralNet(400, 25, 10)
    print(nn)
