import numpy as np

def randInitWeights(L_in, L_out):
    """
    Randomly initialize the weights of with L_in incoming connections
    and L_out outgoing connections

    W should be set to a matrix of size(L_out, 1 + L_in) as the column
    row of W handles the "bias" terms. The first row of W corresponds
    to the parameters for the bias units

    One effective strategy is to select values for W uniformly in the range [-ep, ep]
    A good choice of ep is sqrt(6) / sqrt(L_in + L_out)
    """

    epsilon = np.sqrt(6) / (np.sqrt(L_in + L_out))
    W = np.random.rand(L_out, 1 + L_in) * 2 * epsilon - epsilon

    return W


def sigmoid(z):
    """
    Calculate the sigmoid values of vector z
    @return An 1D vector
    """
    return 1.0 / (1 + np.exp(-z))

def sigmoidGradient(z):
    """
    Calculate the sigmoid gradient of vector z
    @return An 1D vector
    """
    s = sigmoid(z)
    return np.multiply(s, (1 - s))

# Test sigmoid gradient function
def test_sigmoid_gradient():
    print('\tSigmoid gradient evaluated at [-1, -0.5, 0, 0.5, 1]')
    print(sigmoidGradient(np.array([-1, -0.5, 0, 0.5, 1])))
    print('\tThe sigmoid gradient should be about [0.196612 0.235004 0.250000 0.235004 0.196612]')