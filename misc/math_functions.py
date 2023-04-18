import numpy as np

def sigmoid(x, L, x0, k, b):
    '''
    within param_tuple
    :param x: input values
    :param L: L is responsible for scaling the output range from [0,1] to [0,L]
    :param b: b adds bias to the output and changes its range from [0,L] to [b,L+b]
    :param k: k is responsible for scaling the input, which remains in (-inf,inf)
    :param x0: x0 is the point in the middle of the Sigmoid, i.e. the point where Sigmoid should originally output the value 1/2 [since if x=x0, we get 1/(1+exp(0)) = 1/2].
    :return: output values
    '''
    y_hat = L / (1 + np.exp(-k * (x - x0))) + b
    return (y_hat)

