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
    return y_hat


def gamma(x, a, b):
    '''
    :param x: input values
    :param a: a is the shape parameter of the gamma distribution
    :param b: b is the scale parameter of the gamma distribution
    :return: output values
    '''
    y_hat = (b ** a) * (x ** (a - 1)) * np.exp(-b * x) / np.math.gamma(a)
    return y_hat


def gaussian(x, amplitude, xo, sigma, offset):
    xo = float(xo)
    a = 1/(2*sigma**2)
    g = offset + amplitude*np.exp(-a*((x-xo)**2))
    return g


def circular_gaussian(xy, amplitude, xo, yo, sigma, theta, offset):
    x, y = xy
    xo = float(xo)
    yo = float(yo)
    a = (np.cos(theta)**2)/(2*sigma**2) + (np.sin(theta)**2)/(2*sigma**2)
    b = -(np.sin(2*theta))/(4*sigma**2) + (np.sin(2*theta))/(4*sigma**2)
    c = (np.sin(theta)**2)/(2*sigma**2) + (np.cos(theta)**2)/(2*sigma**2)
    g = offset + amplitude*np.exp( - (a*((x-xo)**2) + 2*b*(x-xo)*(y-yo)
                            + c*((y-yo)**2)))
    return g.ravel()