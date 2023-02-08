import numpy as np

def step(x):
    y = x > 0
    return y.astype(np.int)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return np.maximum(x, 0)

def identity(x):
    return x

def softmax(x):
    x_max = np.max(x)
    x_exp = np.exp(x - x_max)
    x_exp_sum = np.sum(x_exp)
    return x_exp / x_exp_sum

def sum_squares_error(y, t):
    return 0.5 * np.sum((y-t)**2)

def cross_entropy_error(y, t):
    if y.dim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size
