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
