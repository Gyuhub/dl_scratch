from dl_scratch.common import *
from data.mnist import load_mnist
import dl_scratch
import numpy as np
import pickle

net = None
with open(dl_scratch.__path__[0] + '/two_layered_network.pkl', 'rb') as f:
    net = pickle.load(f)

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

if __name__ == '__main__':
    x_batch = x_train[:3]
    t_batch = t_train[:3]

    grad_numerical = net.numerical_gradient(x_batch, t_batch)
    grad_backprop = net.gradient(x_batch, t_batch)

    # calculate mean of absolute error from each parameter
    for key in grad_numerical.keys():
        diff = np.average(np.abs(grad_backprop[key] - grad_numerical[key]))
        print(key + ":" + str(diff))