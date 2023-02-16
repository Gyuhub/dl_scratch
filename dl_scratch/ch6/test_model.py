# test the model which trained with two_layered_network_learning.py

from dl_scratch.common import *
from data.mnist import load_mnist
import dl_scratch
import numpy as np
import pickle

net = None
with open(dl_scratch.__path__[0] + '/four_layered_network_AdaGrad.pkl', 'rb') as f:
    net = pickle.load(f)

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

if __name__ == '__main__':
    test_size = x_test.shape[0]
    batch_size = 100
    batch_mask = np.random.choice(test_size, batch_size)
    x_batch = x_test[batch_mask]
    t_batch = t_test[batch_mask]

    y_batch = net.predict(x_batch)
    y_batch = np.argmax(y_batch, axis=1)
    if t_batch.ndim != 1 : t_batch = np.argmax(t_batch, axis=1)

    test_acc = np.sum(y_batch == t_batch) / float(x_batch.shape[0])
    res = y_batch[y_batch != t_batch]
    ans = t_batch[y_batch != t_batch]

    print('Classification result: ' + str(test_acc))
    print('net: ' + str(res))
    print('ans: ' + str(ans))