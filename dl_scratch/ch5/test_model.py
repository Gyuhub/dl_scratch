# test the model which trained with two_layered_network_learning.py

from dl_scratch.common import *
from data.mnist import load_mnist
import dl_scratch
import numpy as np
import pickle
from PIL import Image

net = None
with open(dl_scratch.__path__[0] + '/two_layered_network.pkl', 'rb') as f:
    net = pickle.load(f)

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=False, one_hot_label=True)

if __name__ == '__main__':
    flag = False
    while flag == False:
        idx = int(input("Enter an index(integer) of dataset you want to choose (0 ~ 9999): "))
        if idx < 0 or idx > 9999:
            print('Get wrong number of integer! Please try again...')
            flag = False
        else:
            flag = True    
    x = x_test[idx]
    t = t_test[idx]
    
    img = x.reshape(28, 28)
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()

    # normalize data
    x_norm = x.copy()
    x_norm = x_norm / np.linalg.norm(x)
    out = np.argmax(net.predict(x_norm.reshape(1, 784)))
    ans = np.argmax(t)

    print('Classification result: ' + str(out))
    print('Answer: ' + str(ans))