# practice an example of forward propagation

from dl_scratch.common import *
from data.mnist import load_mnist # get a module for MNIST dataset
from PIL import Image
import pickle

# normalize: make value of pixel of input image normalized
# flatten: make array of input image to 1-dimension array
# one_hot_label: make label one-hot encoding
def get_data():
    (x_train, t_train), (x_test, t_test) = \
        load_mnist(normalize=True, flatten=True, one_hot_label=False)
    return x_test, t_test

def init_network():
    with open("./sample_weight.pkl", 'rb') as f:
        network = pickle.load(f)
    return network

def predict(network, x):
    w1, w2, w3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x,  w1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, w2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, w3) + b3
    y = softmax(a3)
    return y

def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()

x, t = get_data()
network = init_network()

# not using batch
accuracy_cnt = 0
for i in range(len(x)):
    y = predict(network, x[i])
    p = np.argmax(y) # get index of biggest output value
    if p == t[i]:
        accuracy_cnt += 1
print('Accuracy(not using batch): ' + str(float(accuracy_cnt / len(x))))
# using batch
batch_size = 100
accuracy_cnt = 0
for i in range(0, len(x), batch_size):
    y_batch = predict(network, x[i:i+batch_size])
    p = np.argmax(y_batch, axis=1)
    accuracy_cnt += np.sum(p == t[i:i+batch_size])    
print('Accuracy(using batch): ' + str(float(accuracy_cnt / len(x))))