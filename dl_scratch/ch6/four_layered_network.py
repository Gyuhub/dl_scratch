# class module of two layered neural network

from dl_scratch.common import *
from collections import OrderedDict

class FourLayeredNet:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        # initialize weight parameters
        self.params = {}
        self.params['W1'] = weight_init_std * \
                            np.random.randn(input_size, hidden_size[0])
        self.params['b1'] = np.zeros(hidden_size[0])
        self.params['W2'] = weight_init_std * \
                            np.random.randn(hidden_size[0], hidden_size[1])
        self.params['b2'] = np.zeros(hidden_size[1])
        self.params['W3'] = weight_init_std * \
                            np.random.randn(hidden_size[1], hidden_size[2])
        self.params['b3'] = np.zeros(hidden_size[2])
        self.params['W4'] = weight_init_std * \
                            np.random.randn(hidden_size[2], output_size)
        self.params['b4'] = np.zeros(output_size)

        # initialize layers
        self.layers = OrderedDict()
        self.layers['Affine1'] = AffineLayer(self.params['W1'], self.params['b1'])
        self.layers['Relu1'] = ReluLayer()
        self.layers['Affine2'] = AffineLayer(self.params['W2'], self.params['b2'])
        self.layers['Relu2'] = ReluLayer()
        self.layers['Affine3'] = AffineLayer(self.params['W3'], self.params['b3'])
        self.layers['Relu3'] = ReluLayer()
        self.layers['Affine4'] = AffineLayer(self.params['W4'], self.params['b4'])
        self.lastlayer = SoftmaxWithLossLayer()

    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)
        return x

    def loss(self, x, t):
        y = self.predict(x)
        return self.lastlayer.forward(y, t)

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        if t.ndim != 1 : t = np.argmax(t, axis=1)

        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

    def numerical_gradient(self, x, t):
        loss_W = lambda W: self.loss(x, t)

        grads = {}
        grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
        grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])
        grads['W3'] = numerical_gradient(loss_W, self.params['W3'])
        grads['b3'] = numerical_gradient(loss_W, self.params['b3'])
        grads['W4'] = numerical_gradient(loss_W, self.params['W4'])
        grads['b4'] = numerical_gradient(loss_W, self.params['b4'])
        return grads
    
    def gradient(self, x, t):
        # forward
        self.loss(x, t)

        # backward
        dout = 1 # dL/dL = 1
        dout = self.lastlayer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse() # reverse order because there is order in layers
        for layer in layers:
            dout = layer.backward(dout)

        # result
        grads = {}
        grads['W1'], grads['b1'] = self.layers['Affine1'].dW, self.layers['Affine1'].db
        grads['W2'], grads['b2'] = self.layers['Affine2'].dW, self.layers['Affine2'].db
        grads['W3'], grads['b3'] = self.layers['Affine3'].dW, self.layers['Affine3'].db
        grads['W4'], grads['b4'] = self.layers['Affine4'].dW, self.layers['Affine4'].db

        return grads