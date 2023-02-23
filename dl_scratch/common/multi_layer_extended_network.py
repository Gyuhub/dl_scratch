import numpy as np
from dl_scratch.common.functions import *
from dl_scratch.common.layers import *
from dl_scratch.common.optimizers import *
from dl_scratch.common.gradient import numerical_gradient
from collections import OrderedDict

class MultiLayeredExtendedNet:
    '''
     Parameters
    ----------
    input_size : size of input layer neuron
    hidden_size_list : list containing size of each hidden layer e.g. [100, 100, 100]
    output_size : size of output layer class
    activation : name activation function - 'relu' or 'sigmoid'
    weight_init_std : standard deviation of weight (e.g. 0.01)
        'relu' or 'he' set 'He initialization'
        'sigmoid' or 'xavier' set 'Xavier initialization'
    weight_decay_lambda : weight decaying ratio (law of L2)
    use_dropout : true if you use dropout or false
    dropout_ration : ratio of dropout if you use dropout
    use_batch_norm : true if you use batch normalization or false
    '''
    def __init__(self, input_size, hidden_size_list, output_size,
                 activation='relu', weight_init_std='relu', weight_decay_lambda=0,
                 use_dropout=False, dropout_ration=0.0, use_batch_norm=False):
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size_list = hidden_size_list
        self.hidden_layer_num = len(hidden_size_list)
        self.weight_decay_lambda = weight_decay_lambda
        self.use_dropout = use_dropout
        self.dropout_ration = dropout_ration
        self.use_batch_norm = use_batch_norm
        self.params = {}

        # initialize weight
        self.__init_weight(weight_init_std)

        # create layers
        activation_layer = {'sigmoid': SigmoidLayer, 'relu': ReluLayer}
        self.layers = OrderedDict()
        for idx in range(1, self.hidden_layer_num+1):
            self.layers['Affine' + str(idx)] = AffineLayer(self.params['W' + str(idx)],
                                                           self.params['b' + str(idx)])
            if self.use_batch_norm:
                self.params['gamma' + str(idx)] = np.ones(hidden_size_list[idx-1])
                self.params['beta' + str(idx)] = np.zeros(hidden_size_list[idx-1])
                self.layers['BatchNorm' + str(idx)] = BatchNormLayer(self.params['gamma' + str(idx)], self.params['beta' + str(idx)])

            self.layers['Activation_function' + str(idx)] = activation_layer[activation]()

            # if self.use_dropout:
            #     self.layers['Dropout' + str(idx)] = Dropout(dropout_ration)

        idx = self.hidden_layer_num + 1
        self.layers['Affine' + str(idx)] = AffineLayer(self.params['W' + str(idx)],
            self.params['b' + str(idx)])

        self.last_layer = SoftmaxWithLossLayer()

    def __init_weight(self, weight_init_std):
        """initialize weight
        
        Parameters
        ----------
        weight_init_std : standard deviation of normal distribution (e.g. 0.01)
            'relu' or 'he' set 'He initialization'
            'sigmoid' or 'xavier' set 'Xavier initialization'
        """
        all_size_list = [self.input_size] + self.hidden_size_list + [self.output_size]
        for idx in range(1, len(all_size_list)):
            scale = weight_init_std
            if str(weight_init_std).lower() in ('relu', 'he'):
                scale = np.sqrt(2.0 / all_size_list[idx - 1])
            elif str(weight_init_std).lower() in ('sigmoid', 'xavier'):
                scale = np.sqrt(1.0 / all_size_list[idx - 1])
            self.params['W' + str(idx)] = scale * np.random.randn(all_size_list[idx-1], all_size_list[idx])
            self.params['b' + str(idx)] = np.zeros(all_size_list[idx])

    def predict(self, x, train_flag=False):
        for key, layer in self.layers.items():
            if 'Dropout' in key or 'BatchNorm' in key:
                x = layer.forward(x, train_flag)
            else:
                x = layer.forward(x)

        return x

    def loss(self, x, t, train_flag=False):
        """calculate loss
        
        Parameters
        ----------
        x : input data
        t : target value
        train_flag : true if you want to train network or false
        
        Returns
        -------
        value of loss function
        """
        y = self.predict(x, train_flag) # predicted output

        weight_decay = 0
        for idx in range(1, self.hidden_layer_num + 2):
            W = self.params['W' + str(idx)]
            weight_decay += 0.5 * self.weight_decay_lambda * np.sum(W ** 2)

        return self.last_layer.forward(y, t) + weight_decay

    def accuracy(self, x, t):
        y = self.predict(x, train_flag=False)
        y = np.argmax(y, axis=1)
        if t.ndim != 1 : t = np.argmax(t, axis=1)

        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

    def numerical_gradient(self, x, t):
        """calculate numerical gradient
        
        Parameters
        ----------
        x : input data
        t : target value
        
        Returns
        -------
        dictionary variable contains gradient of each layer
            grads['W1']、grads['W2']、... gradinet of each layer
            grads['b1']、grads['b2']、... bias of each layer
        """
        loss_W = lambda W: self.loss(x, t)

        grads = {}
        for idx in range(1, self.hidden_layer_num + 2):
            grads['W' + str(idx)] = numerical_gradient(loss_W, self.params['W' + str(idx)])
            grads['b' + str(idx)] = numerical_gradient(loss_W, self.params['b' + str(idx)])

            if self.use_batch_norm and idx != self.hidden_layer_num + 1:
                grads['gamma' + str(idx)] = numerical_gradient(loss_W, self.params['gamma' + str(idx)])
                grads['beta' + str(idx)] = numerical_gradient(loss_W, self.params['beta' + str(idx)])

        return grads

    def gradient(self, x, t):
        """calculate gradient using backpropagation

        Parameters
        ----------
        x : input data
        t : target value
        
        Returns
        -------
        dictionary variable contains gradient of each layer
            grads['W1']、grads['W2']、... gradinet of each layer
            grads['b1']、grads['b2']、... bias of each layer
        """
        # forward
        self.loss(x, t, train_flag=True)

        # backward
        dout = 1
        dout = self.last_layer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        # store results
        grads = {}
        for idx in range(1, self.hidden_layer_num + 2):
            grads['W' + str(idx)] = self.layers['Affine' + str(idx)].dW + self.weight_decay_lambda * self.layers['Affine' + str(idx)].W
            grads['b' + str(idx)] = self.layers['Affine' + str(idx)].db

            if self.use_batch_norm and idx != self.hidden_layer_num + 1:
                grads['gamma' + str(idx)] = self.layers['BatchNorm' + str(idx)].dgamma
                grads['beta' + str(idx)] = self.layers['BatchNorm' + str(idx)].dbeta

        return grads