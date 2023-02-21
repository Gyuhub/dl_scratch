# compare output of activation function about various initial weight parameters

from dl_scratch.common import *
import time

def tanh(x):
    return (np.exp(2*x)-1)/(np.exp(2*x)+1)

if __name__ == '__main__':
    input_size = 1000
    layer_size = 100
    X = np.random.randn(input_size, layer_size)
    hidden_layer_size = 5
    activations = {}
    weight_std = np.sqrt(1 / layer_size) # standard deviation of weight

    for i in range(hidden_layer_size):
        if i != 0:
            X = activations[i - 1]

        # initialize weight matrix
        W = np.random.randn(layer_size, layer_size) * weight_std
        a = np.dot(X, W) # affine transformation
        # z = sigmoid(a) # activation function
        # z = tanh(a) # activation function
        z = relu(a) # activation function
        activations[i] = z # store result
    
    # plt.figure()
    # x = np.arange(-5,5,0.1)
    # y1 = sigmoid(x)
    # y2 = tanh(x)
    # plt.plot(x,y1,'r',label='sigmoid')
    # plt.plot(x,y2,'b',label='hyperbolic tangent')
    # plt.title('Activation functions')
    # plt.xlabel('x')
    # plt.ylabel('h(x)')
    # plt.legend()
    # plt.grid()
    # plt.show()

    plt.figure(figsize=(20,5))
    for i, a in activations.items():
        plt.subplot(1, hidden_layer_size, i + 1)
        plt.title(str(i + 1) + '-layer')
        plt.hist(a.flatten(), 60, range=(0.001, 1))
    plt.show()