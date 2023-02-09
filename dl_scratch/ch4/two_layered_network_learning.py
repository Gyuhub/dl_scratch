# practice an example of two layered network learning

from dl_scratch.common import *
from data.mnist import load_mnist # get a module for MNIST dataset
import time

class TwoLayeredNet:
    def __init__(self, input_size, hidden_size, output_size,
                weight_init_std=0.01):
        # initialize weight parameters
        self.params = {}
        self.params['W1'] = weight_init_std * \
                            np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * \
                            np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

    def predict(self, x):
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']

        z1 = sigmoid(np.dot(x, W1) + b1)
        y = softmax(np.dot(z1, W2) + b2)
        return y

    def loss(self, x, t):
        y = self.predict(x)
        return cross_entropy_error(y, t)

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        t = np.argmax(t, axis=1)

        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

    def numerical_gradient(self, x, t):
        loss_W = lambda W: self.loss(x, t)

        grads = {}
        grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
        grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])
        return grads
    
    def gradient(self, x, t):
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']
        grads = {}
        
        batch_num = x.shape[0]
        
        # forward
        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)
        
        # backward
        dy = (y - t) / batch_num
        grads['W2'] = np.dot(z1.T, dy)
        grads['b2'] = np.sum(dy, axis=0)
        
        da1 = np.dot(dy, W2.T)
        dz1 = sigmoid_grad(a1) * da1
        grads['W1'] = np.dot(x.T, dz1)
        grads['b1'] = np.sum(dz1, axis=0)

        return grads

(x_train, t_train), (x_test, t_test) = \
    load_mnist(normalize=True, one_hot_label=True)

# hyper parameter
iters_num = 10000
train_size = x_train.shape[0]
batch_size = 100
learing_rate = 0.1

train_loss_list = []
train_acc_list = []
test_acc_list = []

# iteration number per epoch
iter_per_epoch = max(train_size / batch_size, 1)

if __name__ == '__main__':
    net = TwoLayeredNet(input_size=784, hidden_size=100, output_size=10)
    start_time = time.time() # calculate start time
    for i in range(iters_num):
        batch_mask = np.random.choice(train_size, batch_size)
        x_batch = x_train[batch_mask]
        t_batch = t_train[batch_mask]

        # calculate gradient
        # grad = net.numerical_gradient(x_batch, t_batch)
        grad = net.gradient(x_batch, t_batch)

        # update parameters
        for key in ('W1', 'b1', 'W2', 'b2'):
            net.params[key] -= learing_rate * grad[key]
            
        # record learning status
        loss = net.loss(x_batch, t_batch)
        train_loss_list.append(loss)
        
        # calculate accuracy of model per epoch
        if i % iter_per_epoch == 0:
            train_acc = net.accuracy(x_train, t_train)
            test_acc = net.accuracy(x_test, t_test)
            train_acc_list.append(train_acc)
            test_acc_list.append(test_acc)
            print("train acc, test acc | " + str(train_acc) + ", " + str(test_acc))

        # print current status of learning
        print('Elapsed time[s]: %.3f, current iter: %d, current loss: %f' % (time.time() - start_time, i, loss), end='\r')

    plt.figure(1)
    plt.plot(np.arange(0, iters_num), train_loss_list, 'k*')
    plt.xlabel('iterations')
    plt.ylabel('loss')
    plt.grid()
    plt.show()

    plt.figure(2)
    markers = {'train': 'o', 'test': 's'}
    x = np.arange(len(train_acc_list))
    plt.plot(x, train_acc_list, label='train acc')
    plt.plot(x, test_acc_list, label='test acc', linestyle='--')
    plt.xlabel("epochs")
    plt.ylabel("accuracy")
    plt.ylim(0, 1.0)
    plt.legend(loc='lower right')
    plt.grid()
    plt.show()