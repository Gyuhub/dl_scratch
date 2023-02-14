# practice an example of two layered network learning

from dl_scratch.common import *
from dl_scratch.ch5.two_layered_network import TwoLayeredNet
from data.mnist import load_mnist # get a module for MNIST dataset
import time
import pickle

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
    # initialize network
    net = TwoLayeredNet(input_size=784, hidden_size=50, output_size=10)
    start_time = time.time() # calculate start time
    for i in range(iters_num):
        batch_mask = np.random.choice(train_size, batch_size)
        x_batch = x_train[batch_mask]
        t_batch = t_train[batch_mask]

        # calculate gradient
        # grad = net.numerical_gradient(x_batch, t_batch) # numerical gradient
        grad = net.gradient(x_batch, t_batch) # backpropagation method

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

    # save weights to pickle data
    import dl_scratch
    with open(dl_scratch.__path__[0] + '/two_layered_network.pkl', 'wb') as f:
        pickle.dump(net, f)