# compare optimizers with MNIST dataset

from dl_scratch.common import *
from dl_scratch.ch6.four_layered_network import FourLayeredNet
from data.mnist import load_mnist # get a module for MNIST dataset
import time
import pickle

(x_train, t_train), (x_test, t_test) = \
    load_mnist(normalize=True, one_hot_label=True)

# hyper parameter
iters_num = 10000
train_size = x_train.shape[0]
batch_size = 100
learing_rate = 0.01

train_loss_list = [[], [], [], []]
train_acc_list = [[], [], [], []]
test_acc_list = [[], [], [], []]

# iteration number per epoch
iter_per_epoch = max(train_size / batch_size, 1)

if __name__ == '__main__':
    # initialize network
    net, optimizer = [], []
    for _ in range (4):
        net.append(FourLayeredNet(input_size=784, hidden_size=[100, 100, 100], output_size=10))
    optimizer.append(SGD(lr=learing_rate * 10))
    optimizer.append(Momentum(lr=learing_rate))
    optimizer.append(AdaGrad(lr=learing_rate))
    optimizer.append(Adam(lr=learing_rate))
    optimizer_name = ['SGD','Momentum','AdaGrad','Adam']
    start_time = time.time() # calculate start time
    for i in range(iters_num):
        for j in range(4):
            batch_mask = np.random.choice(train_size, batch_size)
            x_batch = x_train[batch_mask]
            t_batch = t_train[batch_mask]

            # calculate gradient
            grad = net[j].gradient(x_batch, t_batch) # backpropagation method

            # update parameters
            optimizer[j].update(net[j].params, grad)

            # record learning status
            loss = net[j].loss(x_batch, t_batch)
            train_loss_list[j].append(loss)
            
            # calculate accuracy of model per epoch
            if i % iter_per_epoch == 0:
                train_acc = net[j].accuracy(x_train, t_train)
                test_acc = net[j].accuracy(x_test, t_test)
                train_acc_list[j].append(train_acc)
                test_acc_list[j].append(test_acc)
                print("\ntrain acc, test acc | " + str(train_acc) + ", " + str(test_acc))

        # print current status of learning
        print('Elapsed time[s]: %.3f, current iter: %d, current %s loss: %.4f'
         % (time.time() - start_time, i, optimizer_name[0], loss), end='\r')

    plt.figure(1)
    plt.plot(np.arange(0, iters_num), train_loss_list[0], '--Dk', label=optimizer_name[0], linewidth=2)
    plt.plot(np.arange(0, iters_num), train_loss_list[1], '--Dr', label=optimizer_name[1], linewidth=2)
    plt.plot(np.arange(0, iters_num), train_loss_list[2], '--Dg', label=optimizer_name[2], linewidth=2)
    plt.plot(np.arange(0, iters_num), train_loss_list[3], '--Db', label=optimizer_name[3], linewidth=2)
    plt.xlabel('iterations')
    plt.ylabel('loss')
    plt.title('Comparison of learning upon optimizers')
    plt.legend()
    plt.grid()

    plt.figure(2)
    for i in range(4):
        plt.subplot(2, 2, i+1)
        markers = {'train': 'o', 'test': 's'}
        x = np.arange(len(train_acc_list[i][:]))
        plt.plot(x, train_acc_list[i][:], label=('train acc of ' + optimizer_name[i]))
        plt.plot(x, test_acc_list[i][:], label=('test acc of ' + optimizer_name[i]), linestyle='--')
        plt.xlabel("epochs")
        plt.ylabel("accuracy")
        plt.ylim(0, 1.0)
        plt.legend(loc='lower right')
        plt.grid()
    plt.show()

    # save best weights to pickle data
    smallest_loss = train_loss_list[0][-1]
    smallest_net = 0
    for i in range(4):
        if smallest_loss < train_loss_list[i][-1]:
            smallest_net = i
            smallest_loss = train_loss_list[i][-1]
    print(optimizer_name[smallest_net] + ' is the best fitted network!')

    import dl_scratch
    with open(dl_scratch.__path__[0] + '/four_layered_network_' + optimizer_name[smallest_net] + '.pkl', 'wb') as f:
        pickle.dump(net[smallest_net], f)
