# compare effect of batch normalization with MNIST dataset

from dl_scratch.common import *
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

train_loss = {}
train_acc = {}
test_acc = {}

# iteration number per epoch
iter_per_epoch = max(train_size / batch_size, 1)

if __name__ == '__main__':
    # initialize network
    net_types = {'std=0.01': 0.01, 'Xavier': 'xavier', 'He': 'he'}
    nets = {}
    for key, net_type in net_types.items():
        nets[key] = MultiLayeredExtendedNet(input_size=784, hidden_size_list=[100, 100, 100, 100], output_size=10,
                                             weight_init_std=net_type, use_batch_norm=True)
        train_loss[key] = []
        train_acc[key] = []
        test_acc[key] = []

    optimizer = SGD(lr=learing_rate)
    start_time = time.time() # calculate start time
    for i in range(iters_num):
        batch_mask = np.random.choice(train_size, batch_size)
        x_batch = x_train[batch_mask]
        t_batch = t_train[batch_mask]

        for key in nets.keys():
            # calculate gradient
            grads = nets[key].gradient(x_batch, t_batch) # backpropagation method

            # update parameters
            optimizer.update(nets[key].params, grads)

            # record learning status
            loss = nets[key].loss(x_batch, t_batch)
            train_loss[key].append(loss)
            
            # calculate accuracy of model per epoch
            if i % iter_per_epoch == 0:
                train_acc_val = nets[key].accuracy(x_train, t_train)
                test_acc_val = nets[key].accuracy(x_test, t_test)
                train_acc[key].append(train_acc_val)
                test_acc[key].append(test_acc_val)
                print("\ntrain acc, test acc | " + str(train_acc_val) + ", " + str(test_acc_val))

        # print current status of learning
        print('Elapsed time[s]: %.3f, current iter: %d, current [%s] loss: %.4f'
         % (time.time() - start_time, i, key, loss), end='\r')

    plt.figure(1)
    plt.plot(np.arange(0, iters_num / 5), train_loss['std=0.01'][0:2000], '--Dr', label='std=0.01', linewidth=1)
    plt.plot(np.arange(0, iters_num / 5), train_loss['Xavier'][0:2000], '--Dg', label='Xavier', linewidth=1)
    plt.plot(np.arange(0, iters_num / 5), train_loss['He'][0:2000], '--Db', label='He', linewidth=1)
    plt.xlabel('iterations')
    plt.ylabel('loss')
    plt.title('Comparison of learning upon effect of batch normalization')
    plt.legend()
    plt.grid()

    plt.figure(2)
    i = 1
    for key in net_types.keys():
        plt.subplot(1, 3, i)
        markers = {'train': 'o', 'test': 's'}
        x = np.arange(len(train_acc[key][:]))
        plt.plot(x, train_acc[key][:], label=('train acc of ' + key))
        plt.plot(x, test_acc[key][:], label=('test acc of ' + key), linestyle='--')
        plt.xlabel("epochs")
        plt.ylabel("accuracy")
        plt.ylim(0, 1.0)
        plt.legend(loc='lower right')
        plt.grid()
        i += 1
    plt.show()

    # save best weights to pickle data
    smallest_loss = train_loss['std=0.01'][-1]
    smallest_net = str()
    for key in net_types.keys():
        if smallest_loss < train_loss[key][-1]:
            smallest_net = key
            smallest_loss = train_loss[key][-1]
    print(str(smallest_net) + ' is the best fitted network!')

    import dl_scratch
    with open(dl_scratch.__path__[0] + '/five_layered_network_' + str(smallest_net) + '.pkl', 'wb') as f:
        pickle.dump(nets[smallest_net], f)
