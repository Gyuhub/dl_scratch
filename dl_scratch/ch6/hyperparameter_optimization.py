# compare optimizers with MNIST dataset

from dl_scratch.common import *
from data.mnist import load_mnist # get a module for MNIST dataset
import time

(x_train, t_train), (x_test, t_test) = \
    load_mnist(normalize=True, one_hot_label=True)

# overfitting
x_train = x_train[:500]
t_train = t_train[:500]

# validation dataset
validation_ratio = 0.2
validation_range = int(x_train.shape[0] * validation_ratio)
permutation = np.random.permutation(x_train.shape[0])
x_train = x_train[permutation,:] if x_train.ndim == 2 else x_train[permutation,:,:,:]
t_train = t_train[permutation]
x_valid = x_train[:validation_range]
t_valid = t_train[:validation_range]
x_train = x_train[validation_range:]
t_train = t_train[validation_range:]

def __train(lr, weight_decay, epocs=50):
    network = MultiLayeredNet(input_size=784, hidden_size_list=[100, 100, 100, 100, 100, 100],
                            output_size=10, weight_decay_lambda=weight_decay)
    trainer = Trainer(network, x_train, t_train, x_valid, t_valid,
                      epochs=epocs, mini_batch_size=100,
                      optimizer='sgd', optimizer_param={'lr': lr}, verbose=False)
    trainer.train()

    return trainer.test_acc_list, trainer.train_acc_list

optimization_iters_num = 100
train_acc_dict = {}
valid_acc_dict = {}

if __name__ == '__main__':
    # initialize network
    start_time = time.time() # calculate start time
    for _ in range(optimization_iters_num):
        weight_decay = 10 ** np.random.uniform(-8, -3)
        lr = 10 ** np.random.uniform(-6, -1)

        val_acc_list, train_acc_list = __train(lr, weight_decay)
        print('Elapsed time: [' + str(time.time() - start_time) + "], val acc:" + str(val_acc_list[-1]) + " | lr:" + str(lr) + ", weight decay:" + str(weight_decay))
        key = "lr:" + str(lr) + ", weight decay:" + str(weight_decay)
        valid_acc_dict[key] = val_acc_list
        train_acc_dict[key] = train_acc_list


    plt.figure(1)
    j = 1
    plt.title('Accuracy about train and validation dataset')
    for key, valid_acc_list in sorted(valid_acc_dict.items(), key=lambda item: item[1][-1], reverse=True):
        if j > 6:
            break
        plt.subplot(2, 3, j)
        x = np.arange(len(valid_acc_list))
        plt.plot(x, train_acc_dict[key], label=('train acc'))
        plt.plot(list((i)*10 for i in range(int(len(train_acc_dict[key])/10))), train_acc_dict[key][0:len(train_acc_dict[key]):10], '*')
        plt.plot(x, valid_acc_list, label=('validation acc'), linestyle='--')
        plt.plot(list((i)*10 for i in range(int(len(valid_acc_list)/10))), valid_acc_list[0:len(valid_acc_list):10], '*')
        plt.xlabel("epochs")
        plt.ylabel("accuracy")
        plt.ylim(0, 1.0)
        plt.legend(loc='lower right')
        plt.grid()
        print('Best ' + str(j) + ' (' + str(valid_acc_list[-1]) + '): ' + key)
        j += 1
    plt.show()
