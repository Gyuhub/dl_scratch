# train CNN with MNIST dataset

from dl_scratch.common import *
from dl_scratch.ch7.simple_conv_net import SimpleConvNet
from data.mnist import load_mnist

(x_train, t_train), (x_test, t_test) = load_mnist(flatten=False)

max_epochs = 20

network = SimpleConvNet(input_dim=(1, 28, 28),
                        conv_param={'filter_num': 30, 'filter_size': 5, 'pad': 0, 'stride': 1},
                        hidden_size=100, output_size=10, weight_init_std=0.01)

trainer = Trainer(network, x_train, t_train, x_test, t_test,
                  epochs=max_epochs, mini_batch_size=100,
                  optimizer='Adam', optimizer_param={'lr': 0.001},
                  evaluate_sample_num_per_epoch=1000)

if __name__ == '__main__':
    # trainer.train()

    # network.save_params('params.pkl')
    # print('Successfully saved the parameters of network')
    network.load_params('params.pkl')
    print('Successfully loaded the parameters of network')
    mask = np.random.choice(x_test.shape[0], 1)
    x = x_test[mask]
    t = t_test[mask]
    y = network.predict(x)
    print('output: %d' % (np.argmax(y)))
    print('answer: %d' % (t))

    # markers = {'train': 'o', 'test': 's'}
    # x = np.arange(max_epochs)
    # plt.plot(x, trainer.train_acc_list, marker='o', label='train', markevery=2)
    # plt.plot(x, trainer.test_acc_list, marker='s', label='test', markevery=2)
    # plt.xlabel("epochs")
    # plt.ylabel("accuracy")
    # plt.title('Accuracy of simple CNN')
    # plt.ylim(0, 1.0)
    # plt.legend(loc='lower right')
    # plt.grid()
    # plt.show()