# plot graph of activation functions
from dl_scratch.common import *

x = np.arange(-5.0, 5.0, 0.1)
y1 = step(x)
y2 = sigmoid(x)
y3 = relu(x)
# step & sigmoid
fig1 = plt.figure(1)
plt.plot(x, y1, label='step', linestyle='--')
plt.plot(x, y2, label='sigmoid')
plt.title('step & sigmoid')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.ylim(-0.1, 1.1)
plt.grid()
# relu
def lse(x):
    x_max = np.max(x)
    return (np.log(np.exp(x-x_max)+np.exp(-x_max)) + x_max)
y4 = lse(x)
fig2 = plt.figure(2)
plt.plot(x, y3, label='relu', linestyle='--')
plt.plot(x, y4, label='lse')
plt.title('ReLU & LSE')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid()
plt.show()
