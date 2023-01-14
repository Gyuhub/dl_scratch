# practice an example of two layered network
from dl_scratch.common import *

x = np.array([0.8, 0.6])
w1 = np.array([[0.2, 0.4, 0.6], [0.9, 0.6, 0.3]])
b1 = np.array([0.3, 0.4, 0.2])

print(x.shape)
print(w1.shape)
print(b1.shape)

a1 = np.dot(x, w1) + b1 #a1 = x*w1 + b1
z1 = sigmoid(a1)

print(a1)
print(z1)
