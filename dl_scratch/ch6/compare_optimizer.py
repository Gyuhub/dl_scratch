# compare optimizers with specific function and plot the result

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

x = np.arange(-10, 10, 0.5)
y = np.arange(-10, 10, 0.5)
x_mesh, y_mesh = np.meshgrid(x,y)

def fn(x):
    return (1/30*x[0]**2 + x[1]**2)
    # return -1/(np.exp(x[0])+np.exp(-x[0])) - 1/(np.exp(x[1])+np.exp(-x[1]))

def numerical_gradient(f, x):
    h = 1e-4
    grads = np.zeros_like(x)

    for i in range(2):
        tmp_val = x[i]
        x[i] = tmp_val+h
        fxh1 = f(x)
        x[i] = tmp_val-h
        fxh2 = f(x)

        grads[i] = ((fxh1-fxh2) / (2*h))
        x[i] = tmp_val

    return grads

z_mesh = fn(np.array([x_mesh, y_mesh]))
fig = plt.figure()
ax = mplot3d.Axes3D(fig)
ax.plot_wireframe(x_mesh, y_mesh, z_mesh, 1, cmap='binary')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('f(x,y)')

z_mesh_grad = numerical_gradient(fn, np.array([x_mesh, y_mesh]))

plt.figure()
plt.contour(x_mesh, y_mesh, z_mesh, levels=np.logspace(-2, 2, 20))
# plt.quiver(x_mesh, y_mesh, -z_mesh_grad[0], -z_mesh_grad[1], color='black', label='gradient')
# plt.xlabel('x')
# plt.ylabel('y')
# plt.title('Contour and gradient')

init_x = np.array([-7.5, -5.0])
sgd = np.array(init_x)
momentum = np.array(init_x)
adagrad = np.array(init_x)
adam = np.array(init_x)
v = np.zeros_like(init_x)
h = 0
v_adam = np.zeros_like(init_x)
m = np.zeros_like(init_x)

# hyper parameter
iter = 20
lr = 1
alpha = 0.9
beta1 = 0.9
beta2 = 0.999

sgd_opt = np.zeros((2, iter))
momentum_opt = np.zeros((2, iter))
adagrad_opt = np.zeros((2, iter))
adam_opt = np.zeros((2, iter))

for i in range(iter):
    sgd_opt[:,i] = sgd
    momentum_opt[:,i] = momentum
    adagrad_opt[:,i] = adagrad
    adam_opt[:,i] = adam

    grad_sgd = numerical_gradient(fn, sgd)
    grad_momentum = numerical_gradient(fn, momentum)
    grad_adagrad = numerical_gradient(fn, adagrad)
    grad_adam = numerical_gradient(fn, adam)
    
    sgd -= lr * grad_sgd # SGD
    v = alpha * v - lr * grad_momentum # Momentum
    momentum = momentum + v
    h += grad_adagrad * grad_adagrad # AdaGrad
    adagrad -= lr * grad_adagrad / (np.sqrt(h) + 1e-7)
    m = beta1 * m + (1 - beta1) * grad_adam # Adam
    v_adam = beta2 * v_adam + (1 - beta2) * grad_adam * grad_adam
    adam -= lr * np.sqrt(1 - beta2**(i+1)) / (1 - beta1**(i+1)) * m / (np.sqrt(v_adam) + 1e-8)

plt.plot(0, 0, 'mo', linewidth=4, label='Optimal point')
plt.plot(sgd_opt[0,:], sgd_opt[1,:], '--ko', label='SGD')
plt.plot(momentum_opt[0,:], momentum_opt[1,:], '--ro', label='Momentum')
plt.plot(adagrad_opt[0,:], adagrad_opt[1,:], '--go', label='AdaGrad')
plt.plot(adam_opt[0,:], adam_opt[1,:], '--bo', label='Adam')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Optimization with gradient')
plt.legend()
plt.show()