import numpy as np
from dl_scratch.common.functions import softmax, cross_entropy_error

class MulLayer:
  def __init__(self):
      self.x = None
      self.y = None

  def forward(self, x, y):
      self.x = x
      self.y = y
      out = x * y
      return out

  def backward(self, dout):
      dx = dout * self.y
      dy = dout * self.x
      return dx, dy

class AddLayer:
  def __init__(self):
      pass # do not anything

  def forward(self, x, y):
      out = x + y
      return out

  def backward(self, dout):
      dx = dout * 1
      dy = dout * 1
      return dx, dy

class ReluLayer:
    def __init__(self):
        self.mask = None # boolean numpy array. true when x <= 0, false when others

    def forward(self, x):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0
        return out

    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout
        return dx

class SigmoidLayer:
    def __init__(self):
        self.out = None

    def forward(self, x):
        self.out = 1 / (1 + np.exp(-x))
        return self.out

    def backward(self, dout):
        dx = dout * self.out * (1.0 - self.out)
        return dx

class AffineLayer:
    def __init__(self, W, b):
        self.W = W
        self.b = b
        self.x = None
        self.original_x_shape = None
        self.dW = None
        self.db = None

    def forward(self, x):
        self.original_x_shape = x.shape # for tensor input
        x = x.reshape(x.shape[0], -1)
        self.x = x
        return (np.dot(self.x, self.W) + self.b)

    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)
        dx = dx.reshape(*self.original_x_shape) # for tensor input
        return dx

class SoftmaxWithLossLayer:
    def __init__(self):
        self.loss = None
        self.y = None
        self.t = None

    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, self.t)
        return self.loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        dx = (self.y - self.t) / batch_size
        return dx