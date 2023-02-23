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

class BatchNormLayer:
     def __init__(self, gamma, beta, momentum=0.9, rolling_mean=None, rolling_var=None):
          self.gamma = gamma
          self.beta = beta
          self.momentum = momentum
          self.input_shape = None # 4-D for convolution layer, 2-D for affine layer

          # used when the network is run for testing, not learning
          self.rolling_mean = rolling_mean
          self.rolling_var = rolling_var

          # used when backpropagation
          self.batch_size = None
          self.xd = None # deviation between data and mean
          self.std = None # standarad deviation
          self.xhat = None # normalized data
          self.dgamma = None
          self.dbeta = None

     def forward(self, x, train_flag=True):
          self.input_shape = x.shape
          if x.ndim != 2:
               N, C, H, W = x.shape # batch, channel, height, width
               x = x.reshape(N, -1)
          
          out = self.__forward(x,train_flag)
          return out.reshape(*self.input_shape)

     def __forward(self, x, train_flag):
          if self.rolling_mean == None:
               N, D = x.shape
               self.rolling_mean = np.zeros(D)
               self.rolling_var = np.zeros(D)

          if train_flag:
               mu = np.mean(x, axis=0)
               xd = x - mu
               var = np.mean(xd**2, axis=0)
               std = np.sqrt(var + 10e-7)
               xhat = xd / std

               self.batch_size = x.shape[0]
               self.xd = xd
               self.std = std
               self.xhat = xhat

               # Exponential Moving Average (EMA)
               self.rolling_mean = self.momentum * self.rolling_mean (1 - self.momentum) * mu
               self.rolling_var = self.momentum * self.rolling_var (1 - self.momentum) * var
          else:
               xd = x - self.rolling_mean
               xhat = xd / (np.sqrt(self.running_var + 10e-7))
          
          out = self.gamma * xhat + self.beta
          return out

     def backward(self, dout):
          if dout.ndim != 2:
               N, C, H, W = dout.shape # batch, channel, height, width
               dout = dout.reshape(N, -1)
          
          dx = self.__backward(dout)
          dx = dx.reshape(*self.input_shape)
          return dx

     def __backward(self, dout):
          dbeta = np.sum(dout, axis=0)
          dgamma = np.sum(dout * self.xhat, axis=0)
          dxhat = self.gamma * dout
          dxd = dxhat / self.std
          dstd = -np.sum((dxhat * self.xd) / (self.std**2), axis=0)
          dvar = 0.5 * dstd / self.std
          dxd += (2.0 / self.batch_size) * self.xd * dvar
          dmu = np.sum(dxd, axis=0)
          dx = dxd - dmu / self.batch_size

          self.dgamma = dgamma
          self.dbeta = dbeta
          return dx