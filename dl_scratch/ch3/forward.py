# practice an example of forward propagation

from dl_scratch.common import *
from data.mnist import load_mnist # get a module for MNIST dataset
from PIL import Image

# normalize: make value of pixel of input image normalized
# flatten: make array of input image to 1-dimension array
# one_hot_label: make label one-hot encoding
(x_train, t_train), (x_test, t_test) = \
    load_mnist(flatten=True, normalize=False)

def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()

img = x_train[0]
label = t_train[0]
print(label)

print(img.shape)
img = img.reshape(28, 28)
print(img.shape)

img_show(img)