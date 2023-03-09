import numpy as np

def im2col(input_data, filter_h, filter_w, stride=1, pad=0):
    """convert multi dimensional image into 2-dimensions image(flattening).
    
    Parameters
    ----------
    input_data : 4-dimensions input image(number of images, number of channels, height of image, width of image)
    filter_h : height of filter
    filter_w : width of filter
    stride : stride of filter
    pad : padding of input image
    
    Returns
    -------
    col : 2-dimensional array(column-wise)
    """
    N, C, H, W = input_data.shape
    OH = int((H + 2 * pad - filter_h) // stride + 1)
    OW = int((W + 2 * pad - filter_w) // stride + 1)

    img = np.pad(input_data, [(0, 0), (0, 0), (pad, pad), (pad, pad)], 'constant')
    col = np.zeros((N, C, filter_h, filter_w, OH, OW))

    for y in range(filter_h):
        y_max = y * stride + OH
        for x in range(filter_w):
            x_max = x * stride + OW
            col[:, :, y, x, :, :] = img[:, :, y*stride : y_max, x*stride : x_max]

    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N * OH * OW, C * filter_h * filter_w)
    return col

def col2im(col, input_shape, filter_h, filter_w, stride=1, pad=0):
    """convert 2-dimensions image into multi dimensional image(flattening).
    
    Parameters
    ----------
    col : 2-dimensional array(column-wise)
    input_shape : shape of original input image(e.g. (10, 1, 28, 28))
    filter_h : height of filter
    filter_w : width of filter
    stride : stride of filter
    pad : padding of input image
    
    Returns
    -------
    img : 4-dimensions input image(number of images, number of channels, height of image, width of image)
    """
    N, C, H, W = input_shape
    OH = int((H + 2 * pad - filter_h) // stride + 1)
    OW = int((W + 2 * pad - filter_w) // stride + 1)
    col = col.reshape(N, OH, OW, C, filter_h, filter_w).transpose(0, 3, 4, 5, 1, 2)

    img = np.zeros((N, C, H + 2 * pad + stride - 1, W + 2 * pad + stride - 1))
    for y in range(filter_h):
        y_max = y * stride + OH
        for x in range(filter_w):
            x_max = x * stride + OW
            img[:, :, y*stride : y_max, x*stride : x_max]
    
    return img[:, :, pad : H+pad, pad : W+pad]