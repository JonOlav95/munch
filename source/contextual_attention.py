import tensorflow as tf
import numpy as np

from config import FLAGS


def resize(x, scale=2, to_shape=None, align_corners=True, dynamic=False,
           func=tf.compat.v1.image.resize_bilinear, name='resize'):
    if dynamic:
        xs = tf.compat.v1.cast(tf.compat.v1.shape(x), tf.compat.v1.float32)
        new_xs = [tf.compat.v1.cast(xs[1] * scale, tf.compat.v1.int32),
                  tf.compat.v1.cast(xs[2] * scale, tf.compat.v1.int32)]
    else:
        xs = x.get_shape().as_list()
        new_xs = [int(xs[1] * scale), int(xs[2] * scale)]
    with tf.compat.v1.variable_scope(name):
        if to_shape is None:
            x = tf.image.resize(x, [new_xs[0], new_xs[1]], method='nearest')
            #x = func(x, new_xs, align_corners=align_corners)
        else:
            x = tf.image.resize(x, [to_shape[0], to_shape[1]], method='nearest')
            #x = func(x, [to_shape[0], to_shape[1]], align_corners=align_corners)
    return x


def contextual_attention(f, b, mask=None, ksize=3, stride=1, rate=1,
                         fuse_k=3, softmax_scale=10., training=True, fuse=True):
    """ Contextual attention layer implementation.
    Contextual attention is first introduced in publication:
        Generative Image Inpainting with Contextual Attention, Yu et al.
    Args:
        x: Input feature to match (foreground).
        t: Input feature for match (background).
        mask: Input mask for t, indicating patches not available.
        ksize: Kernel size for contextual attention.
        stride: Stride for extracting patches from t.
        rate: Dilation for matching.
        softmax_scale: Scaled softmax for attention.
        training: Indicating if current graph is training or inference.
    Returns:
        tf.compat.v1.Tensor: output
    """

    # get shapes
    raw_fs = tf.compat.v1.shape(f)

    raw_int_fs = f.get_shape().as_list()
    raw_int_bs = b.get_shape().as_list()

    # extract patches from background with stride and rate
    kernel = 2 * rate
    raw_w = tf.compat.v1.extract_image_patches(
        b, [1, kernel, kernel, 1], [1, rate * stride, rate * stride, 1], [1, 1, 1, 1], padding='SAME')

    # Original code does not get the batch size as None
    raw_w = tf.compat.v1.reshape(raw_w, [FLAGS["replica_batch_size"], -1, kernel, kernel, raw_int_bs[3]])
    raw_w = tf.compat.v1.transpose(raw_w, [0, 2, 3, 4, 1])  # transpose to b*k*k*c*hw
    # downscaling foreground option: downscaling both foreground and
    # background for matching and use original background for reconstruction.
    f = resize(f, scale=1. / rate, func=tf.compat.v1.image.resize_nearest_neighbor)
    b = resize(b, to_shape=[int(raw_int_bs[1] / rate), int(raw_int_bs[2] / rate)],
               func=tf.compat.v1.image.resize_nearest_neighbor)  # https://github.com/tensorflow/tensorflow/issues/11651
    if mask is not None:
        mask = resize(mask, scale=1. / rate, func=tf.compat.v1.image.resize_nearest_neighbor)
    fs = tf.compat.v1.shape(f)
    int_fs = f.get_shape().as_list()
    f_groups = tf.compat.v1.split(f, int_fs[0], axis=0)
    # from t(H*W*C) to w(b*k*k*c*h*w)
    bs = tf.compat.v1.shape(b)
    int_bs = b.get_shape().as_list()
    w = tf.compat.v1.extract_image_patches(
        b, [1, ksize, ksize, 1], [1, stride, stride, 1], [1, 1, 1, 1], padding='SAME')
    w = tf.compat.v1.reshape(w, [int_fs[0], -1, ksize, ksize, int_fs[3]])
    w = tf.compat.v1.transpose(w, [0, 2, 3, 4, 1])  # transpose to b*k*k*c*hw
    # process mask
    if mask is None:
        mask = tf.compat.v1.zeros([1, bs[1], bs[2], 1])
    m = tf.compat.v1.extract_image_patches(
        mask, [1, ksize, ksize, 1], [1, stride, stride, 1], [1, 1, 1, 1], padding='SAME')
    m = tf.compat.v1.reshape(m, [1, -1, ksize, ksize, 1])
    m = tf.compat.v1.transpose(m, [0, 2, 3, 4, 1])  # transpose to b*k*k*c*hw
    m = m[0]
    mm = tf.compat.v1.cast(tf.compat.v1.equal(tf.compat.v1.reduce_mean(m, axis=[0, 1, 2], keep_dims=True), 0.),
                           tf.compat.v1.float32)
    w_groups = tf.compat.v1.split(w, int_bs[0], axis=0)
    raw_w_groups = tf.compat.v1.split(raw_w, int_bs[0], axis=0)
    y = []
    offsets = []
    k = fuse_k
    scale = softmax_scale
    fuse_weight = tf.compat.v1.reshape(tf.compat.v1.eye(k), [k, k, 1, 1])
    for xi, wi, raw_wi in zip(f_groups, w_groups, raw_w_groups):
        # conv for compare
        wi = wi[0]
        wi_normed = wi / tf.compat.v1.maximum(
            tf.compat.v1.sqrt(tf.compat.v1.reduce_sum(tf.compat.v1.square(wi), axis=[0, 1, 2])), 1e-4)
        yi = tf.compat.v1.nn.conv2d(xi, wi_normed, strides=[1, 1, 1, 1], padding="SAME")

        # conv implementation for fuse scores to encourage large patches
        if fuse:
            yi = tf.compat.v1.reshape(yi, [1, fs[1] * fs[2], bs[1] * bs[2], 1])
            yi = tf.compat.v1.nn.conv2d(yi, fuse_weight, strides=[1, 1, 1, 1], padding='SAME')
            yi = tf.compat.v1.reshape(yi, [1, fs[1], fs[2], bs[1], bs[2]])
            yi = tf.compat.v1.transpose(yi, [0, 2, 1, 4, 3])
            yi = tf.compat.v1.reshape(yi, [1, fs[1] * fs[2], bs[1] * bs[2], 1])
            yi = tf.compat.v1.nn.conv2d(yi, fuse_weight, strides=[1, 1, 1, 1], padding='SAME')
            yi = tf.compat.v1.reshape(yi, [1, fs[2], fs[1], bs[2], bs[1]])
            yi = tf.compat.v1.transpose(yi, [0, 2, 1, 4, 3])
        yi = tf.compat.v1.reshape(yi, [1, fs[1], fs[2], bs[1] * bs[2]])

        # softmax to match
        yi *= mm  # mask
        yi = tf.compat.v1.nn.softmax(yi * scale, 3)
        yi *= mm  # mask

        offset = tf.compat.v1.argmax(yi, axis=3, output_type=tf.compat.v1.int32)
        offset = tf.compat.v1.stack([offset // fs[2], offset % fs[2]], axis=-1)
        # deconv for patch pasting
        # 3.1 paste center
        wi_center = raw_wi[0]
        yi = tf.compat.v1.nn.conv2d_transpose(yi, wi_center, tf.compat.v1.concat([[1], raw_fs[1:]], axis=0),
                                              strides=[1, rate, rate, 1]) / 4.
        y.append(yi)
        offsets.append(offset)
    y = tf.compat.v1.concat(y, axis=0)
    y.set_shape(raw_int_fs)
    offsets = tf.compat.v1.concat(offsets, axis=0)
    offsets.set_shape(int_bs[:3] + [2])
    # case1: visualize optical flow: minus current position
    h_add = tf.compat.v1.tile(tf.compat.v1.reshape(tf.compat.v1.range(bs[1]), [1, bs[1], 1, 1]), [bs[0], 1, bs[2], 1])
    w_add = tf.compat.v1.tile(tf.compat.v1.reshape(tf.compat.v1.range(bs[2]), [1, 1, bs[2], 1]), [bs[0], bs[1], 1, 1])
    offsets = offsets - tf.compat.v1.concat([h_add, w_add], axis=3)
    # to flow image
    flow = flow_to_image_tf(offsets)
    # # case2: visualize which pixels are attended
    # flow = highlight_flow_tf.compat.v1(offsets * tf.compat.v1.cast(mask, tf.compat.v1.int32))
    if rate != 1:
        flow = resize(flow, scale=rate, func=tf.compat.v1.image.resize_bilinear)
    return y, flow


def flow_to_image_tf(flow, name='flow_to_image'):
    """Tensorflow ops for computing flow to image.
    """
    with tf.compat.v1.variable_scope(name), tf.compat.v1.device('/cpu:0'):
        #img = tf.compat.v1.py_func(flow_to_image, [flow], tf.compat.v1.float32, stateful=False)
        img = tf.py_function(flow_to_image, [flow], tf.compat.v1.float32)
        img.set_shape(flow.get_shape().as_list()[0:-1] + [3])
        img = img / 127.5 - 1.
        return img


def flow_to_image(flow):
    """Transfer flow map to image.
    Part of code forked from flownet.
    """
    out = []
    maxu = -999.
    maxv = -999.
    minu = 999.
    minv = 999.
    maxrad = -1
    for i in range(flow.shape[0]):
        u = flow[i, :, :, 0]
        v = flow[i, :, :, 1]
        idxunknow = (abs(u) > 1e7) | (abs(v) > 1e7)
        u[idxunknow] = 0
        v[idxunknow] = 0
        maxu = max(maxu, np.max(u))
        minu = min(minu, np.min(u))
        maxv = max(maxv, np.max(v))
        minv = min(minv, np.min(v))
        rad = np.sqrt(u ** 2 + v ** 2)
        maxrad = max(maxrad, np.max(rad))
        u = u / (maxrad + np.finfo(float).eps)
        v = v / (maxrad + np.finfo(float).eps)
        img = compute_color(u, v)
        out.append(img)
    return np.float32(np.uint8(out))


def compute_color(u, v):
    h, w = u.shape
    img = np.zeros([h, w, 3])
    nanIdx = np.isnan(u) | np.isnan(v)
    u[nanIdx] = 0
    v[nanIdx] = 0
    # colorwheel = COLORWHEEL
    colorwheel = make_color_wheel()
    ncols = np.size(colorwheel, 0)
    rad = np.sqrt(u ** 2 + v ** 2)
    a = np.arctan2(-v, -u) / np.pi
    fk = (a + 1) / 2 * (ncols - 1) + 1
    k0 = np.floor(fk).astype(int)
    k1 = k0 + 1
    k1[k1 == ncols + 1] = 1
    f = fk - k0
    for i in range(np.size(colorwheel, 1)):
        tmp = colorwheel[:, i]
        col0 = tmp[k0 - 1] / 255
        col1 = tmp[k1 - 1] / 255
        col = (1 - f) * col0 + f * col1
        idx = rad <= 1
        col[idx] = 1 - rad[idx] * (1 - col[idx])
        notidx = np.logical_not(idx)
        col[notidx] *= 0.75
        img[:, :, i] = np.uint8(np.floor(255 * col * (1 - nanIdx)))
    return img


def make_color_wheel():
    RY, YG, GC, CB, BM, MR = (15, 6, 4, 11, 13, 6)
    ncols = RY + YG + GC + CB + BM + MR
    colorwheel = np.zeros([ncols, 3])
    col = 0
    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.transpose(np.floor(255 * np.arange(0, RY) / RY))
    col += RY
    # YG
    colorwheel[col:col + YG, 0] = 255 - np.transpose(np.floor(255 * np.arange(0, YG) / YG))
    colorwheel[col:col + YG, 1] = 255
    col += YG
    # GC
    colorwheel[col:col + GC, 1] = 255
    colorwheel[col:col + GC, 2] = np.transpose(np.floor(255 * np.arange(0, GC) / GC))
    col += GC
    # CB
    colorwheel[col:col + CB, 1] = 255 - np.transpose(np.floor(255 * np.arange(0, CB) / CB))
    colorwheel[col:col + CB, 2] = 255
    col += CB
    # BM
    colorwheel[col:col + BM, 2] = 255
    colorwheel[col:col + BM, 0] = np.transpose(np.floor(255 * np.arange(0, BM) / BM))
    col += + BM
    # MR
    colorwheel[col:col + MR, 2] = 255 - np.transpose(np.floor(255 * np.arange(0, MR) / MR))
    colorwheel[col:col + MR, 0] = 255
    return colorwheel
