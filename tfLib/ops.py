import tensorflow as tf
from tensorflow.contrib.layers.python.layers import batch_norm
import functools
import numpy as np
import math

def log_sum_exp(x, axis=1):
    m = tf.reduce_max(x, keep_dims=True)
    return m + tf.log(tf.reduce_sum(tf.exp(x - m), axis=axis))

def lrelu(x, alpha=0.2, name="LeakyReLU"):
    with tf.variable_scope(name):
        return tf.maximum(x , alpha*x) * tf.sqrt(2.0)

def get_weight(shape, gain=1, use_wscale=True, lrmul=1, weight_var='weight'):

    fan_in = np.prod(shape[:-1]) # [kernel, kernel, fmaps_in, fmaps_out] or [in, out]d
    he_std = gain / math.sqrt(int(fan_in)) # He init
    # Equalized learning rate and custom learning rate multiplier.
    if use_wscale:
        init_std = 1.0 / lrmul
        runtime_coef = he_std * lrmul
    else:
        init_std = he_std / lrmul
        runtime_coef = lrmul

    # Create variable.
    init = tf.initializers.random_normal(0, init_std)
    return tf.get_variable(weight_var, shape=shape, initializer=init) * runtime_coef


def conv2d(input_, output_dim, k=4, s=2, gain=1, use_wscale=True, lrmul=1,
           weight_var='w', padding='SAME', scope="conv2d", use_bias=True):

    assert padding in ['SAME', 'VALID', 'REFLECT']
    with tf.variable_scope(scope):
        w = get_weight([k, k, input_.get_shape()[-1], output_dim], gain=gain, use_wscale=use_wscale, lrmul=lrmul,
                       weight_var=weight_var)

        if padding == 'REFLECT':
            input_ = tf.pad(input_, paddings=tf.constant([[0,0], [1,1], [1,1],[0,0]]), mode='REFLECT')
            conv = tf.nn.conv2d(input_, w, strides=[1, s, s, 1], padding='VALID')
        else:
            conv = tf.nn.conv2d(input_, w, strides=[1, s, s, 1], padding=padding)

        if use_bias:
            biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
            conv = tf.reshape(tf.nn.bias_add(conv, biases), tf.shape(conv))

        return conv

def fully_connect(input_, output_dim, scope=None, use_sp=False, gain=1, use_wscale=True, lrmul=1, weight_war='w',
                  bias_start=0.0, with_w=False):

  shape = input_.get_shape().as_list()
  with tf.variable_scope(scope or "Linear"):
    w = get_weight([shape[1], output_dim], gain=gain, use_wscale=use_wscale, lrmul=lrmul, weight_var=weight_war)
    bias = tf.get_variable("bias", [output_dim], tf.float32,
      initializer=tf.constant_initializer(bias_start))
    if use_sp:
        mul = tf.matmul(input_, w)
    else:
        mul = tf.matmul(input_, w)
    if with_w:
        return mul + bias, w, bias
    else:
        return mul + bias

# def instance_norm(input, scope="instance_norm", affine=True):
#     with tf.variable_scope(scope):
#         depth = input.get_shape()[-1]
#
#         mean, variance = tf.nn.moments(input, axes=[1, 2], keep_dims=True)
#         epsilon = 1e-5
#         inv = tf.rsqrt(variance + epsilon)
#         normalized = (input - mean) * inv
#         if affine:
#             scale = tf.get_variable("scale", [depth],
#                                     initializer=tf.random_normal_initializer(1.0, 0.02, dtype=tf.float32))
#             offset = tf.get_variable("offset", [depth], initializer=tf.constant_initializer(0.0))
#             return scale * normalized + offset
#         else:
#             return normalized

def instance_norm(x, scope='instance_norm'):
    return tf.contrib.layers.instance_norm(x,
                                           epsilon=1e-05,
                                           center=True, scale=True,
                                           scope=scope)

def Adaptive_instance_norm(input, beta, gamma, epsilon=1e-5, scope="adaptive_instance_norm"):

    ch = beta.get_shape().as_list()[-1]
    with tf.variable_scope(scope):

        mean, variance = tf.nn.moments(input, axes=[1,2], keep_dims=True)
        inv = tf.rsqrt(variance + epsilon)
        normalized = (input - mean) * inv
        beta = tf.reshape(beta, shape=[-1, 1, 1, ch])
        gamma = tf.reshape(gamma, shape=[-1, 1, 1, ch])

        return gamma * normalized + beta

# def modulated_conv2d_layer(input_, style_code, k=3, output_dim=512, padding='SAME', scope='modulated_conv2d'):
#     assert k >= 1 and k % 2 == 1
#     ful = functools.partial(fully_connect)
#     with tf.variable_scope(scope):
#         print(input_)
#         input_ = tf.transpose(input_, [0, 3, 1, 2])
#         w = tf.get_variable('w', [k, k, input_.get_shape()[1], output_dim],
#                             initializer=tf.contrib.layers.variance_scaling_initializer())
#         ww = w[np.newaxis]
#         fmaps = input_.shape[1].value
#         #Modulate
#         style = ful(style_code, output_dim=fmaps)
#         style = style + 1
#         ww = ww * tf.cast(style[:, np.newaxis, np.newaxis, :, np.newaxis], w.dtype)
#         # Demodulate
#         d = tf.rsqrt(tf.reduce_sum(tf.square(ww), axis=[1,2,3]) + 1e-8)
#         ww *= d[:, np.newaxis, np.newaxis, np.newaxis, :]
#         ## Reshape/scale output.
#         input_ = tf.reshape(input_, [1, -1, input_.shape[2], input_.shape[3]])
#         w = tf.reshape(tf.transpose(ww, [1, 2, 3, 0, 4]), [ww.shape[1], ww.shape[2], ww.shape[3], -1])
#
#         input_ = tf.nn.conv2d(input_, w, strides=[1, 1, 1, 1], data_format='NCHW', padding=padding)
#         print(input_)
#         # Reshape/scale output
#         input_ = tf.reshape(input_, [-1, output_dim, input_.shape[2], input_.shape[3]])
#         input_ = tf.transpose(input_, [0, 2, 3, 1])
#
#         return input_

def modulated_conv2d_layer(input_, style_code, k=1, output_dim=512, us=False, gain=1, use_wscale=True, lrmul=1, weight_war='w',
                           padding='SAME', scope='modulated_conv2d'):
    assert k >= 1 and k % 2 == 1
    ful = functools.partial(fully_connect)
    with tf.variable_scope(scope):

        # input_ = tf.transpose(input_, [0, 3, 1, 2])
        # w = tf.get_variable('w', [k, k, input_.get_shape()[-1], output_dim],
        #                     initializer=tf.contrib.layers.variance_scaling_initializer())
        w = get_weight([k, k, input_.get_shape()[-1], output_dim], gain=gain, use_wscale=use_wscale, lrmul=lrmul, weight_var=weight_war)
        ww = w[np.newaxis]
        fmaps = input_.shape[-1].value
        #Modulate
        style = ful(style_code, output_dim=fmaps)
        style = style + 1
        ww = ww * tf.cast(style[:, np.newaxis, np.newaxis, :, np.newaxis], w.dtype)
        # Demodulate
        d = tf.rsqrt(tf.reduce_sum(tf.square(ww), axis=[1,2,3]) + 1e-8)
        ww *= d[:, np.newaxis, np.newaxis, np.newaxis, :]

        ## Reshape/scale output.
        input_ = tf.transpose(input_, [1, 2, 0, 3])
        input_ = tf.reshape(input_, [1, input_.shape[0], input_.shape[1], -1])
        w = tf.reshape(tf.transpose(ww, [1, 2, 3, 0, 4]), [ww.shape[1], ww.shape[2], ww.shape[3], -1])
        if us:
            # print("hha", w.shape)
            # w = tf.transpose(w, [0, 1, 3, 2])
            # print(w.shape)
            # print(input_.shape)
            # input_ = tf.nn.conv2d_transpose(input_, w, output_shape=[input_.shape[0], input_.shape[1]*2,
            #     input_.shape[2]*2, w.shape[-2]], strides=[1, 2, 2, 1])
            # print(input_.shape)
            input_ = upscale(input_, scale=2)
            input_ = tf.nn.conv2d(input_, w, strides=[1, 1, 1, 1], data_format='NHWC', padding=padding)
        else:
            input_ = tf.nn.conv2d(input_, w, strides=[1, 1, 1, 1], data_format='NHWC', padding=padding)
        # Reshape/scale output
        input_ = tf.reshape(input_, [input_.shape[1], input_.shape[2], -1, output_dim])

        input_ = tf.transpose(input_, [2, 0, 1, 3])

        return input_

def Resblock_Mo_Affline_layers(x_init, o_dim, style_code, noise_strength, us=True, scope='resblock'):

    _, x_init_h, x_init_w, input_ch = x_init.get_shape().as_list()
    with tf.variable_scope(scope):

        def shortcut(x):
            if us:
                x = upscale(x, scale=2)
            if input_ch != o_dim:
                x = conv2d(x, output_dim=o_dim, k=1, s=1, scope='conv', use_bias=False)
            return x

        with tf.variable_scope('res1'):

            x = lrelu(x_init)
            # if us:
            #     x = upscale(x, scale=2)
            x = modulated_conv2d_layer(x, style_code, us=us, output_dim=o_dim, scope='mc1')
            noise = tf.random_normal([tf.shape(x)[0], 1, x.shape[2], x.shape[3]], dtype=x.dtype)
            x += noise * tf.cast(noise_strength, x.dtype)

        with tf.variable_scope('res2'):

            x = lrelu(x)
            x = modulated_conv2d_layer(x, style_code, us=False, output_dim=o_dim, scope='mc2')
            noise = tf.random_normal([tf.shape(x)[0], 1, x.shape[2], x.shape[3]], dtype=x.dtype)
            x += noise * tf.cast(noise_strength, x.dtype)

        if o_dim != input_ch or us:
            x_init = shortcut(x_init)

        return (x + x_init) / tf.sqrt(2.0)

def Resblock_AdaIn_Affline_layers(x_init, o_dim, style_code, us=True, scope='resblock'):

    input_ch = x_init.get_shape().as_list()[-1]
    affline_layers = functools.partial(fully_connect, output_dim=input_ch*2)
    affline_layers2 = functools.partial(fully_connect, output_dim=o_dim*2)

    with tf.variable_scope(scope):

        def shortcut(x):
            if us:
                x = upscale(x, scale=2)
            if input_ch != o_dim:
                x = conv2d(x, output_dim=o_dim, k=1, s=1, scope='conv', padding='VALID', use_bias=False)
            return x

        with tf.variable_scope('res1'):
            bg = affline_layers(style_code, scope='fc1')
            beta, gamma = bg[:, 0:input_ch], bg[:, input_ch: input_ch*2]
            x = Adaptive_instance_norm(x_init, beta=beta, gamma=gamma, scope='AdaIn1')
            x = lrelu(x)
            if us:
                x = upscale(x, scale=2)
            x = conv2d(x, o_dim, k=3, s=1, padding='SAME')

        with tf.variable_scope('res2'):
            bg = affline_layers2(style_code, scope='fc2')
            beta, gamma = bg[:, 0:o_dim], bg[:, o_dim: o_dim*2]
            x = Adaptive_instance_norm(x, beta=beta, gamma=gamma, scope='AdaIn2')
            x = lrelu(x)
            x = conv2d(x, o_dim, k=3, s=1, padding='SAME')

        if o_dim != input_ch or us:
            x_init = shortcut(x_init)

        return (x + x_init) / tf.sqrt(2.0)

def Resblock(x_init, o_dim=256, relu_type="lrelu", padding='REFLECT', use_IN=True, ds=True, scope='resblock'):

    dim = x_init.get_shape().as_list()[-1]
    conv1 = functools.partial(conv2d, output_dim=dim, padding=padding, k=3, s=1)
    conv2 = functools.partial(conv2d, output_dim=o_dim, padding=padding, k=3, s=1)
    In = functools.partial(instance_norm)

    input_ch = x_init.get_shape().as_list()[-1]
    with tf.variable_scope(scope):

        def relu(relu_type):
            relu_dict = {
                "relu": tf.nn.relu,
                "lrelu": lrelu
            }
            return relu_dict[relu_type]

        def shortcut(x):
            if input_ch != o_dim:
                x = conv2d(x, output_dim=o_dim, k=1, s=1, scope='conv', use_bias=False)
            if ds:
                x = avgpool2d(x, k=2)
            return x

        if use_IN:
            x = conv1(relu(relu_type)(In(x_init, scope='bn1')), scope='c1')
            if ds:
                x = avgpool2d(x, k=2)
            x = conv2(relu(relu_type)(In(x, scope='bn2')), scope='c2')
        else:
            x = conv1(relu(relu_type)(x_init), scope='c1')
            if ds:
                x = avgpool2d(x, k=2)
            x = conv2(relu(relu_type)(x), scope='c2')

        if input_ch != o_dim or ds:
            x_init = shortcut(x_init)

        return (x + x_init) / tf.sqrt(2.0)  #unit variance

def de_conv(input_, output_dim, k_h=4, k_w=4, d_h=2, d_w=2, scope="deconv2d", with_w=False):

    with tf.variable_scope(scope):

        w = tf.get_variable('w', [k_h, k_w, output_dim[-1], input_.get_shape()[-1]], dtype=tf.float32,
                            initializer=tf.contrib.layers.variance_scaling_initializer())

        deconv = tf.nn.conv2d_transpose(input_, w, output_shape=output_dim,
                                        strides=[1, d_h, d_w, 1])

        biases = tf.get_variable('biases', [output_dim[-1]], tf.float32, initializer=tf.constant_initializer(0.0))
        deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())

        if with_w:
            return deconv, w, biases
        else:
            return deconv

def avgpool2d(x, k=2):
    return tf.nn.avg_pool(x, ksize=[1, k, k ,1], strides=[1, k, k, 1], padding='SAME')

def Adaptive_pool2d(x, output_size=1):
    input_size = get_conv_shape(x)[-1]
    stride = int(input_size / (output_size))
    kernel_size = input_size - (output_size - 1) * stride
    return tf.nn.avg_pool(x, ksize=[1, kernel_size, kernel_size, 1], strides=[1, kernel_size, kernel_size, 1], padding='SAME')

def upscale(x, scale, method='bilinear'):
    _, h, w, _ = get_conv_shape(x)
    return tf.image.resize(x, size=(h * scale, w * scale), method=method)

def get_conv_shape(tensor):
    shape = int_shape(tensor)
    return shape

def int_shape(tensor):
    shape = tensor.get_shape().as_list()
    return [num if num is not None else -1 for num in shape]

def resize_nearest_neighbor(x, new_size):
    x = tf.image.resize_nearest_neighbor(x, new_size)
    return x

def conv_cond_concat(x, y):
    """Concatenate conditioning vector on feature map axis."""
    x_shapes = x.get_shape()
    y_shapes = y.get_shape()
    y_reshaped = tf.reshape(y, [y_shapes[0], 1, 1, y_shapes[-1]])
    return tf.concat([x , y_reshaped*tf.ones([x_shapes[0], x_shapes[1], x_shapes[2] , y_shapes[-1]])], 3)

def batch_normal(input, scope="scope", reuse=False):
    return batch_norm(input, epsilon=1e-5, decay=0.9, scale=True, scope=scope, reuse=reuse, fused=True, updates_collections=None)

def _l2normalize(v, eps=1e-12):
  return v / (tf.reduce_sum(v ** 2) ** 0.5 + eps)

def getWeight_Decay(scope='discriminator'):
    return tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES, scope=scope))

def getTrainVariable(vars, scope='discriminator'):
    return [var for var in vars if scope in var.name]


