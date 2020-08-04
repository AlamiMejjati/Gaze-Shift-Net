import tensorflow as tf

weight_regularizer = tf.contrib.layers.l2_regularizer(scale=0.0001)
weight_init = tf.contrib.layers.variance_scaling_initializer() # kaming init for encoder / decoder


def instance_norm(x, name='IN'):

    with tf.variable_scope(name):
        epsilon = 1e-9
        mean, var = tf.nn.moments(x, [1, 2], keep_dims=True)
        scale = tf.get_variable('scale', [x.get_shape()[-1]],
                                initializer=tf.truncated_normal_initializer(
                                    mean=1.0, stddev=0.02
        ))
        offset = tf.get_variable(
            'offset', [x.get_shape()[-1]],
            initializer=tf.constant_initializer(0.0)
        )
        out = scale * tf.div(x - mean, tf.sqrt(var + epsilon)) + offset

        return out

def adaptive_instance_norm(x, scale, offset, name='ADAIN'):

    with tf.variable_scope(name):
        epsilon = 1e-9
        mean, var = tf.nn.moments(x, [1, 2], keep_dims=True)
        out = scale * tf.div(x - mean, tf.sqrt(var + epsilon)) + offset
        return out
    

# def general_conv2d(inputconv, activation=tf.nn.leaky_relu, nf=64, ks=3, s=1, stddev=0.02,
#                    padding="VALID", name="conv2d", do_norm=True):
#     with tf.variable_scope(name):
#
#         conv = tf.layers.conv2d(
#             inputconv, nf, ks, s, padding,
#             activation=activation,
#             kernel_initializer=tf.truncated_normal_initializer(
#                 stddev=stddev
#             ),
#             bias_initializer=tf.constant_initializer(0.0)
#         )
#         if do_norm:
#             conv = instance_norm(conv)
#
#         return conv

def general_conv2d(inputconv, activation=tf.nn.leaky_relu, nf=64, ks=3, s=1, stddev=0.05,
                   padding="VALID", name="conv2d", do_norm=True):
    with tf.variable_scope(name):

        conv = tf.layers.conv2d(
            inputconv, nf, ks, s, padding,
            activation=activation,
            kernel_initializer=tf.truncated_normal_initializer(
                stddev=stddev
            ),
            bias_initializer=tf.constant_initializer(0.0)
        )
        if do_norm:
            conv = instance_norm(conv)

        return conv

def general_deconv2d(inputconv, activation=tf.nn.leaky_relu, nf=64, ks=3, s=1, stddev=0.05,
                   padding="VALID", name="conv2d", do_norm=True):
    with tf.variable_scope(name):

        conv = tf.layers.conv2d_transpose(
            inputconv, nf, ks, s, padding,
            activation=activation,
            kernel_initializer=tf.truncated_normal_initializer(
                stddev=stddev
            ),
            bias_initializer=tf.constant_initializer(0.0)
        )
        if do_norm:
            conv = instance_norm(conv)

        return conv

def conv(x, channels, kernel=4, stride=2, pad=0, pad_type='zero', use_bias=True, scope='conv'):
    with tf.variable_scope(scope):
        # if scope.__contains__("discriminator") :
        #     weight_init = tf.random_normal_initializer(mean=0.0, stddev=0.02)
        # else :
        weight_init = tf.contrib.layers.variance_scaling_initializer()

        if pad_type == 'zero' :
            x = tf.pad(x, [[0, 0], [pad, pad], [pad, pad], [0, 0]])
        if pad_type == 'reflect' :
            x = tf.pad(x, [[0, 0], [pad, pad], [pad, pad], [0, 0]], mode='REFLECT')

        x = tf.layers.conv2d(inputs=x, filters=channels,
                             kernel_size=kernel, kernel_initializer=weight_init,
                             kernel_regularizer=weight_regularizer,
                             strides=stride, use_bias=use_bias)

        return x



def single_conv(tupl):
    x, kernel = tupl
    return tf.nn.depthwise_conv2d(x, kernel, strides=(1, 1, 1, 1), padding='SAME')

def batch_conv2d(inp, K):
    # Assume kernels shape is [tf.shape(inp)[0], fh, fw, c_in, c_out]
    batch_wise_conv = tf.squeeze(tf.map_fn(
        single_conv, (tf.expand_dims(inp, 1), K), dtype=tf.float32),
        axis=1
    )
    return batch_wise_conv