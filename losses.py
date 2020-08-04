"""Contains losses """
import tensorflow as tf
from utils import construct_gram_matrix



def hinge_lsgan_loss_generator(prob_fake_is_real):

    return tf.reduce_mean(-prob_fake_is_real, name='g_loss')


def hinge_lsgan_loss_discriminator(prob_real_is_real, prob_fake_is_real):

    d_real = tf.reduce_mean(-tf.minimum(0., tf.subtract(prob_real_is_real, 1.)), name='d_real')
    d_fake = tf.reduce_mean(-tf.minimum(0., tf.add(-prob_fake_is_real, -1.)), name='d_fake')
    d_loss = tf.multiply(d_real + d_fake, 0.5, name='d_loss')

    return d_loss

def hinge_lsgan_loss_discriminator_triplet(prob_real_is_real, prob_fake_is_real_inc, prob_fake_is_real_dec):

    d_real = tf.reduce_mean(-tf.minimum(0., tf.subtract(prob_real_is_real, 1.)))
    d_fake_inc = tf.reduce_mean(-tf.minimum(0., tf.add(-prob_fake_is_real_inc, -1.)))
    d_fake_dec = tf.reduce_mean(-tf.minimum(0., tf.add(-prob_fake_is_real_dec, -1.)))
    d_loss = tf.multiply(d_real + d_fake_inc + d_fake_dec, 0.33)

    return d_loss

def hinge_lsgan_loss_discriminator_single(prob_fake_is_real_inc):


    d_fake_inc = tf.reduce_mean(-tf.minimum(0., tf.add(-prob_fake_is_real_inc, -1.)))

    return d_fake_inc

def styleloss(f1, f2, f3, f4):

    gen_f, style_f = tf.split(f1, 2, 0)
    size = tf.size(gen_f)
    style_loss = tf.nn.l2_loss(construct_gram_matrix(gen_f) - construct_gram_matrix(style_f))*2 / tf.to_float(size)

    gen_f, style_f = tf.split(f2, 2, 0)
    size = tf.size(gen_f)
    style_loss += tf.nn.l2_loss(construct_gram_matrix(gen_f) - construct_gram_matrix(style_f)) * 2 / tf.to_float(size)

    gen_f, style_f = tf.split(f3, 2, 0)
    size = tf.size(gen_f)
    style_loss += tf.nn.l2_loss(construct_gram_matrix(gen_f) - construct_gram_matrix(style_f)) * 2 / tf.to_float(size)

    gen_f, style_f = tf.split(f4, 2, 0)
    size = tf.size(gen_f)
    style_loss += tf.nn.l2_loss(construct_gram_matrix(gen_f) - construct_gram_matrix(style_f)) * 2 / tf.to_float(size)

    return style_loss


def featReconLoss(f):

    gen_f, img_f = tf.split(f, 2, 0)
    content_loss = tf.nn.l2_loss(gen_f - img_f) / tf.to_float(tf.size(gen_f))

    return content_loss


def sparsity_loss(params):
    # sloss = 0
    sparsifier = tf.contrib.layers.l1_regularizer(tf.constant(1., tf.float32))
    param_list=[]
    for k,v in params.items():
        if 'all' in k:
            pass
        elif 'gamma' in k:
            param_list.append(tf.layers.flatten(v-1))
        else:
            param_list.append(tf.layers.flatten(v))

    all_params = tf.concat(param_list, axis=-1)

    return sparsifier(all_params)/tf.to_float(tf.size(all_params))