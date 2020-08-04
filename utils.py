import tensorflow as tf
import tensorflow.keras.backend as K
import os
import numpy as np
import cv2
import sys 
import tarfile
import matplotlib.pyplot as plt
import pickle
import argparse
import collections
from tifffile import imsave
# from operations_np import *

def minmax_norm(x):
    mini = tf.reduce_min(x, axis=[1, 2, 3], keepdims=True)
    maxi = tf.reduce_max(x, axis=[1, 2, 3], keepdims=True)
    x = tf.div(x - mini, maxi - mini + 1e-7)
    return x

def gaussian_kernel(size=25, mean=0., std=1.):
    """Makes 2D gaussian Kernel for convolution."""

    d = tf.distributions.Normal(mean, std)

    vals = d.prob(tf.range(start = -size, limit = size + 1, dtype = tf.float32))

    gauss_kernel = tf.einsum('i,j->ij',
                                  vals,
                                  vals)

    return gauss_kernel / tf.reduce_sum(gauss_kernel)

def apply_gaussian_kernel(im, gk):
    # Make Gaussian Kernel with desired specs.

    # Expand dimensions of `gauss_kernel` for `tf.nn.conv2d` signature.
    gauss_kernel = gk[:, :, tf.newaxis, tf.newaxis]

    # Convolve.

    return tf.nn.conv2d(im, gauss_kernel, strides=[1, 1, 1, 1], padding="SAME")

def load_graph(frozen_graph_filename):
    # We load the protobuf file from the disk and parse it to retrieve the
    # unserialized graph_def
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # Then, we import the graph_def into a new Graph and returns it
    with tf.Graph().as_default() as graph:
        # The name var will prefix every op/nodes in your graph
        # Since we load everything in a new graph, this is not needed
        # tf.keras.backend.set_learning_phase(0)
        tf.import_graph_def(graph_def, name='')
    return graph

def apply_nn(pixel, weights, biases):

    h1 = tf.nn.leaky_relu(weights[0]*pixel+biases[0])
    # h1 = tf.tile(h1[:, :, None], [1, 1, weights[0].get_shape().as_list()[-1]])
    # tmp = int(np.sqrt(weights[1].get_shape().as_list()[-1]))
    # weights[1] = tf.reshape(weights[1][:,:,None], [-1, tmp, tmp])
    # h2 = tf.nn.leaky_relu(tf.matmul(h1, weights[1]) + biases[1])
    h2 = tf.nn.tanh(tf.reduce_sum(h1, axis=2)+tf.squeeze(biases[1], axis=-2))

    return h2

def get_sal(model, im):
    return tf.keras.layers.UpSampling2D(2, data_format=None)(model(im))

def down_sample(x, scale_factor_h, scale_factor_w) :
    _, h, w, _ = x.get_shape().as_list()
    new_size = [h // scale_factor_h, w // scale_factor_w]

    return tf.image.resize_nearest_neighbor(x, size=new_size)

    # losses

def interpolate(a, b):
    shape = tf.concat((tf.shape(a)[0:1], tf.tile([1], [a.shape.ndims - 1])), axis=0)
    alpha = tf.random_uniform(shape=shape, minval=0., maxval=1.)
    inter = a + alpha * (b - a)
    inter.set_shape(a.get_shape().as_list())
    return inter

def gradient_penalty(real, fake, f):

    x = interpolate(real, fake)
    pred = f.get_outputs(real, x)
    pred = pred[2]
    gradients = tf.gradients(pred, x)[0]
    slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1,2,3]))
    gp = tf.reduce_mean((slopes - 1.)**2)
    return gp

def down_sample_avg(x, scale_factor=2) :
    imshape = x.get_shape().as_list()
    newshape = [int(imshape[1]/scale_factor), int(imshape[2]/scale_factor)]
    return tf.image.resize(x, newshape)

def get_sal_keras(model, im, means, gk):
    sal = model(processim(im, means))
    # print(sal.shape)
    sal = sal[:, 1, :, :, :]
    minsalinit = tf.reduce_min(sal, axis=[1, 2, 3], keepdims=True)
    maxsalinit = tf.reduce_max(sal, axis=[1, 2, 3], keepdims=True)
    sal = tf.div(sal - minsalinit, maxsalinit - minsalinit + 1e-7)
    sal = apply_gaussian_kernel(sal, gk)
    sal = downsample(sal)/4
    return sal

def get_sal_keras_sftmax(model, im, means, gk):
    sal = model(processim(im, means))
    # print(sal.shape)
    sal = sal[:, 1, :, :, :]
    # minsalinit = tf.reduce_min(sal, axis=[1, 2, 3], keepdims=True)
    # maxsalinit = tf.reduce_max(sal, axis=[1, 2, 3], keepdims=True)
    # sal = tf.div(sal - minsalinit, maxsalinit - minsalinit + 1e-7)
    sal = apply_gaussian_kernel(sal, gk)
    sal = downsample(sal)
    sizes = sal.get_shape().as_list()
    sal_reshape = tf.reshape(sal, [sizes[0], -1])
    sal = tf.nn.softmax(sal_reshape)
    sal = tf.reshape(sal, sizes)
    return sal

def compute_gradients(tensor, var_list):
  grads = tf.gradients(tensor, var_list)
  return [grad if grad is not None else tf.zeros_like(var)
          for var, grad in zip(var_list, grads)]

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def processim(im, means):
    '''' im is supposed to be in [-1,1]'''
    binim = (im+1)*0.5
    rgbim = binim*255.
    rgbim = rgbim - means
    return rgbim

def downsample(im):
    with tf.variable_scope('downsampler', reuse=tf.AUTO_REUSE):
        return tf.layers.conv2d(im, 1, 2, 2, use_bias=False, kernel_initializer=tf.ones_initializer(), trainable=False)

def construct_gram_matrix(layer):
    shape = tf.shape(layer)
    num_images = shape[0]
    width = shape[1]
    height = shape[2]
    num_filters = shape[3]
    filters = tf.reshape(layer, tf.stack([num_images, -1, num_filters]))
    grams = tf.matmul(filters, filters, transpose_a=True) / tf.to_float(width * height * num_filters)
    return grams

def print_params(params, string):
    for q, v in params.items():
        if np.size(v) == 1:
            string += '%s: %.4f, ' % (q, v.squeeze().flatten())
        else:
            string += '%s: ' %q + '%.3f ' * np.size(v) %tuple(v.squeeze().flatten().tolist())
            string +=','
    return string

def print_params_zbin(params, string, z):
    z = z.astype(int)
    z_i = np.split(z, 9, axis=-1)
    string += 'zbin:' + '%d ' * np.size(z) % tuple(z.squeeze().flatten().tolist())
    string += ','
    for q, v in params.items():
        if np.size(v) == 1:
            if 'gamma' in q:
                v = v*z_i[0]
            if 'sharp' in q:
                v = v * z_i[1]
            # if 'bias' in q:
            #     v = v * z_i[2]
            if 'WB' in q:
                v = v * z_i[2]
            if 'exposure' in q:
                v = v * z_i[3]
            if 'contrast' in q:
                v = v * z_i[4]
            if 'saturation' in q:
                v = v * z_i[5]
            # if 'BnW' in q:
            #     v = v * z_i[7]
            if 'blur' in q:
                v = v * z_i[8]
            string += '%s: %.3f, ' % (q, v.squeeze().flatten())
        else:
            v = v.squeeze().flatten()
            if 'tone' in q:
                v = v * z_i[6]
            if 'color' in q:
                v = v * z_i[7]
            string += '%s: ' %q + '%.3f ' * np.size(v) %tuple(v.squeeze().tolist())
            string +=','


    return string

def rgb2lum(image):
  image = 0.27 * image[:, :, :, 0] + 0.67 * image[:, :, :, 1] + 0.06 * image[:, :, :, 2]
  return image[:, :, :, None]

def lerp(a, b, l):
  return (1 - l) * a + l * b

def l1_l2(y_true, y_pred):
    loss2 = K.mean(K.square(y_pred - y_true), axis=-1)
    loss1 = K.mean(K.abs(y_pred - y_true), axis=-1)
    loss = loss1 + loss2
    return loss

def down_sample(x) :
    return tf.layers.average_pooling2d(x, pool_size=3, strides=2, padding='SAME')

def print_options(parser, opt, log_dir, to_train):
    message = ''
    message += '----------------- Options ---------------\n'
    for k, v in sorted(vars(opt).items()):
        comment = ''
        default = parser.get_default(k)
        if v != default:
            comment = '\t[default: %s]' % str(default)
        message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
    message += '----------------- End -------------------'
    print(message)

    # save to the disk
    if to_train:
        file_name = os.path.join(log_dir, 'opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write(message)
            opt_file.write('\n')

def savefixedImages(webpage, dict_images, savepath, epoch):
    """supposes that dictimages contain binary images as values"""

    for k in range(dict_images['im'].shape[0]):
        im_list=[]
        txt_link=[]
        links = []
        for i in dict_images.keys():
            im = dict_images[i] *255
            if 'mask' in i: 
                shape = im.shape[1:3][::-1]
        # for k in range(im.shape[0]):
            imk = im[k,:,:,:]
            if imk.shape[-1]==3:
                imname = os.path.join(savepath, i+'%d_%d.png' %(epoch, k))
                cv2.imwrite(imname, imk.squeeze()[:,:,::-1].astype(np.uint8))

                im_list.append(imname)
                txt_link.append('im')
                links.append(imname)
                
            elif 'sal' in i:
                im_tmp = cv2.resize(imk.squeeze(), (shape))
                salname = os.path.join(savepath, i + '%d_%d.png' % (epoch, k))
                cv2.imwrite(salname, im_tmp.astype(np.uint8))
                
                im_list.append(salname)
                txt_link.append('sal')
                links.append(salname)
                
            else:
                maskname = os.path.join(savepath, i + '%d_%d.png' % (epoch, k))
                cv2.imwrite(maskname, imk.squeeze().astype(np.uint8))
                
                im_list.append(maskname)
                txt_link.append('mask')
                links.append(maskname)

        webpage.add_images(im_list, txt_link, links)

def saveImages(webpage, dict_images, savepath, epoch, s, meansalinit, meansalfinal):
    """supposes that dictimages contain binary images as values"""

    for k in range(dict_images['im'].shape[0]):
        im_list = []
        txt_link = []
        links = []
        for i in dict_images.keys():
            im = dict_images[i] * 255
            if 'mask' in i:
                shape = im.shape[1:3][::-1]
            # for k in range(im.shape[0]):
            imk = im[k, :, :, :]
            if imk.shape[-1] == 3:
                if 'final' in i:
                    imname = os.path.join(savepath, i+'%d_%d_%.2f.png' % (epoch, k,s[k]))
                else:
                    imname = os.path.join(savepath, i + '%d_%d.png' % (epoch, k))
                cv2.imwrite(imname, np.clip(imk.squeeze()[:, :, ::-1], 0, 255).astype(np.uint8))

                im_list.append(imname)
                txt_link.append(i)
                links.append(imname.replace(os.path.dirname(os.path.dirname(imname)), '.'))

            elif 'salinit' in i:
                im_tmp = cv2.resize(imk.squeeze(), (shape))
                salname = os.path.join(savepath, i + '%d_%d.png' % (epoch, k))
                im_tmp = cv2.applyColorMap(im_tmp.astype(np.uint8), cv2.COLORMAP_JET)
                cv2.imwrite(salname, im_tmp.astype(np.uint8))
                im_list.append(salname)
                txt_link.append('%.3f'%meansalinit[k])
                links.append(salname.replace(os.path.dirname(os.path.dirname(salname)), '.'))
                
            elif 'salfinal' in i:
                im_tmp = cv2.resize(imk.squeeze(), (shape))
                salname = os.path.join(savepath, i+'%d_%d_%.2f.png' % (epoch, k,s[k]))
                im_tmp = cv2.applyColorMap(im_tmp.astype(np.uint8), cv2.COLORMAP_JET)
                cv2.imwrite(salname, im_tmp)
                im_list.append(salname)
                txt_link.append('%.3f'%meansalfinal[k])
                links.append(salname.replace(os.path.dirname(os.path.dirname(salname)), '.'))
                
            elif 'diffsal' in i:
                im_tmp = cv2.resize(imk.squeeze(), (shape))
                salname = os.path.join(savepath, i+'%d_%d_%.2f.png' % (epoch, k,s[k]))
                im_tmp = cv2.applyColorMap(im_tmp.astype(np.uint8), cv2.COLORMAP_JET)
                cv2.imwrite(salname, im_tmp)
                im_list.append(salname)
                txt_link.append('%.3f_%.3f' %(s[k], meansalfinal[k]-meansalinit[k]))
                links.append(salname.replace(os.path.dirname(os.path.dirname(salname)), '.'))

            else:
                maskname = os.path.join(savepath, i + '%d_%d.png' % (epoch, k))
                cv2.imwrite(maskname, imk.squeeze().astype(np.uint8))

                im_list.append(maskname)
                txt_link.append(i)
                links.append(maskname.replace(os.path.dirname(os.path.dirname(maskname)), '.'))

        webpage.add_images(im_list, txt_link, links)

def saveImages_bis(webpage, dict_images, savepath):
    """supposes that dictimages contain binary images as values"""

    # od = collections.OrderedDict(sorted(dict_images.items()))
    for k in range(dict_images[list(dict_images.keys())[0]].shape[0]):
        im_list = []
        txt_link = []
        links = []
        # for i in dict_images.keys():
        for i, im in dict_images.items():
            # im = dict_images[i]# * 255
            if 'mask' in i:
                shape = im.shape[1:3][::-1]
            # for k in range(im.shape[0]):
            imk = im[k, :, :, :]
            if imk.shape[-1] == 3:
                if 'final' in i:
                    imk = (imk+1)*0.5*255.
                    iname, iparams = i.split('separator')
                    iname = iname + '.png'
                    imname = os.path.join(savepath, iname)
                    # cv2.imwrite(imname, np.clip(imk.squeeze()[:, :, ::-1], 0, 255).astype(np.uint8))
                    # im_list.append(imname)
                    txt_link.append(iparams)
                    # links.append(imname.replace(os.path.dirname(os.path.dirname(imname)), '.'))
                else:
                    imk = imk*255.
                    iname = i + '.png'
                    imname = os.path.join(savepath, iname)
                    # iname = iname + '.png'
                    # cv2.imwrite(iname, np.clip(imk.squeeze()[:, :, ::-1], 0, 255).astype(np.uint8))
                    # im_list.append(iname)
                    txt_link.append(iname)
                    # links.append(imname.replace(os.path.dirname(os.path.dirname(imname)), '.'))

                im_list.append(imname)
                cv2.imwrite(imname, np.clip(imk.squeeze()[:, :, ::-1], 0, 255).astype(np.uint8))
                links.append(imname.replace(os.path.dirname(os.path.dirname(imname)), '.'))

            elif 'salinit' in i:
                im_tmp = imk.squeeze()#*255#cv2.resize(imk.squeeze()*255., (shape))
                salname = os.path.join(savepath, i)
                # im_tmp = cv2.applyColorMap(im_tmp.astype(np.uint8), cv2.COLORMAP_JET)
                # cv2.imwrite(salname, im_tmp.astype(np.uint8))
                plt.imsave(salname, im_tmp, cmap='jet')
                im_list.append(salname)
                txt_link.append(i)
                links.append(salname.replace(os.path.dirname(os.path.dirname(salname)), '.'))

            elif 'salfinal' in i:
                im_tmp = imk.squeeze()#*255#cv2.resize(imk.squeeze()*255., (shape))
                salname = os.path.join(savepath, i)
                # im_tmp = cv2.applyColorMap(im_tmp.astype(np.uint8), cv2.COLORMAP_JET)
                # cv2.imwrite(salname, np.clip(im_tmp.squeeze()[:, :, ::-1], 0, 255).astype(np.uint8))
                # cv2.imwrite(salname,plt.imshow(im_tmp, cmap='jet'))
                plt.imsave(salname, im_tmp, cmap='jet')
                # cv2.imwrite(salname, im_tmp)
                im_list.append(salname)
                txt_link.append(i)
                links.append(salname.replace(os.path.dirname(os.path.dirname(salname)), '.'))

            elif 'diffsal' in i:
                im_tmp = imk.squeeze()*255.#cv2.resize(imk.squeeze()*255., (shape))
                salname = os.path.join(savepath, i)
                im_tmp = cv2.applyColorMap(im_tmp.astype(np.uint8), cv2.COLORMAP_JET)
                cv2.imwrite(salname, im_tmp)
                im_list.append(salname)
                txt_link.append(i)
                links.append(salname.replace(os.path.dirname(os.path.dirname(salname)), '.'))

            else:
                imk = imk*255.
                maskname = os.path.join(savepath, i)
                cv2.imwrite(maskname, imk.squeeze().astype(np.uint8))

                im_list.append(maskname)
                txt_link.append(i)
                links.append(maskname.replace(os.path.dirname(os.path.dirname(maskname)), '.'))

        webpage.add_images(im_list, txt_link, links)

def saveImages_bis_bis(webpage, dict_images, savepath):
    """print paramerters bellow salfinal rather than bellow imfinal"""

    # od = collections.OrderedDict(sorted(dict_images.items()))
    for k in range(dict_images[list(dict_images.keys())[0]].shape[0]):
        im_list = []
        txt_link = []
        links = []
        # for i in dict_images.keys():
        for i, im in dict_images.items():
            # im = dict_images[i]# * 255
            if 'mask' in i:
                shape = im.shape[1:3][::-1]
            # for k in range(im.shape[0]):
            imk = im[k, :, :, :]
            if imk.shape[-1] == 3:
                if 'final' in i:
                    imk = (imk+1)*0.5*255.
                    iname = i + '.png'
                    imname = os.path.join(savepath, iname)
                    # cv2.imwrite(imname, np.clip(imk.squeeze()[:, :, ::-1], 0, 255).astype(np.uint8))
                    # im_list.append(imname)
                    txt_link.append(iname)
                    # links.append(imname.replace(os.path.dirname(os.path.dirname(imname)), '.'))
                else:
                    imk = imk*255.
                    iname = i + '.png'
                    imname = os.path.join(savepath, iname)
                    # iname = iname + '.png'
                    # cv2.imwrite(iname, np.clip(imk.squeeze()[:, :, ::-1], 0, 255).astype(np.uint8))
                    # im_list.append(iname)
                    txt_link.append(iname)
                    # links.append(imname.replace(os.path.dirname(os.path.dirname(imname)), '.'))

                im_list.append(imname)
                # cv2.imwrite(imname, np.clip(imk.squeeze()[:, :, ::-1], 0, 255).astype(np.uint8))
                cv2.imwrite(imname, (imk.squeeze()[:, :, ::-1]).astype(np.uint8))
                links.append(imname.replace(os.path.dirname(os.path.dirname(imname)), '.'))

            elif 'salinit' in i:
                im_tmp = imk.squeeze()#*255#cv2.resize(imk.squeeze()*255., (shape))
                salname = os.path.join(savepath, i)
                # im_tmp = cv2.applyColorMap(im_tmp.astype(np.uint8), cv2.COLORMAP_JET)
                # cv2.imwrite(salname, im_tmp.astype(np.uint8))
                plt.imsave(salname, im_tmp, vmin=0, vmax=1, cmap='jet')
                imsave(salname.replace('.png', '.tif'), im_tmp)
                im_list.append(salname)
                txt_link.append(i)
                links.append(salname.replace(os.path.dirname(os.path.dirname(salname)), '.'))

            elif 'salfinal' in i:
                im_tmp = imk.squeeze()#*255#cv2.resize(imk.squeeze()*255., (shape))
                iname, iparams = i.split('separator')
                iname = iname
                salname = os.path.join(savepath, iname)
                # cv2.imwrite(imname, np.clip(imk.squeeze()[:, :, ::-1], 0, 255).astype(np.uint8))
                # im_list.append(imname)
                txt_link.append(iparams)

                # im_tmp = cv2.applyColorMap(im_tmp.astype(np.uint8), cv2.COLORMAP_JET)
                # cv2.imwrite(salname, np.clip(im_tmp.squeeze()[:, :, ::-1], 0, 255).astype(np.uint8))
                # cv2.imwrite(salname,plt.imshow(im_tmp, cmap='jet'))
                # if 'dec' in i:
                #     plt.imsave(salname, im_tmp, cmap='hot')
                # else:
                plt.imsave(salname, im_tmp,vmin=0, vmax=1, cmap='jet')
                imsave(salname.replace('.png', '.tif'), im_tmp)
                # cv2.imwrite(salname, im_tmp)
                im_list.append(salname)
                # txt_link.append(i)
                links.append(salname.replace(os.path.dirname(os.path.dirname(salname)), '.'))

            else:
                imk = imk*255.
                maskname = os.path.join(savepath, i)
                cv2.imwrite(maskname, imk.squeeze().astype(np.uint8))

                im_list.append(maskname)
                txt_link.append(i)
                links.append(maskname.replace(os.path.dirname(os.path.dirname(maskname)), '.'))

        webpage.add_images(im_list, txt_link, links)

def saveImages_all(webpage, dict_images, savepath):
    """print paramerters bellow salfinal rather than bellow imfinal"""

    # od = collections.OrderedDict(sorted(dict_images.items()))
    for k in range(dict_images[list(dict_images.keys())[0]].shape[0]):
        im_list = []
        txt_link = []
        links = []
        # for i in dict_images.keys():
        for i, im in dict_images.items():
            # im = dict_images[i]# * 255
            if 'mask' in i:
                shape = im.shape[1:3][::-1]
            # for k in range(im.shape[0]):
            imk = im[k, :, :, :]
            if imk.shape[-1] == 3:
                if 'final' in i:
                    imk = (imk+1)*0.5*255.
                    iname = i + '.png'
                    imname = os.path.join(savepath, iname)
                    # cv2.imwrite(imname, np.clip(imk.squeeze()[:, :, ::-1], 0, 255).astype(np.uint8))
                    # im_list.append(imname)
                    txt_link.append(iname)
                    # links.append(imname.replace(os.path.dirname(os.path.dirname(imname)), '.'))
                else:
                    imk = imk*255.
                    iname = i + '.png'
                    imname = os.path.join(savepath, iname)
                    # iname = iname + '.png'
                    # cv2.imwrite(iname, np.clip(imk.squeeze()[:, :, ::-1], 0, 255).astype(np.uint8))
                    # im_list.append(iname)
                    txt_link.append(iname)
                    # links.append(imname.replace(os.path.dirname(os.path.dirname(imname)), '.'))

                im_list.append(imname)
                # cv2.imwrite(imname, np.clip(imk.squeeze()[:, :, ::-1], 0, 255).astype(np.uint8))
                cv2.imwrite(imname, (imk.squeeze()[:, :, ::-1]).astype(np.uint8))
                links.append(imname.replace(os.path.dirname(os.path.dirname(imname)), '.'))

            elif 'sal' in i:
                im_tmp = imk.squeeze()#*255#cv2.resize(imk.squeeze()*255., (shape))
                salname = os.path.join(savepath, i)
                # im_tmp = cv2.applyColorMap(im_tmp.astype(np.uint8), cv2.COLORMAP_JET)
                # cv2.imwrite(salname, im_tmp.astype(np.uint8))
                plt.imsave(salname, im_tmp, cmap='jet')
                imsave(salname.replace('.png', '.tif'), im_tmp)
                im_list.append(salname)
                txt_link.append(i)
                links.append(salname.replace(os.path.dirname(os.path.dirname(salname)), '.'))

            elif 'diffsal' in i:
                im_tmp = imk.squeeze()*255.#cv2.resize(imk.squeeze()*255., (shape))
                salname = os.path.join(savepath, i)
                im_tmp = cv2.applyColorMap(im_tmp.astype(np.uint8), cv2.COLORMAP_JET)
                cv2.imwrite(salname, im_tmp)
                im_list.append(salname)
                txt_link.append(i)
                links.append(salname.replace(os.path.dirname(os.path.dirname(salname)), '.'))

            else:
                imk = imk*255.
                maskname = os.path.join(savepath, i)
                cv2.imwrite(maskname, imk.squeeze().astype(np.uint8))

                im_list.append(maskname)
                txt_link.append(i)
                links.append(maskname.replace(os.path.dirname(os.path.dirname(maskname)), '.'))

        webpage.add_images(im_list, txt_link, links)

def saveImages_resize(webpage, dict_images, savepath, imshape):
    """supposes that dictimages contain binary images as values"""

    # od = collections.OrderedDict(sorted(dict_images.items()))
    for k in range(dict_images[list(dict_images.keys())[0]].shape[0]):
        im_list = []
        txt_link = []
        links = []
        # for i in dict_images.keys():
        for i, im in dict_images.items():
            # im = dict_images[i]# * 255
            if 'mask' in i:
                shape = im.shape[1:3][::-1]
            # for k in range(im.shape[0]):
            imk = im[k, :, :, :]
            if imk.shape[-1] == 3:
                if 'final' in i:
                    imk = (imk+1)*0.5*255.
                    iname, iparams = i.split('separator')
                    iname = iname + '.png'
                    imname = os.path.join(savepath, iname)
                    # cv2.imwrite(imname, np.clip(imk.squeeze()[:, :, ::-1], 0, 255).astype(np.uint8))
                    # im_list.append(imname)
                    txt_link.append(iparams)
                    # links.append(imname.replace(os.path.dirname(os.path.dirname(imname)), '.'))
                else:
                    imk = imk*255.
                    iname = i + '.png'
                    imname = os.path.join(savepath, iname)
                    # iname = iname + '.png'
                    # cv2.imwrite(iname, np.clip(imk.squeeze()[:, :, ::-1], 0, 255).astype(np.uint8))
                    # im_list.append(iname)
                    txt_link.append(iname)
                    # links.append(imname.replace(os.path.dirname(os.path.dirname(imname)), '.'))

                im_list.append(imname)
                imtosave = np.clip(imk.squeeze()[:, :, ::-1], 0, 255)
                imtosave = cv2.resize(imtosave, tuple(imshape))
                cv2.imwrite(imname, imtosave.astype(np.uint8))
                links.append(imname.replace(os.path.dirname(os.path.dirname(imname)), '.'))

            elif 'salinit' in i:
                im_tmp = imk.squeeze()#*255#cv2.resize(imk.squeeze()*255., (shape))
                im_tmp = cv2.resize(im_tmp, tuple(imshape))
                salname = os.path.join(savepath, i)
                # im_tmp = cv2.applyColorMap(im_tmp.astype(np.uint8), cv2.COLORMAP_JET)
                # cv2.imwrite(salname, im_tmp.astype(np.uint8))
                plt.imsave(salname, im_tmp, cmap='jet')
                im_list.append(salname)
                txt_link.append(i)
                links.append(salname.replace(os.path.dirname(os.path.dirname(salname)), '.'))

            elif 'salfinal' in i:
                im_tmp = imk.squeeze()#*255#cv2.resize(imk.squeeze()*255., (shape))
                im_tmp = cv2.resize(im_tmp, tuple(imshape))
                salname = os.path.join(savepath, i)
                # im_tmp = cv2.applyColorMap(im_tmp.astype(np.uint8), cv2.COLORMAP_JET)
                # cv2.imwrite(salname, np.clip(im_tmp.squeeze()[:, :, ::-1], 0, 255).astype(np.uint8))
                # cv2.imwrite(salname,plt.imshow(im_tmp, cmap='jet'))
                plt.imsave(salname, im_tmp, cmap='jet')
                # cv2.imwrite(salname, im_tmp)
                im_list.append(salname)
                txt_link.append(i)
                links.append(salname.replace(os.path.dirname(os.path.dirname(salname)), '.'))

            elif 'diffsal' in i:
                im_tmp = imk.squeeze()#cv2.resize(imk.squeeze()*255., (shape))
                im_tmp = cv2.resize(im_tmp, tuple(imshape))
                salname = os.path.join(savepath, i)
                im_tmp = cv2.applyColorMap(im_tmp.astype(np.uint8), cv2.COLORMAP_JET)
                cv2.imwrite(salname, im_tmp)
                im_list.append(salname)
                txt_link.append(i)
                links.append(salname.replace(os.path.dirname(os.path.dirname(salname)), '.'))

            else:
                imk = imk*255.
                imk = cv2.resize(imk, tuple(imshape))
                maskname = os.path.join(savepath, i)
                cv2.imwrite(maskname, imk.squeeze().astype(np.uint8))

                im_list.append(maskname)
                txt_link.append(i)
                links.append(maskname.replace(os.path.dirname(os.path.dirname(maskname)), '.'))

        webpage.add_images(im_list, txt_link, links)

def savefilterchart(webpage, images_list, titles, savepath):
    txt_link=[]
    im_list=[]
    links=[]
    for k in range(len(images_list)):
        imk = images_list[k] * 255.
        iname = titles[k] + '.png'
        imname = os.path.join(savepath, iname)
        txt_link.append(iname.split('.png')[0])
        im_list.append(imname)
        cv2.imwrite(imname, np.clip(imk.numpy().squeeze()[:, :, ::-1], 0, 255).astype(np.uint8))
        links.append(imname.replace(os.path.dirname(os.path.dirname(imname)), '.'))
    webpage.add_images_filterchart(im_list, txt_link, links)
    webpage.save()

def savevaryingImages(webpage, dict_images, savepath, epoch, s):
    """supposes that dictimages contain binary images as values"""

    for i in dict_images.keys():
        im = dict_images[i] *255
        if 'im' in i: 
            shape = im.shape[1:3][::-1]
        for k in range(im.shape[0]):
            imk = im[k,:,:,:]
            if imk.shape[-1]==3:
                cv2.imwrite(os.path.join(savepath, i+'%d_%d_%.2f.png' % (epoch, k,s[k])), imk.squeeze()[:,:,::-1].astype(np.uint8))
            elif 'sal' in i:
                im_tmp = cv2.resize(imk.squeeze(), (shape))
                cv2.imwrite(os.path.join(savepath, i + '%d_%d_%.2f.png' % (epoch, k,s[k])), im_tmp.astype(np.uint8))

def str_to_class(classname):
    return getattr(sys.modules[__name__], classname)

def make_tarfile(output_filename, source_dir):
    with tarfile.open(output_filename, "w:gz") as tar:
        tar.add(source_dir, arcname=os.path.basename(source_dir))

def init_counters(dict):
    counters={}
    for q,v in dict.items():
        counters[q] = 0
    return counters

def build_histogram(params, counter, sloss):
    if sloss:
        for q,v in params.items():
            if 'gamma' in q:
                if np.abs(params[q] - 1) >= 0.05:
                    counter[q]+=1
            if 'bias' in q:
                if np.abs(params[q]) >= 0.02:
                    counter[q]+=1
            if 'blur' in q:
                if np.abs(params[q]) >= 0.5:
                    counter[q]+=1
            if 'sharp' in q:
                if np.abs(params[q]) >= 0.5:
                    counter[q] += 1
            if 'WB' in q:
                if np.abs(params[q]) >= 0.2:
                    counter[q] += 1
            if 'exposure' in q:
                if np.abs(params[q]) >= 0.1:
                    counter[q] += 1
            if 'saturation' in q:
                if np.abs(params[q]) >= 0.3:
                    counter[q] += 1
            if 'contrast' in q:
                if np.abs(params[q]) >= 0.2:
                    counter[q] += 1
            if 'BnW' in q:
                if np.abs(params[q]) >= 0.2:
                    counter[q] += 1
            if 'tone' in q:
                if np.sum(np.abs(params[q]-1) >= 0.2)>0:
                    counter[q] += 1
            if 'color' in q:
                if np.sum(np.abs(params[q]-1) >= 0.05)>0:
                    counter[q] += 1
    else:
        pass

    return counter

def save_deltasal_histogram(deltas_values, title, savepath, name, num_bins=10):
    n, bins, patches = plt.hist(np.array(deltas_values), num_bins, facecolor='blue', alpha=0.5)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(os.path.join(savepath, name))
    plt.close()

def save_success_barplot(fg_param_counter, bg_param_counter, savepath, title, name = 'success_barplot.png'):
    plt.subplot(1, 2, 1)
    plt.bar(list(fg_param_counter.keys()), fg_param_counter.values(), color='g')
    plt.xticks(rotation='90')

    plt.subplot(1, 2, 2)
    plt.bar(list(bg_param_counter.keys()), bg_param_counter.values(), color='b')
    plt.xticks(rotation='90')
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(os.path.join(savepath, name))
    plt.close()

def save_success_barplot_perclass(fg_param_counters, bg_param_counters, savepath, title):

    for classname in list(fg_param_counters.keys()):
        plt.subplot(1, 2, 1)
        plt.bar(list(fg_param_counters[classname].keys()), fg_param_counters[classname].values(), color='g')
        plt.xticks(rotation='90')

        plt.subplot(1, 2, 2)
        plt.bar(list(bg_param_counters[classname].keys()), bg_param_counters[classname].values(), color='b')
        plt.xticks(rotation='90')
        plt.suptitle(title + 'for class %s' %classname)
        plt.tight_layout()
        plt.savefig(os.path.join(savepath, 'params_used_for_%s'%classname))
        plt.close()

def save_barplot(counter_dict, savepath, title, name):
    plt.bar(list(counter_dict.keys()), counter_dict.values())
    plt.xticks(rotation='90')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(os.path.join(savepath, name))
    print(os.path.join(savepath, name))
    plt.close()

def accumulate_s_and_dloss(s_val, sloss, dloss, sd_mat, eps):

    if 0 < sloss <eps:
        entry = np.expand_dims(np.stack([s_val, dloss], axis=-1), axis=0)
        if np.sum(sd_mat[0,:]) == 0:
            sd_mat = entry
        else:
            sd_mat = np.concatenate([sd_mat, entry], axis=0)

    return sd_mat

def save_success_curve(sd_mat, savepath, eps):

    sd_mat = sd_mat[sd_mat[:,0].argsort(),:]
    plt.plot(sd_mat[:,0], sd_mat[:,1], '.')
    plt.title('realism function of s conditioned on success pm %.3f' %eps)
    plt.tight_layout()
    plt.savefig(os.path.join(savepath, 'success_realismcurve.png'))
    plt.close()

def accumulate_s_and_deltas(s_val, deltas, sds_mat):

    entry = np.expand_dims(np.stack([s_val, deltas.squeeze()], axis=-1), axis=0)

    if np.sum(sds_mat) == 0:
        sds_mat = entry
    else:
        sds_mat = np.concatenate([sds_mat, entry], axis=0)

    return sds_mat

def save_curve_sorted(lx,ly,title, name, savepath):
    mat = np.stack([np.array(lx), np.array(ly)], axis=1)
    mat = mat[mat[:, 0].argsort(), :]
    plt.plot(mat[:,0], mat[:,1], '.')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(os.path.join(savepath, name))
    plt.close()

def save_curve_tuples_sorted(lx,ly,title, name, savepath):
    ly = np.stack(ly, axis=0)
    imf_is_real = ly[:, 0]
    imr_is_real = ly[:, 1]
    mat = np.stack([np.array(lx), np.array(imf_is_real)], axis=1)
    mat = mat[mat[:, 0].argsort(), :]
    plt.plot(mat[:,0], mat[:,1], 'r.', label='fake ims')

    mat = np.stack([np.array(lx), np.array(imr_is_real)], axis=1)
    mat = mat[mat[:, 0].argsort(), :]
    plt.plot(mat[:,0], mat[:,1], 'g*', label='real ims')

    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(savepath, name))
    plt.close()

def save_sdeltas_curve(sds_mat, savepath):

    sds_mat = sds_mat[sds_mat[:,0].argsort(),:]
    plt.plot(sds_mat[:,0], sds_mat[:,1], '.')
    plt.title('delta_s function of s')
    plt.tight_layout()
    plt.savefig(os.path.join(savepath, 'sdeltas_curve.png'))
    plt.close()

def save_data(data, name, savepath):
    with open(os.path.join(savepath, name+'.pickle'), 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

def save_tsne_success_fail(arr, labels, savepath):
    labels = np.array(labels)
    plt.scatter(arr[labels==0,0], arr[labels==0,1], label='failure')
    plt.scatter(arr[labels == 1, 0], arr[labels == 1, 1], label='success')
    plt.title('t-sne of shared representation success (<0.01) vs fail')
    plt.legend(loc=2)
    plt.savefig(os.path.join(savepath, 'tsne_fail_success.png'))
    plt.close()

def save_tsne_increase_decrease(arr, labels, savepath):
    labels = np.array(labels)
    plt.scatter(arr[labels == 0,0], arr[labels == 0,1], label='decrease')
    plt.scatter(arr[labels == 1, 0], arr[labels == 1, 1], label='increase')
    plt.title('t-sne of shared representation for increased vs decreased saliency')
    plt.legend(loc=2)
    plt.savefig(os.path.join(savepath, 'tsne_increase_decrease.png'))
    plt.close()

def save_tsne_salloss(arr, labels, savepath):
    labels = np.array(labels)
    labels = (labels - np.min(labels))/(np.max(labels) - np.min(labels))
    plt.scatter(arr[:,0], arr[:,1], c = labels, cmap='gray')
    plt.title('t-sne of shared representation, darker values represents smaller saliency loss')
    # plt.legend(loc=2)
    plt.savefig(os.path.join(savepath, 'tsne_sloss.png'))
    plt.close()

# def update_params(params_fg_begin, params_bg_begin, params_fg_end, params_bg_end,  k):
#
#     params_fg ={}
#     params_bg = {}
#     params_fg['gamma_fg'] = params_fg_begin['gamma_fg']*(1-k) + params_fg_end['gamma_fg']*k
#     params_bg['gamma_bg'] = params_bg_begin['gamma_bg']*(1-k) + params_bg_end['gamma_bg']*k
#
#     params_fg['sharp_fg'] = params_fg_begin['sharp_fg']*(1-k) + params_fg_end['sharp_fg']*k
#     params_bg['sharp_bg'] = params_bg_begin['sharp_bg']*(1-k) + params_bg_end['sharp_bg']*k
#
#     params_fg['bias_fg'] = params_fg_begin['bias_fg']*(1-k) + params_fg_end['bias_fg']*k
#     params_bg['bias_bg'] = params_bg_begin['bias_bg']*(1-k) + params_bg_end['bias_bg']*k
#
#     params_fg['WB_fg'] = params_fg_begin['WB_fg']*(1-k) + params_fg_end['WB_fg']*k
#     params_bg['WB_bg'] = params_bg_begin['WB_bg']*(1-k) + params_bg_end['WB_bg']*k
#
#     params_fg['exposure_fg'] = params_fg_begin['exposure_fg']*(1-k) + params_fg_end['exposure_fg']*k
#     params_bg['exposure_bg'] = params_bg_begin['exposure_bg']*(1-k) + params_bg_end['exposure_bg']*k
#
#     params_fg['contrast_fg'] = params_fg_begin['contrast_fg']*(1-k) + params_fg_end['contrast_fg']*k
#     params_bg['contrast_bg'] = params_bg_begin['contrast_bg']*(1-k) + params_bg_end['contrast_bg']*k
#
#     params_fg['saturation_fg'] = params_fg_begin['saturation_fg']*(1-k) + params_fg_end['saturation_fg']*k
#     params_bg['saturation_bg'] = params_bg_begin['saturation_bg']*(1-k) + params_bg_end['saturation_bg']*k
#
#     params_fg['BnW_fg'] = params_fg_begin['BnW_fg']*(1-k) + params_fg_end['BnW_fg']*k
#     params_bg['BnW_bg'] = params_bg_begin['BnW_bg']*(1-k) + params_bg_end['BnW_bg']*k
#
#     params_fg['toneAdjustement_fg'] = params_fg_begin['toneAdjustement_fg']*(1-k) + params_fg_end['toneAdjustement_fg']*k
#     params_bg['toneAdjustement_bg'] = params_bg_begin['toneAdjustement_bg']*(1-k) + params_bg_end['toneAdjustement_bg']*k
#
#     params_fg['colorAdjustement_fg'] = params_fg_begin['colorAdjustement_fg']*(1-k) + params_fg_end['colorAdjustement_fg']*k
#     params_bg['colorAdjustement_bg'] = params_bg_begin['colorAdjustement_bg']*(1-k) + params_bg_end['colorAdjustement_bg']*k
#
#     params_bg['blur_bg'] = params_bg_begin['blur_bg']*(1-k) + params_bg_end['blur_bg']*k
#
#     return params_fg, params_bg

def initialize_params(params_fg_end, params_bg_end):

    params_fg ={}
    params_bg = {}
    params_fg['gamma'] = params_fg_end['gamma'] * 0 + 1
    params_bg['gamma'] = params_bg_end['gamma'] * 0 + 1

    params_fg['sharp'] = params_fg_end['sharp'] * 0
    params_bg['sharp'] = params_bg_end['sharp'] * 0


    params_fg['WB'] = params_fg_end['WB'] * 0
    params_bg['WB'] = params_bg_end['WB'] * 0

    params_fg['exposure'] = params_fg_end['exposure'] * 0
    params_bg['exposure'] = params_bg_end['exposure'] * 0

    params_fg['contrast'] = params_fg_end['contrast'] * 0
    params_bg['contrast'] = params_bg_end['contrast'] * 0

    params_fg['saturation'] = params_fg_end['saturation'] * 0
    params_bg['saturation'] = params_bg_end['saturation'] * 0


    params_fg['tone'] = params_fg_end['tone'] * 0 + 1
    params_bg['tone'] = params_bg_end['tone'] * 0 + 1

    params_fg['color'] = params_fg_end['color'] * 0 + 1
    params_bg['color'] = params_bg_end['color'] * 0 + 1

    params_bg['blur'] = params_bg_end['blur'] * 0

    return params_fg, params_bg

def initialize_params_gui(params_fg_end, params_bg_end):

    params_fg ={}
    params_bg = {}

    params_fg['sharp'] = params_fg_end['sharp'] * 0
    params_bg['sharp'] = params_bg_end['sharp'] * 0


    params_fg['exposure'] = params_fg_end['exposure'] * 0
    params_bg['exposure'] = params_bg_end['exposure'] * 0

    params_fg['contrast'] = params_fg_end['contrast'] * 0
    params_bg['contrast'] = params_bg_end['contrast'] * 0


    params_fg['tone'] = params_fg_end['tone'] * 0 + 1
    params_bg['tone'] = params_bg_end['tone'] * 0 + 1

    params_fg['color'] = params_fg_end['color'] * 0 + 1
    params_bg['color'] = params_bg_end['color'] * 0 + 1

    return params_fg, params_bg

def extrapolation_params(fg_params, bg_params, percentage):

    params_fg ={}
    params_bg = {}
    params_fg['gamma_fg'] = fg_params['gamma_fg'] * (1+percentage)
    params_bg['gamma_bg'] = bg_params['gamma_bg'] * (1+percentage)

    params_fg['sharp_fg'] = fg_params['sharp_fg'] * (1+percentage)
    params_bg['sharp_bg'] = bg_params['sharp_bg'] * (1+percentage)

    params_fg['bias_fg'] = fg_params['bias_fg'] * (1+percentage)
    params_bg['bias_bg'] = bg_params['bias_bg'] * (1+percentage)

    params_fg['WB_fg'] = fg_params['WB_fg'] * (1+percentage)
    params_bg['WB_bg'] = bg_params['WB_bg'] * (1+percentage)

    params_fg['exposure_fg'] = fg_params['exposure_fg'] * (1+percentage)
    params_bg['exposure_bg'] = bg_params['exposure_bg'] * (1+percentage)

    params_fg['contrast_fg'] = fg_params['contrast_fg'] * (1+percentage)
    params_bg['contrast_bg'] = bg_params['contrast_bg'] * (1+percentage)

    params_fg['saturation_fg'] = fg_params['saturation_fg'] * (1+percentage)
    params_bg['saturation_bg'] = bg_params['saturation_bg'] * (1+percentage)

    params_fg['BnW_fg'] = fg_params['BnW_fg'] * (1+percentage)
    params_bg['BnW_bg'] = bg_params['BnW_bg'] * (1+percentage)

    params_fg['toneAdjustement_fg'] = fg_params['toneAdjustement_fg'] * (1+percentage)
    params_bg['toneAdjustement_bg'] = bg_params['toneAdjustement_bg'] * (1+percentage)

    params_fg['colorAdjustement_fg'] = fg_params['colorAdjustement_fg'] * (1+percentage)
    params_bg['colorAdjustement_bg'] = bg_params['colorAdjustement_bg'] * (1+percentage)

    params_bg['blur_bg'] = bg_params['blur_bg'] * (1+percentage)

    return params_fg, params_bg

def update_params(params_fg_begin, params_bg_begin, params_fg_end, params_bg_end,  k):

    params_fg ={}
    params_bg = {}
    params_fg['gamma'] = params_fg_begin['gamma']*(1-k) + params_fg_end['gamma']*k
    params_bg['gamma'] = params_bg_begin['gamma']*(1-k) + params_bg_end['gamma']*k

    params_fg['sharp'] = params_fg_begin['sharp']*(1-k) + params_fg_end['sharp']*k
    params_bg['sharp'] = params_bg_begin['sharp']*(1-k) + params_bg_end['sharp']*k


    params_fg['WB'] = params_fg_begin['WB']*(1-k) + params_fg_end['WB']*k
    params_bg['WB'] = params_bg_begin['WB']*(1-k) + params_bg_end['WB']*k

    params_fg['exposure'] = params_fg_begin['exposure']*(1-k) + params_fg_end['exposure']*k
    params_bg['exposure'] = params_bg_begin['exposure']*(1-k) + params_bg_end['exposure']*k

    params_fg['contrast'] = params_fg_begin['contrast']*(1-k) + params_fg_end['contrast']*k
    params_bg['contrast'] = params_bg_begin['contrast']*(1-k) + params_bg_end['contrast']*k

    params_fg['saturation'] = params_fg_begin['saturation']*(1-k) + params_fg_end['saturation']*k
    params_bg['saturation'] = params_bg_begin['saturation']*(1-k) + params_bg_end['saturation']*k


    params_fg['tone'] = params_fg_begin['tone']*(1-k) + params_fg_end['tone']*k
    params_bg['tone'] = params_bg_begin['tone']*(1-k) + params_bg_end['tone']*k

    params_fg['color'] = params_fg_begin['color']*(1-k) + params_fg_end['color']*k
    params_bg['color'] = params_bg_begin['color']*(1-k) + params_bg_end['color']*k

    params_bg['blur'] = params_bg_begin['blur']*(1-k) + params_bg_end['blur']*k

    return params_fg, params_bg

def update_params_gui(params_fg_begin, params_bg_begin, params_fg_end, params_bg_end,  k):

    params_fg ={}
    params_bg = {}

    params_fg['sharp'] = params_fg_begin['sharp']*(1-k) + params_fg_end['sharp']*k
    params_bg['sharp'] = params_bg_begin['sharp']*(1-k) + params_bg_end['sharp']*k


    params_fg['exposure'] = params_fg_begin['exposure']*(1-k) + params_fg_end['exposure']*k
    params_bg['exposure'] = params_bg_begin['exposure']*(1-k) + params_bg_end['exposure']*k

    params_fg['contrast'] = params_fg_begin['contrast']*(1-k) + params_fg_end['contrast']*k
    params_bg['contrast'] = params_bg_begin['contrast']*(1-k) + params_bg_end['contrast']*k



    params_fg['tone'] = params_fg_begin['tone']*(1-k) + params_fg_end['tone']*k
    params_bg['tone'] = params_bg_begin['tone']*(1-k) + params_bg_end['tone']*k

    params_fg['color'] = params_fg_begin['color']*(1-k) + params_fg_end['color']*k
    params_bg['color'] = params_bg_begin['color']*(1-k) + params_bg_end['color']*k

    # params_bg['blur_bg'] = params_bg_begin['blur_bg']*(1-k) + params_bg_end['blur']*k

    return params_fg, params_bg

def update_params_gui_tf(params_fg_begin, params_bg_begin, params_fg_end, params_bg_end,  k):

    params_fg ={}
    params_bg = {}

    params_fg['sharp'] = tf.constant(params_fg_begin['sharp']*(1-k) + params_fg_end['sharp']*k)
    params_bg['sharp'] = tf.constant(params_bg_begin['sharp']*(1-k) + params_bg_end['sharp']*k)


    params_fg['exposure'] = tf.constant(params_fg_begin['exposure']*(1-k) + params_fg_end['exposure']*k)
    params_bg['exposure'] = tf.constant(params_bg_begin['exposure']*(1-k) + params_bg_end['exposure']*k)

    params_fg['contrast'] = tf.constant(params_fg_begin['contrast']*(1-k) + params_fg_end['contrast']*k)
    params_bg['contrast'] = tf.constant(params_bg_begin['contrast']*(1-k) + params_bg_end['contrast']*k)



    params_fg['tone'] = tf.constant(params_fg_begin['tone']*(1-k) + params_fg_end['tone']*k)
    params_bg['tone'] = tf.constant(params_bg_begin['tone']*(1-k) + params_bg_end['tone']*k)

    params_fg['color'] = tf.constant(params_fg_begin['color']*(1-k) + params_fg_end['color']*k)
    params_bg['color'] = tf.constant(params_bg_begin['color']*(1-k) + params_bg_end['color']*k)

    # params_bg['blur_bg'] = params_bg_begin['blur_bg']*(1-k) + params_bg_end['blur']*k

    return params_fg, params_bg

def initialize_params_nobias(params_fg_end, params_bg_end):

    params_fg ={}
    params_bg = {}
    params_fg['gamma_fg'] = params_fg_end['gamma_fg'] * 0 + 1
    params_bg['gamma_bg'] = params_bg_end['gamma_bg'] * 0 + 1

    params_fg['sharp_fg'] = params_fg_end['sharp_fg'] * 0
    params_bg['sharp_bg'] = params_bg_end['sharp_bg'] * 0

    # params_fg['bias_fg'] = params_fg_end['bias_fg'] * 0
    # params_bg['bias_bg'] = params_bg_end['bias_bg'] * 0

    params_fg['WB_fg'] = params_fg_end['WB_fg'] * 0
    params_bg['WB_bg'] = params_bg_end['WB_bg'] * 0

    params_fg['exposure_fg'] = params_fg_end['exposure_fg'] * 0
    params_bg['exposure_bg'] = params_bg_end['exposure_bg'] * 0

    params_fg['contrast_fg'] = params_fg_end['contrast_fg'] * 0
    params_bg['contrast_bg'] = params_bg_end['contrast_bg'] * 0

    params_fg['saturation_fg'] = params_fg_end['saturation_fg'] * 0
    params_bg['saturation_bg'] = params_bg_end['saturation_bg'] * 0

    params_fg['BnW_fg'] = params_fg_end['BnW_fg'] * 0
    params_bg['BnW_bg'] = params_bg_end['BnW_bg'] * 0

    params_fg['toneAdjustement_fg'] = params_fg_end['toneAdjustement_fg'] * 0 + 1
    params_bg['toneAdjustement_bg'] = params_bg_end['toneAdjustement_bg'] * 0 + 1

    params_fg['colorAdjustement_fg'] = params_fg_end['colorAdjustement_fg'] * 0 + 1
    params_bg['colorAdjustement_bg'] = params_bg_end['colorAdjustement_bg'] * 0 + 1

    params_bg['blur_bg'] = params_bg_end['blur_bg'] * 0

    return params_fg, params_bg

def extrapolation_params_nobias(fg_params, bg_params, percentage):

    params_fg ={}
    params_bg = {}
    params_fg['gamma_fg'] = fg_params['gamma_fg'] * (1+percentage)
    params_bg['gamma_bg'] = bg_params['gamma_bg'] * (1+percentage)

    params_fg['sharp_fg'] = fg_params['sharp_fg'] * (1+percentage)
    params_bg['sharp_bg'] = bg_params['sharp_bg'] * (1+percentage)

    # params_fg['bias_fg'] = fg_params['bias_fg'] * (1+percentage)
    # params_bg['bias_bg'] = bg_params['bias_bg'] * (1+percentage)

    params_fg['WB_fg'] = fg_params['WB_fg'] * (1+percentage)
    params_bg['WB_bg'] = bg_params['WB_bg'] * (1+percentage)

    params_fg['exposure_fg'] = fg_params['exposure_fg'] * (1+percentage)
    params_bg['exposure_bg'] = bg_params['exposure_bg'] * (1+percentage)

    params_fg['contrast_fg'] = fg_params['contrast_fg'] * (1+percentage)
    params_bg['contrast_bg'] = bg_params['contrast_bg'] * (1+percentage)

    params_fg['saturation_fg'] = fg_params['saturation_fg'] * (1+percentage)
    params_bg['saturation_bg'] = bg_params['saturation_bg'] * (1+percentage)

    params_fg['BnW_fg'] = fg_params['BnW_fg'] * (1+percentage)
    params_bg['BnW_bg'] = bg_params['BnW_bg'] * (1+percentage)

    params_fg['toneAdjustement_fg'] = fg_params['toneAdjustement_fg'] * (1+percentage)
    params_bg['toneAdjustement_bg'] = bg_params['toneAdjustement_bg'] * (1+percentage)

    params_fg['colorAdjustement_fg'] = fg_params['colorAdjustement_fg'] * (1+percentage)
    params_bg['colorAdjustement_bg'] = bg_params['colorAdjustement_bg'] * (1+percentage)

    params_bg['blur_bg'] = bg_params['blur_bg'] * (1+percentage)

    return params_fg, params_bg

def update_params_decrease(params_fg_begin, params_bg_begin, params_fg_end, params_bg_end,  k):

    params_fg ={}
    params_bg = {}
    params_fg['gamma_fg'] = params_fg_begin['gamma_fg']*(1-k) + params_fg_end['gamma_fg']*k
    params_bg['gamma_bg'] = params_bg_begin['gamma_bg']*(1-k) + params_bg_end['gamma_bg']*k

    params_fg['sharp_fg'] = params_fg_begin['sharp_fg']*(1-k) + params_fg_end['sharp_fg']*k
    params_bg['sharp_bg'] = params_bg_begin['sharp_bg']*(1-k) + params_bg_end['sharp_bg']*k

    # params_fg['bias_fg'] = params_fg_begin['bias_fg']*(1-k) + params_fg_end['bias_fg']*k
    # params_bg['bias_bg'] = params_bg_begin['bias_bg']*(1-k) + params_bg_end['bias_bg']*k

    params_fg['WB_fg'] = params_fg_begin['WB_fg']*(1-k) + params_fg_end['WB_fg']*k
    params_bg['WB_bg'] = params_bg_begin['WB_bg']*(1-k) + params_bg_end['WB_bg']*k

    params_fg['exposure_fg'] = params_fg_begin['exposure_fg']*(1-k) + params_fg_end['exposure_fg']*k
    params_bg['exposure_bg'] = params_bg_begin['exposure_bg']*(1-k) + params_bg_end['exposure_bg']*k

    params_fg['contrast_fg'] = params_fg_begin['contrast_fg']*(1-k) + params_fg_end['contrast_fg']*k
    params_bg['contrast_bg'] = params_bg_begin['contrast_bg']*(1-k) + params_bg_end['contrast_bg']*k

    params_fg['saturation_fg'] = params_fg_begin['saturation_fg']*(1-k) + params_fg_end['saturation_fg']*k
    params_bg['saturation_bg'] = params_bg_begin['saturation_bg']*(1-k) + params_bg_end['saturation_bg']*k

    params_fg['BnW_fg'] = params_fg_begin['BnW_fg']*(1-k) + params_fg_end['BnW_fg']*k
    params_bg['BnW_bg'] = params_bg_begin['BnW_bg']*(1-k) + params_bg_end['BnW_bg']*k

    params_fg['toneAdjustement_fg'] = params_fg_begin['toneAdjustement_fg']*(1-k) + params_fg_end['toneAdjustement_fg']*k
    params_bg['toneAdjustement_bg'] = params_bg_begin['toneAdjustement_bg']*(1-k) + params_bg_end['toneAdjustement_bg']*k

    params_fg['colorAdjustement_fg'] = params_fg_begin['colorAdjustement_fg']*(1-k) + params_fg_end['colorAdjustement_fg']*k
    params_bg['colorAdjustement_bg'] = params_bg_begin['colorAdjustement_bg']*(1-k) + params_bg_end['colorAdjustement_bg']*k


    return params_fg, params_bg

def initialize_params_decrease(params_fg_end, params_bg_end):

    params_fg ={}
    params_bg = {}
    params_fg['gamma_fg'] = params_fg_end['gamma_fg'] * 0 + 1
    params_bg['gamma_bg'] = params_bg_end['gamma_bg'] * 0 + 1

    params_fg['sharp_fg'] = params_fg_end['sharp_fg'] * 0
    params_bg['sharp_bg'] = params_bg_end['sharp_bg'] * 0

    # params_fg['bias_fg'] = params_fg_end['bias_fg'] * 0
    # params_bg['bias_bg'] = params_bg_end['bias_bg'] * 0

    params_fg['WB_fg'] = params_fg_end['WB_fg'] * 0
    params_bg['WB_bg'] = params_bg_end['WB_bg'] * 0

    params_fg['exposure_fg'] = params_fg_end['exposure_fg'] * 0
    params_bg['exposure_bg'] = params_bg_end['exposure_bg'] * 0

    params_fg['contrast_fg'] = params_fg_end['contrast_fg'] * 0
    params_bg['contrast_bg'] = params_bg_end['contrast_bg'] * 0

    params_fg['saturation_fg'] = params_fg_end['saturation_fg'] * 0
    params_bg['saturation_bg'] = params_bg_end['saturation_bg'] * 0

    params_fg['BnW_fg'] = params_fg_end['BnW_fg'] * 0
    params_bg['BnW_bg'] = params_bg_end['BnW_bg'] * 0

    params_fg['toneAdjustement_fg'] = params_fg_end['toneAdjustement_fg'] * 0 + 1
    params_bg['toneAdjustement_bg'] = params_bg_end['toneAdjustement_bg'] * 0 + 1

    params_fg['colorAdjustement_fg'] = params_fg_end['colorAdjustement_fg'] * 0 + 1
    params_bg['colorAdjustement_bg'] = params_bg_end['colorAdjustement_bg'] * 0 + 1


    return params_fg, params_bg

def extrapolation_params_decrease(fg_params, bg_params, percentage):

    params_fg ={}
    params_bg = {}
    params_fg['gamma_fg'] = fg_params['gamma_fg'] * (1+percentage)
    params_bg['gamma_bg'] = bg_params['gamma_bg'] * (1+percentage)

    params_fg['sharp_fg'] = fg_params['sharp_fg'] * (1+percentage)
    params_bg['sharp_bg'] = bg_params['sharp_bg'] * (1+percentage)

    # params_fg['bias_fg'] = fg_params['bias_fg'] * (1+percentage)
    # params_bg['bias_bg'] = bg_params['bias_bg'] * (1+percentage)

    params_fg['WB_fg'] = fg_params['WB_fg'] * (1+percentage)
    params_bg['WB_bg'] = bg_params['WB_bg'] * (1+percentage)

    params_fg['exposure_fg'] = fg_params['exposure_fg'] * (1+percentage)
    params_bg['exposure_bg'] = bg_params['exposure_bg'] * (1+percentage)

    params_fg['contrast_fg'] = fg_params['contrast_fg'] * (1+percentage)
    params_bg['contrast_bg'] = bg_params['contrast_bg'] * (1+percentage)

    params_fg['saturation_fg'] = fg_params['saturation_fg'] * (1+percentage)
    params_bg['saturation_bg'] = bg_params['saturation_bg'] * (1+percentage)

    params_fg['BnW_fg'] = fg_params['BnW_fg'] * (1+percentage)
    params_bg['BnW_bg'] = bg_params['BnW_bg'] * (1+percentage)

    params_fg['toneAdjustement_fg'] = fg_params['toneAdjustement_fg'] * (1+percentage)
    params_bg['toneAdjustement_bg'] = bg_params['toneAdjustement_bg'] * (1+percentage)

    params_fg['colorAdjustement_fg'] = fg_params['colorAdjustement_fg'] * (1+percentage)
    params_bg['colorAdjustement_bg'] = bg_params['colorAdjustement_bg'] * (1+percentage)


    return params_fg, params_bg