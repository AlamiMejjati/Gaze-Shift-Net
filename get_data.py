import os
import glob
import tensorflow as tf
import sys
from dataloader import *

def get_data_coco(dtaLoader, maskPath, datadir, isTrain, batch_size, random_s, shape1, shape2, drop_remainder = False, shuffle=0):
    loader = getattr(sys.modules[__name__], dtaLoader)
    path_type = 'train' if isTrain else 'val'
    # path_type = 'train'
    path = os.path.join(datadir, '%s2017' % path_type)
    files = sorted(glob.glob(os.path.join(maskPath, '*.jpg')))
    if isTrain==1:
        shuffle = 1

    df = loader(files, path, channel=3, shuffle=shuffle, random_s=random_s, shape1=shape1, shape2=shape2)
    if ('HR' in dtaLoader):
        cocods = tf.data.Dataset.from_generator(lambda: df.__iter__(),
                                                 output_types=(tf.float32, tf.float32, tf.float32, tf.float32, tf.float32),
                                                 output_shapes=((shape1, shape2, 3), (shape1, shape2, 1),
                                                                (None, None, 3), (None, None, 1), (1)))
    else:
        cocods = tf.data.Dataset.from_generator(lambda: df.__iter__(),
                                                 output_types=(tf.float32, tf.float32, tf.float32),
                                                 output_shapes=((shape1, shape2, 3), (shape1, shape2, 1), (1)))

    if drop_remainder==0:
        cocods = cocods.batch(batch_size)
    else:
        cocods = cocods.batch(batch_size, drop_remainder=True)
        # ds = dataset.apply(tf.contrib.data.batch_and_drop_remainder(batch_size))
    cocoiter = cocods.make_initializable_iterator()
    return cocoiter, len(df)

def get_data_mix(dtaLoader, maskPath, datadir, isTrain, batch_size, random_s, shape1, shape2, drop_remainder = False):
    loader = getattr(sys.modules[__name__], dtaLoader)
    path_type = 'train' if isTrain else 'val'
    path = os.path.join(datadir, '%s2017' % path_type)
    files = sorted(glob.glob(os.path.join(maskPath, '*.jpg')))
    cocoimfiles = [os.path.join(path, os.path.basename(file).split('_')[0]) + '.jpg' for file in files]
    adobe5k_path = os.path.join(os.path.dirname(os.path.dirname(datadir)), 'adobe5k/jpeg')
    adobe5kfiles = sorted(glob.glob(os.path.join(adobe5k_path, '*.jpg')))
    df = loader(cocoimfiles, adobe5kfiles, channel=3, shuffle=isTrain, random_s=random_s, shape1=shape1, shape2=shape2)

    mixds = tf.data.Dataset.from_generator(lambda: df.__iter__(),
                                             output_types=tf.float32,
                                             output_shapes=(shape1, shape2, 3))
    if drop_remainder==0:
        mixds = mixds.batch(batch_size)
    else:
        mixds = mixds.batch(batch_size, drop_remainder=True)
    mixiter = mixds.make_initializable_iterator()
    return mixiter, len(files+adobe5kfiles)

