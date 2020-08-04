from losses import *
from utils import *
from tqdm import tqdm
import html
import numpy as np
from operations_np import *
from operations import *
import os
import collections
import random
import time

class adv_incdec_sftmx:
    """The trainer class"""

    def __init__(self, g_model, d_model, salmodel_path, Giterator, print_freq, log_dir, to_restore,
                 base_lr, max_step, checkpoint_dir, max_images_G, batch_size, args):

        # self.sess = tf.InteractiveSession(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
        # K.set_session(self.sess)

        self.print_freq = print_freq
        self.LAMBDAsal = args.LAMBDAsal
        self.LAMBDAFM = args.LAMBDAFM
        self.LAMBDA_r = args.LAMBDA_r
        self.LAMBDAD = args.LAMBDAD
        self.LAMBDA_p = args.LAMBDA_p
        self.output_dir = log_dir
        self.images_dir = os.path.join(self.output_dir, 'imgs')
        self.num_imgs_to_save = 20
        self.to_restore = to_restore
        self.base_lr = base_lr
        self.base_lrd = args.lrd
        self.max_step = max_step
        self.nb_its = args.nb_its
        self.checkpoint_dir = checkpoint_dir
        self.Giterator = Giterator
        self.max_images_G = max_images_G
        
        self.start_decay_p = args.startdecay
        self.batch_size = batch_size
        self.nb_iterations = max_images_G / self.batch_size
        self.salmodelpath = salmodel_path
        self.nb_gpus = args.nb_gpu
        self.gmodel = g_model
        self.dmodel = d_model
        self.lr = tf.placeholder(tf.float32, name='lr')
        self.lrd = tf.placeholder(tf.float32, name='lrd')
        self.isTrain = tf.placeholder(tf.bool, shape=(), name='istrain')
        self.args = args
        tf.keras.backend.set_learning_phase(0)  # this line is what allows us to freeze the graph without errors
        with tf.variable_scope('snet', reuse=False):
            self.salmodel = tf.keras.models.load_model(salmodel_path)
        print("Loaded sal model from disk")

    def model_setup(self, gpu):
        """
        This function sets up the model to train.
        """

        self.g_model = self.gmodel(self.args)
        # self.d_model = self.dmodel(NF=self.args.ndf, n_scale=self.args.n_scale, training=self.isTrain, n_dis=self.args.n_dis,
        #                            do_norm=self.args.donorm)
        self.d_model = self.dmodel(self.args)
        self.im, self.mask, self.s = self.Giterator.get_next()
        gk = gaussian_kernel()
        self.mask = apply_gaussian_kernel(self.mask, gk)
        means = np.array([103.939, 116.779, 123.68])
        means = means[None, None, None, :]
        sizes = self.im.get_shape().as_list()
        means = np.tile(means, [1, sizes[1], sizes[2], 1])
        means = means[:, :, :, ::-1]
        self.bin_mask = (self.mask + 1) * 0.5
        self.global_step = tf.train.get_or_create_global_step()
        self.num_fake_inputs = 0
        with tf.variable_scope('snet', reuse=True):
            self.sal_init = get_sal_keras_sftmax(self.salmodel, self.im, means, gk)

        inputs = {
            'im': self.im,
            'masks': self.mask,
            's': self.s,
            'salinit': self.sal_init}

        with tf.variable_scope('gen', reuse=gpu > 0):
            self.genim_list, self.fg_params_inc, self.bg_params_inc, self.fg_params_dec, \
            self.bg_params_dec = self.g_model.get_outputs(inputs)

        with tf.variable_scope('disc', reuse=gpu > 0):
            self.prob_im_is_real, self.features_im_real, self.prob_im_final_is_real, self.features_im_final = \
                self.d_model.get_outputs(self.im, self.genim_list)

        with tf.variable_scope('snet', reuse=True):
            sal_final_tmp = get_sal_keras_sftmax(self.salmodel, tf.concat(self.genim_list, axis=0), means, gk)

        self.sal_final_inc, self.sal_final_dec = tf.split(sal_final_tmp, 2, axis=0)

        # slim = tf.contrib.slim
        # vgg_vars = slim.get_variables_to_restore(include=['vgg_16'], exclude=self.exclude)
        # self.init_fn = slim.assign_from_checkpoint_fn('./vgg_16.ckpt', vgg_vars)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr, beta1=0.5)
        self.optimizerD = tf.train.AdamOptimizer(learning_rate=self.lrd, beta1=0.5)

    def model_setup_test_HR(self, gpu):
        """
        This function sets up the model to train.
        """

        self.g_model = self.gmodel(self.args)
        self.d_model = self.dmodel(self.args)
        # self.vgg16 = vgg.vgg_16
        self.im, self.mask, self.im_hr, self.mask_hr, self.s = self.Giterator.get_next()
        gk = gaussian_kernel()
        self.mask = apply_gaussian_kernel(self.mask, gk)
        self.mask_hr = apply_gaussian_kernel(self.mask_hr, gk)
        means = np.array([103.939, 116.779, 123.68])
        means = means[None, None, None, :]
        sizes = self.im.get_shape().as_list()
        means = np.tile(means, [1, sizes[1], sizes[2], 1])
        means = means[:, :, :, ::-1]
        self.bin_mask = (self.mask + 1) * 0.5
        self.global_step = tf.train.get_or_create_global_step()
        self.num_fake_inputs = 0
        with tf.variable_scope('snet', reuse=True):
            sal_init = get_sal_keras_sftmax(self.salmodel, self.im, means, gk)
            self.sal_init = get_sal_keras(self.salmodel, self.im, means, gk)
        self.nb_zs = 10
        self.salfinals = []
        self.imfinals = []
        self.fgparams = []
        self.bgparams = []
        self.probs_ims_are_real = []
        self.probs_imfinals_are_real = []
        self.features_ims_real = []
        self.features_ims_final = []
        self.f3s = []
        self.losses = []
        inputs = {
            'im': self.im,
            'masks': self.mask,
            's': self.s,
            'salinit': sal_init}

        with tf.variable_scope('gen', reuse=False):
            im_final, fg_params_inc, bg_params_inc, fg_params_dec, bg_params_dec = self.g_model.get_outputs(inputs)

        im_final_hr_inc = self.g_model.get_outputs_hr((self.im_hr + 1) * 0.5, self.mask_hr, fg_params_inc, bg_params_inc)
        im_final_hr_inc = tf.clip_by_value(im_final_hr_inc, -1., 1.)
        im_final_hr_dec = self.g_model.get_outputs_hr((self.im_hr + 1) * 0.5, self.mask_hr, fg_params_dec, bg_params_dec)
        im_final_hr_dec = tf.clip_by_value(im_final_hr_dec, -1., 1.)
        # im_final_hr = apply_transformations_sharp_exp_cont_tone_color_bis((self.im_hr + 1) * 0.5, self.mask_hr, fg_params,
        #                                                               bg_params,
        #                                                               self.g_model.tf1, self.g_model.tf2,
        #                                                               self.g_model.ksize)

        self.imfinals += [im_final_hr_inc, im_final_hr_dec]
        self.fgparams += [fg_params_inc, fg_params_dec]
        self.bgparams += [bg_params_inc, bg_params_dec]

        with tf.variable_scope('snet', reuse=True):
            self.salfinals.append(get_sal_keras(self.salmodel, im_final[0], means, gk))
        with tf.variable_scope('snet', reuse=True):
            self.salfinals.append(get_sal_keras(self.salmodel, im_final[1], means, gk))

    def viz(self, name):
        im = tf.concat([self.im, tf.image.grayscale_to_rgb(self.mask)] + self.genim_list, axis=2)
        im = (im + 1.0) * 0.5  # 127.5
        # im = tf.image.hsv_to_rgb(im) * 255
        im = im * 255
        # im = tf.concat([im, tf.clip_by_value(im, 0, 255)], axis=1)
        im = tf.cast(im, tf.uint8)
        return tf.summary.image(name, im, max_outputs=50)

    def compute_losses(self, gpu):

        with tf.name_scope('losses'):
            with tf.name_scope('g_loss'):
                # l2_loss = tf.reduce_mean(tf.square(self.fg_params)) + tf.reduce_mean(tf.square(self.bg_params),
                # name='l2_loss')

                numelmask = tf.reduce_sum(self.bin_mask, axis=[1, 2, 3])
                numelall = tf.ones_like(numelmask) * tf.size(self.bin_mask[0], out_type=tf.float32)
                numelmask = tf.where(tf.equal(numelmask, 0), numelall, numelmask)
                weight_recon_loss = numelall / numelmask

                saldiff_inc = -tf.reduce_mean((tf.reduce_mean(self.bin_mask * self.sal_final_inc, axis=[1, 2, 3]) -
                                          tf.reduce_mean(self.bin_mask * self.sal_init,
                                                         axis=[1, 2, 3])) * weight_recon_loss)
                saldiff_dec = tf.reduce_mean((tf.reduce_mean(self.bin_mask * self.sal_final_dec, axis=[1, 2, 3]) -
                                          tf.reduce_mean(self.bin_mask * self.sal_init,
                                                         axis=[1, 2, 3])) * weight_recon_loss)

                sal_loss = saldiff_dec + saldiff_inc

                g_gan_loss = []
                for k in range(len(self.prob_im_final_is_real)):
                    actualk = self.prob_im_final_is_real[k]
                    g_gan_loss += [hinge_lsgan_loss_generator(actualk[j]) for j in range(len(actualk))]

                g_gan_loss = tf.add_n(g_gan_loss) / float(self.args.n_scale)
                g_loss = g_gan_loss + self.LAMBDAsal * sal_loss

            d_gan_loss = []
            with tf.name_scope('d_loss'):
                for k in range(len(self.prob_im_final_is_real)):
                    actualfk = self.prob_im_final_is_real[k]
                    d_gan_loss += [hinge_lsgan_loss_discriminator(self.prob_im_is_real[j], actualfk[j]) for j in
                                   range(len(actualfk))]

                d_gan_loss = tf.add_n(d_gan_loss) / float(self.args.n_scale)
                d_loss = d_gan_loss

        self.losses = {'gan_loss': g_gan_loss, 'sal_loss': self.LAMBDAsal * sal_loss,
                       'saldiff_inc':self.LAMBDAsal * saldiff_inc,
                       'saldiff_dec':self.LAMBDAsal *saldiff_dec, 'd_loss': d_loss}

        self.model_vars = tf.trainable_variables()

        if gpu == 0:
            for var in self.model_vars:
                print(var.name)

        self.d_vars = [var for var in self.model_vars if 'disc' in var.name]
        self.g_vars = [var for var in self.model_vars if 'gen' in var.name]
        with tf.variable_scope('ADAM_op', reuse=gpu > 0):
            Ggrads = self.optimizer.compute_gradients(g_loss, self.g_vars)
            Dgrads = self.optimizerD.compute_gradients(d_loss, self.d_vars)

        # Summary variables for tensorboard
        g_loss_summ = tf.summary.merge([tf.summary.scalar("loss_all", g_loss),
                                        tf.summary.scalar("g_gan_loss", g_gan_loss),
                                        tf.summary.scalar("sal_loss", sal_loss),
                                        tf.summary.scalar("saldiff_inc", saldiff_inc),
                                        tf.summary.scalar("saldiff_dec", saldiff_dec),
                                        ])
        self.g_loss_summ = [g_loss_summ]
        self.g_loss_summ += [tf.summary.histogram(q + '_fg_inc', v) for q, v in self.fg_params_inc.items()]
        self.g_loss_summ += [tf.summary.histogram(q + '_bg_inc', v) for q, v in self.bg_params_inc.items()]
        self.g_loss_summ += [tf.summary.histogram(q + '_fg_dec', v) for q, v in self.fg_params_dec.items()]
        self.g_loss_summ += [tf.summary.histogram(q + '_bg_dec', v) for q, v in self.bg_params_dec.items()]
        self.d_loss_summ = [tf.summary.scalar("d_loss", d_loss)]

        return Ggrads, Dgrads

    def average_gradients(self, tower_grads):
        """Calculate the average gradient for each shared variable across all towers.
        Note that this function provides a synchronization point across all towers.
        Args:
          tower_grads: List of lists of (gradient, variable) tuples. The outer list
            is over individual gradients. The inner list is over the gradient
            calculation for each tower.
        Returns:
           List of pairs of (gradient, variable) where the gradient has been averaged
           across all towers.
        """
        average_grads = []
        for grad_and_vars in zip(*tower_grads):
            # Note that each grad_and_vars looks like the following:
            #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
            grads = []
            for g, var in grad_and_vars:
                # Add 0 dimension to the gradients to represent the tower.
                if g == None:
                    g = tf.zeros_like(var)

                expanded_g = tf.expand_dims(g, 0)

                # Append on a 'tower' dimension which we will average over below.
                grads.append(expanded_g)

            # Average over the 'tower' dimension.
            grad = tf.concat(axis=0, values=grads)
            grad = tf.reduce_mean(grad, 0)

            # Keep in mind that the Variables are redundant because they are shared
            # across towers. So .. we will just return the first tower's pointer to
            # the Variable.
            v = grad_and_vars[0][1]
            grad_and_var = (grad, v)
            average_grads.append(grad_and_var)
        return average_grads

    def train(self):
        """Training Function."""
        # Build the network
        G_tower_grad = []
        D_tower_grad = []
        for i in range(self.nb_gpus):
            print('Building graph for gpu nb %d /n' % i)
            with tf.device('/gpu:%d' % i):
                with tf.name_scope('Tower_%d' % (i)) as scope:
                    self.model_setup(i)
                    # Loss function calculations
                    gradG, gradD = self.compute_losses(i)
                    # print(gradG)
                    # print(gradD)
                    G_tower_grad.append(gradG)
                    D_tower_grad.append(gradD)
                    # tf.get_variable_scope().reuse_variables()

        averageGgrags = self.average_gradients(G_tower_grad)
        averageDgrags = self.average_gradients(D_tower_grad)
        with tf.variable_scope('ADAM_op'):
            self.g_trainer = self.optimizer.apply_gradients(averageGgrags)
            self.d_trainer = self.optimizerD.apply_gradients(averageDgrags)

        for grad, var in averageGgrags:
            if grad is not None:
                self.g_loss_summ.append(tf.summary.histogram(var.op.name + '/gradients', grad))

        for grad, var in averageDgrags:
            if grad is not None:
                self.d_loss_summ.append(tf.summary.histogram(var.op.name + '/gradients', grad))

        self.im_summ = self.viz('viz')

        init = tf.variables_initializer(self.g_vars + [self.global_step] + self.d_vars +
                                        tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='ADAM_op'))
        saver = tf.train.Saver(max_to_keep=2)
        # get_next_val = self.iterator_test.get_next()
        config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        # config.gpu_options.allow_growth = True
        with tf.keras.backend.get_session() as sess:
            sess.run(init)
            # self.salmodel.load_weights(os.path.join(os.path.dirname(self.salmodelpath), "mdsem_model_LSUN_weights.h5"))
            # Restore the model to run the model from last checkpoint
            if self.to_restore:
                chkpt_fname = tf.train.latest_checkpoint(self.checkpoint_dir)
                saver.restore(sess, chkpt_fname)

            writer = tf.summary.FileWriter(self.output_dir)
            writer.add_graph(sess.graph)

            if not os.path.exists(self.output_dir):
                os.makedirs(self.output_dir)

            start_decay = np.floor(self.nb_its * self.start_decay_p / 100.)
            sess.run(self.Giterator.initializer)
            for it in tqdm(range(sess.run(self.global_step), self.nb_its, self.nb_gpus * self.batch_size),
                           desc='iterations'):
                self.base_lr = self.args.lr
                self.base_lrd = self.args.lrd
                if np.mod(it, 5000) == 0:
                    sess.run(tf.assign(self.global_step, it))
                    saver.save(sess, os.path.join(self.output_dir, "saledit"), global_step=it)
                if it > start_decay:
                    self.base_lr -= ((float(it) - start_decay) / (self.nb_its - start_decay)) * self.base_lr
                    self.base_lrd -= ((float(it) - start_decay) / (self.nb_its - start_decay)) * self.base_lrd
                try:
                    _ = sess.run(self.g_trainer, feed_dict={self.lr: self.base_lr, self.isTrain: True})
                    _ = sess.run(self.d_trainer, feed_dict={self.lrd: self.base_lrd, self.isTrain: True})

                    if np.mod(it, self.print_freq) == 0:
                        im_summ, losses, summary_strG, summary_strD = sess.run(
                            [self.im_summ, self.losses, self.g_loss_summ, self.d_loss_summ],
                            feed_dict={self.lr: self.base_lr, self.lrd: self.base_lrd, self.isTrain: True})
                        logging = ['it%d, ' % it]
                        logging += ['lr: %.6f, ' % self.base_lr]
                        logging += ['lrd: %.6f, ' % self.base_lrd]
                        logging += [h + ': %.3f, ' % losses[h] for h in list(losses.keys())]
                        print(''.join(logging))
                        with open(os.path.join(self.output_dir, 'logs.txt'), "a") as log_file:
                            log_file.write('%s\n' % (''.join(logging)))
                        [writer.add_summary(summary_strG[j], self.num_fake_inputs)
                         for j in range(len(summary_strG))]
                        [writer.add_summary(summary_strD[j], self.num_fake_inputs)
                         for j in range(len(summary_strD))]
                        writer.add_summary(im_summ, it)
                    writer.flush()
                    self.num_fake_inputs += self.nb_gpus
                    # pbar.update(1)

                except tf.errors.OutOfRangeError:
                    sess.run(self.Giterator.initializer)
            sess.run(tf.assign(self.global_step, it))
            saver.save(sess, os.path.join(self.output_dir, "saledit"), global_step=it)
            # pbar.update(self.batch_size * self.nb_gpus)

        sess.close()
        tf.reset_default_graph()

    def test(self):
        """Testing Function."""
        # Build the network
        self.model_setup_test_HR(0)
        # self.model_setup(0)
        init = tf.variables_initializer(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='ADAM'))
        saver = tf.train.Saver(max_to_keep=2)
        config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        # config.gpu_options.allow_growth = True
        with tf.keras.backend.get_session() as sess:
            sess.run(init)
            chkpt_fname = tf.train.latest_checkpoint(self.checkpoint_dir)
            saver.restore(sess, chkpt_fname)
            epoch_id = sess.run(self.global_step)

            imsaving_path = os.path.join(self.checkpoint_dir, 'results', 'epoch%d' % (epoch_id),
                                         'ims')
            # totar = os.path.join(self.checkpoint_dir, 'results')
            if not os.path.exists(imsaving_path):
                os.makedirs(imsaving_path)

            sess.run(self.Giterator.initializer)
            webpage = html.HTML(os.path.dirname(imsaving_path),
                                'Experiment = %s, Phase = %s, Epoch = %s' % ('exp', 'test', epoch_id))

            mean_inc = []
            mean_dec = []

            try:
                with tqdm(total=int(self.max_images_G)) as pbar:
                    while True:
                        im, fake_ims_hr_inc, mask, mask_lr, salinit, salfinals_inc, fgparams_inc, bgparams_inc = sess.run(
                            [self.im_hr,
                             self.imfinals,
                             self.mask_hr,
                             self.bin_mask,
                             self.sal_init,
                             self.salfinals,
                             self.fgparams,
                             self.bgparams,
                             ], feed_dict={self.isTrain: False}
                        )

                        scaling_weights = (np.size(mask_lr[0, :, :, :])) / np.sum(mask_lr, axis=(1, 2, 3))
                        mean_sal_init = np.mean(salinit * mask_lr, axis=(1, 2, 3)) * scaling_weights
                        mean_salfinals = [np.mean(salfinal * mask_lr, axis=(1, 2, 3)) * scaling_weights
                                              for salfinal in salfinals_inc]

                        mean_inc.append(mean_salfinals[0] - mean_sal_init)
                        mean_dec.append(mean_salfinals[1] - mean_sal_init)
                        all_dict = collections.OrderedDict()
                        all_dict['im%d' % self.num_fake_inputs] = (im + 1) * 0.5
                        all_dict['mask%d.png' % self.num_fake_inputs] = mask

                        suffix = '%d_%.3f' % (self.num_fake_inputs, mean_sal_init)
                        all_dict_bis = collections.OrderedDict()
                        all_dict_bis['im%d' % self.num_fake_inputs] = (im + 1) * 0.5
                        all_dict_bis['salinit' + suffix + '.png'] = salinit

                        for k in range(len(fake_ims_hr_inc)):
                            txt_params = ''
                            txt_params = print_params(fgparams_inc[k], txt_params)
                            txt_params = print_params(bgparams_inc[k], txt_params)
                            suffix = '%d_%.3f' % (self.num_fake_inputs, mean_salfinals[k] - mean_sal_init)
                            suffixsal = '%d_%.3f.png' % (
                                self.num_fake_inputs, mean_salfinals[k] - mean_sal_init) + 'separator' + txt_params
                            if k==0:
                                all_dict['imfinal_inc%d' % (k) + suffix] = fake_ims_hr_inc[k]
                                all_dict_bis['salfinal_inc%d' % (k) + suffixsal] = salfinals_inc[k]
                            else:
                                all_dict['imfinal_dec%d' % (k) + suffix] = fake_ims_hr_inc[k]
                                all_dict_bis['salfinal_dec%d' % (k) + suffixsal] = salfinals_inc[k]

                        saveImages_bis_bis(webpage, all_dict, imsaving_path)
                        saveImages_bis_bis(webpage, all_dict_bis, imsaving_path)
                        self.num_fake_inputs += 1
                        pbar.update(self.batch_size)

            except tf.errors.OutOfRangeError:
                pass
        webpage.save()
        sess.close()
        tf.reset_default_graph()
        f = open(os.path.join(os.path.dirname(imsaving_path), 'mean_sals.txt'), 'w')
        f.write('Mean saliency increase %.3f\n' % np.mean(mean_inc))
        f.write('Mean saliency decrease %.3f\n' % np.mean(mean_dec))
        f.write('===================================\n\n')

class adv_increase_sftmx:
    """The trainer class"""

    def __init__(self, g_model, d_model, salmodel_path, Giterator, print_freq, log_dir, to_restore,
                 base_lr, max_step, checkpoint_dir, max_images_G, batch_size, args):

        # self.sess = tf.InteractiveSession(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
        # K.set_session(self.sess)

        self.print_freq = print_freq
        self.LAMBDAsal = args.LAMBDAsal
        self.LAMBDAFM = args.LAMBDAFM
        self.LAMBDA_r = args.LAMBDA_r
        self.LAMBDAD = args.LAMBDAD
        self.LAMBDA_p = args.LAMBDA_p
        self.output_dir = log_dir
        self.images_dir = os.path.join(self.output_dir, 'imgs')
        self.num_imgs_to_save = 20
        self.to_restore = to_restore
        self.base_lr = base_lr
        self.base_lrd = args.lrd
        self.max_step = max_step
        self.nb_its = args.nb_its
        self.checkpoint_dir = checkpoint_dir
        self.Giterator = Giterator
        self.max_images_G = max_images_G
        
        self.start_decay_p = args.startdecay
        self.batch_size = batch_size
        self.nb_iterations = max_images_G / self.batch_size
        self.salmodelpath = salmodel_path
        self.nb_gpus = args.nb_gpu
        self.gmodel = g_model
        self.dmodel = d_model
        self.lr = tf.placeholder(tf.float32, name='lr')
        self.lrd = tf.placeholder(tf.float32, name='lrd')
        self.isTrain = tf.placeholder(tf.bool, shape=(), name='istrain')
        self.args = args
        tf.keras.backend.set_learning_phase(0)  # this line is what allows us to freeze the graph without errors
        with tf.variable_scope('snet', reuse=False):
            self.salmodel = tf.keras.models.load_model(salmodel_path)
        print("Loaded sal model from disk")

    def model_setup(self, gpu):
        """
        This function sets up the model to train.
        """

        self.g_model = self.gmodel(self.args)
        # self.d_model = self.dmodel(NF=self.args.ndf, n_scale=self.args.n_scale, training=self.isTrain, n_dis=self.args.n_dis,
        #                            do_norm=self.args.donorm)
        self.d_model = self.dmodel(self.args)
        self.im, self.mask, self.s = self.Giterator.get_next()
        gk = gaussian_kernel()
        self.mask = apply_gaussian_kernel(self.mask, gk)
        means = np.array([103.939, 116.779, 123.68])
        means = means[None, None, None, :]
        sizes = self.im.get_shape().as_list()
        means = np.tile(means, [1, sizes[1], sizes[2], 1])
        means = means[:, :, :, ::-1]
        self.bin_mask = (self.mask + 1) * 0.5
        self.global_step = tf.train.get_or_create_global_step()
        self.num_fake_inputs = 0
        with tf.variable_scope('snet', reuse=True):
            self.sal_init = get_sal_keras_sftmax(self.salmodel, self.im, means, gk)

        inputs = {
            'im': self.im,
            'masks': self.mask,
            's': self.s,
            'salinit': self.sal_init}

        with tf.variable_scope('gen', reuse=gpu > 0):
            self.genim_list, self.fg_params, self.bg_params = self.g_model.get_outputs(inputs)

        with tf.variable_scope('disc', reuse=gpu > 0):
            self.prob_im_is_real, self.features_im_real, self.prob_im_final_is_real, self.features_im_final = \
                self.d_model.get_outputs(self.im, self.genim_list)

        with tf.variable_scope('snet', reuse=True):
            self.sal_final = get_sal_keras_sftmax(self.salmodel, self.genim_list[-1], means, gk)

        # slim = tf.contrib.slim
        # vgg_vars = slim.get_variables_to_restore(include=['vgg_16'], exclude=self.exclude)
        # self.init_fn = slim.assign_from_checkpoint_fn('./vgg_16.ckpt', vgg_vars)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr, beta1=0.5)
        self.optimizerD = tf.train.AdamOptimizer(learning_rate=self.lrd, beta1=0.5)

    def model_setup_test_HR(self, gpu):
        """
        This function sets up the model to train.
        """

        self.g_model = self.gmodel(self.args)
        self.d_model = self.dmodel(self.args)
        # self.vgg16 = vgg.vgg_16
        self.im, self.mask, self.im_hr, self.mask_hr, self.s = self.Giterator.get_next()
        gk = gaussian_kernel()
        self.mask = apply_gaussian_kernel(self.mask, gk)
        self.mask_hr = apply_gaussian_kernel(self.mask_hr, gk)
        means = np.array([103.939, 116.779, 123.68])
        means = means[None, None, None, :]
        sizes = self.im.get_shape().as_list()
        means = np.tile(means, [1, sizes[1], sizes[2], 1])
        means = means[:, :, :, ::-1]
        self.bin_mask = (self.mask + 1) * 0.5
        self.global_step = tf.train.get_or_create_global_step()
        self.num_fake_inputs = 0
        with tf.variable_scope('snet', reuse=True):
            sal_init = get_sal_keras_sftmax(self.salmodel, self.im, means, gk)
            self.sal_init = get_sal_keras(self.salmodel, self.im, means, gk)
        self.nb_zs = 10
        self.salfinals = []
        self.imfinals = []
        self.fgparams = []
        self.bgparams = []
        self.probs_ims_are_real = []
        self.probs_imfinals_are_real = []
        self.features_ims_real = []
        self.features_ims_final = []
        self.f3s = []
        self.losses = []
        inputs = {
            'im': self.im,
            'masks': self.mask}

        with tf.variable_scope('gen', reuse=False):
            im_final, fg_params, bg_params = self.g_model.get_outputs(inputs)
        with tf.variable_scope('disc', reuse=False):
            self.prob_im_is_real, self.features_im_real, self.prob_im_final_is_real, self.features_im_final = \
                self.d_model.get_outputs(self.im, im_final)

        im_final_hr = self.g_model.get_outputs_hr((self.im_hr + 1) * 0.5, self.mask_hr, fg_params, bg_params)
        im_final_hr = tf.clip_by_value(im_final_hr, -1., 1.)
        # im_final_hr = apply_transformations_sharp_exp_cont_tone_color_bis((self.im_hr + 1) * 0.5, self.mask_hr, fg_params,
        #                                                               bg_params,
        #                                                               self.g_model.tf1, self.g_model.tf2,
        #                                                               self.g_model.ksize)

        self.imfinals.append(im_final_hr)
        self.fgparams.append(fg_params)
        self.bgparams.append(bg_params)
        with tf.variable_scope('snet', reuse=True):
            self.salfinals.append(get_sal_keras(self.salmodel, im_final[-1], means, gk))

        # slim = tf.contrib.slim
        # vgg_vars = slim.get_variables_to_restore(include=['vgg_16'], exclude=exclude)
        # self.init_fn = slim.assign_from_checkpoint_fn('./vgg_16.ckpt', vgg_vars)
        # self.init_fn = slim.assign_from_checkpoint_fn('/mnt/ilcompfad1/user/bylinski/gazeshift/gazeshift/vgg_16.ckpt', vgg_vars)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr, beta1=0.5)

    def viz(self, name):
        im = tf.concat([self.im, tf.image.grayscale_to_rgb(self.mask)] + self.genim_list, axis=2)
        im = (im + 1.0) * 0.5  # 127.5
        # im = tf.image.hsv_to_rgb(im) * 255
        im = im * 255
        # im = tf.concat([im, tf.clip_by_value(im, 0, 255)], axis=1)
        im = tf.cast(im, tf.uint8)
        return tf.summary.image(name, im, max_outputs=50)

    def compute_losses(self, gpu):

        with tf.name_scope('losses'):
            with tf.name_scope('g_loss'):
                # l2_loss = tf.reduce_mean(tf.square(self.fg_params)) + tf.reduce_mean(tf.square(self.bg_params),
                # name='l2_loss')

                numelmask = tf.reduce_sum(self.bin_mask, axis=[1, 2, 3])
                numelall = tf.ones_like(numelmask) * tf.size(self.bin_mask[0], out_type=tf.float32)
                numelmask = tf.where(tf.equal(numelmask, 0), numelall, numelmask)
                weight_recon_loss = numelall / numelmask

                saldiff = tf.reduce_mean((tf.reduce_mean(self.bin_mask * self.sal_final, axis=[1, 2, 3]) -
                                          tf.reduce_mean(self.bin_mask * self.sal_init,
                                                         axis=[1, 2, 3])) * weight_recon_loss)

                sal_loss = -saldiff

                g_gan_loss = []
                for k in range(len(self.prob_im_final_is_real)):
                    actualk = self.prob_im_final_is_real[k]
                    g_gan_loss += [hinge_lsgan_loss_generator(actualk[j]) for j in range(len(actualk))]

                g_gan_loss = tf.add_n(g_gan_loss) / float(self.args.n_scale)
                g_loss = g_gan_loss + self.LAMBDAsal * sal_loss

            d_gan_loss = []
            with tf.name_scope('d_loss'):
                for k in range(len(self.prob_im_final_is_real)):
                    actualfk = self.prob_im_final_is_real[k]
                    d_gan_loss += [hinge_lsgan_loss_discriminator(self.prob_im_is_real[j], actualfk[j]) for j in
                                   range(len(actualfk))]

                d_gan_loss = tf.add_n(d_gan_loss) / float(self.args.n_scale)
                d_loss = d_gan_loss

        self.losses = {'gan_loss': g_gan_loss, 'sal_loss': self.LAMBDAsal * sal_loss, 'd_loss': d_loss}

        self.model_vars = tf.trainable_variables()

        if gpu == 0:
            for var in self.model_vars:
                print(var.name)

        self.d_vars = [var for var in self.model_vars if 'disc' in var.name]
        self.g_vars = [var for var in self.model_vars if 'gen' in var.name]
        with tf.variable_scope('ADAM_op', reuse=gpu > 0):
            Ggrads = self.optimizer.compute_gradients(g_loss, self.g_vars)
            Dgrads = self.optimizerD.compute_gradients(d_loss, self.d_vars)

        # Summary variables for tensorboard
        g_loss_summ = tf.summary.merge([tf.summary.scalar("loss_all", g_loss),
                                        tf.summary.scalar("g_gan_loss", g_gan_loss),
                                        tf.summary.scalar("sal_loss", self.LAMBDAsal * sal_loss),
                                        tf.summary.scalar("sal_diff", self.LAMBDAsal * saldiff),
                                        ])
        self.g_loss_summ = [g_loss_summ]
        self.g_loss_summ += [tf.summary.histogram(q + '_fg', v) for q, v in self.fg_params.items()]
        self.g_loss_summ += [tf.summary.histogram(q + '_bg', v) for q, v in self.bg_params.items()]
        self.d_loss_summ = [tf.summary.scalar("d_loss", d_loss)]

        return Ggrads, Dgrads

    def average_gradients(self, tower_grads):
        """Calculate the average gradient for each shared variable across all towers.
        Note that this function provides a synchronization point across all towers.
        Args:
          tower_grads: List of lists of (gradient, variable) tuples. The outer list
            is over individual gradients. The inner list is over the gradient
            calculation for each tower.
        Returns:
           List of pairs of (gradient, variable) where the gradient has been averaged
           across all towers.
        """
        average_grads = []
        for grad_and_vars in zip(*tower_grads):
            # Note that each grad_and_vars looks like the following:
            #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
            grads = []
            for g, var in grad_and_vars:
                # Add 0 dimension to the gradients to represent the tower.
                if g == None:
                    g = tf.zeros_like(var)

                expanded_g = tf.expand_dims(g, 0)

                # Append on a 'tower' dimension which we will average over below.
                grads.append(expanded_g)

            # Average over the 'tower' dimension.
            grad = tf.concat(axis=0, values=grads)
            grad = tf.reduce_mean(grad, 0)

            # Keep in mind that the Variables are redundant because they are shared
            # across towers. So .. we will just return the first tower's pointer to
            # the Variable.
            v = grad_and_vars[0][1]
            grad_and_var = (grad, v)
            average_grads.append(grad_and_var)
        return average_grads

    def train(self):
        """Training Function."""
        # Build the network
        G_tower_grad = []
        D_tower_grad = []
        for i in range(self.nb_gpus):
            print('Building graph for gpu nb %d /n' % i)
            with tf.device('/gpu:%d' % i):
                with tf.name_scope('Tower_%d' % (i)) as scope:
                    self.model_setup(i)
                    # Loss function calculations
                    gradG, gradD = self.compute_losses(i)
                    # print(gradG)
                    # print(gradD)
                    G_tower_grad.append(gradG)
                    D_tower_grad.append(gradD)
                    # tf.get_variable_scope().reuse_variables()

        averageGgrags = self.average_gradients(G_tower_grad)
        averageDgrags = self.average_gradients(D_tower_grad)
        with tf.variable_scope('ADAM_op'):
            self.g_trainer = self.optimizer.apply_gradients(averageGgrags)
            self.d_trainer = self.optimizerD.apply_gradients(averageDgrags)

        for grad, var in averageGgrags:
            if grad is not None:
                self.g_loss_summ.append(tf.summary.histogram(var.op.name + '/gradients', grad))

        for grad, var in averageDgrags:
            if grad is not None:
                self.d_loss_summ.append(tf.summary.histogram(var.op.name + '/gradients', grad))

        self.im_summ = self.viz('viz')

        init = tf.variables_initializer(self.g_vars + [self.global_step] + self.d_vars +
                                        tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='ADAM_op'))
        saver = tf.train.Saver(max_to_keep=2)
        # get_next_val = self.iterator_test.get_next()
        config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        # config.gpu_options.allow_growth = True
        with tf.keras.backend.get_session() as sess:
            sess.run(init)
            # self.salmodel.load_weights(os.path.join(os.path.dirname(self.salmodelpath), "mdsem_model_LSUN_weights.h5"))
            # Restore the model to run the model from last checkpoint
            if self.to_restore:
                chkpt_fname = tf.train.latest_checkpoint(self.checkpoint_dir)
                saver.restore(sess, chkpt_fname)

            writer = tf.summary.FileWriter(self.output_dir)
            writer.add_graph(sess.graph)

            if not os.path.exists(self.output_dir):
                os.makedirs(self.output_dir)

            start_decay = np.floor(self.nb_its * self.start_decay_p / 100.)
            sess.run(self.Giterator.initializer)
            for it in tqdm(range(sess.run(self.global_step), self.nb_its, self.nb_gpus * self.batch_size),
                           desc='iterations'):
                self.base_lr = self.args.lr
                self.base_lrd = self.args.lrd
                if np.mod(it, 5000) == 0:
                    sess.run(tf.assign(self.global_step, it))
                    saver.save(sess, os.path.join(self.output_dir, "saledit"), global_step=it)
                if it > start_decay:
                    self.base_lr -= ((float(it) - start_decay) / (self.nb_its - start_decay)) * self.base_lr
                    self.base_lrd -= ((float(it) - start_decay) / (self.nb_its - start_decay)) * self.base_lrd
                try:
                    _ = sess.run(self.g_trainer, feed_dict={self.lr: self.base_lr, self.isTrain: True})
                    _ = sess.run(self.d_trainer, feed_dict={self.lrd: self.base_lrd, self.isTrain: True})

                    if np.mod(it, self.print_freq) == 0:
                        im_summ, losses, summary_strG, summary_strD = sess.run(
                            [self.im_summ, self.losses, self.g_loss_summ, self.d_loss_summ],
                            feed_dict={self.lr: self.base_lr, self.lrd: self.base_lrd, self.isTrain: True})
                        logging = ['it%d, ' % it]
                        logging += ['lr: %.6f, ' % self.base_lr]
                        logging += ['lrd: %.6f, ' % self.base_lrd]
                        logging += [h + ': %.3f, ' % losses[h] for h in list(losses.keys())]
                        print(''.join(logging))
                        with open(os.path.join(self.output_dir, 'logs.txt'), "a") as log_file:
                            log_file.write('%s\n' % (''.join(logging)))
                        [writer.add_summary(summary_strG[j], self.num_fake_inputs)
                         for j in range(len(summary_strG))]
                        [writer.add_summary(summary_strD[j], self.num_fake_inputs)
                         for j in range(len(summary_strD))]
                        writer.add_summary(im_summ, it)
                    writer.flush()
                    self.num_fake_inputs += self.nb_gpus
                    # pbar.update(1)

                except tf.errors.OutOfRangeError:
                    sess.run(self.Giterator.initializer)
            sess.run(tf.assign(self.global_step, it))
            saver.save(sess, os.path.join(self.output_dir, "saledit"), global_step=it)
            # pbar.update(self.batch_size * self.nb_gpus)

        sess.close()
        tf.reset_default_graph()

    def resave(self):
        """Training Function."""
        # Build the network
        G_tower_grad = []
        D_tower_grad = []
        for i in range(self.nb_gpus):
            print('Building graph for gpu nb %d /n' % i)
            with tf.device('/gpu:%d' % i):
                with tf.name_scope('Tower_%d' % (i)) as scope:
                    self.model_setup(i)
                    # Loss function calculations
                    gradG, gradD = self.compute_losses(i)
                    # print(gradG)
                    # print(gradD)
                    G_tower_grad.append(gradG)
                    D_tower_grad.append(gradD)
                    # tf.get_variable_scope().reuse_variables()

        averageGgrags = self.average_gradients(G_tower_grad)
        averageDgrags = self.average_gradients(D_tower_grad)
        with tf.variable_scope('ADAM_op'):
            self.g_trainer = self.optimizer.apply_gradients(averageGgrags)
            self.d_trainer = self.optimizerD.apply_gradients(averageDgrags)

        for grad, var in averageGgrags:
            if grad is not None:
                self.g_loss_summ.append(tf.summary.histogram(var.op.name + '/gradients', grad))

        for grad, var in averageDgrags:
            if grad is not None:
                self.d_loss_summ.append(tf.summary.histogram(var.op.name + '/gradients', grad))

        self.im_summ = self.viz('viz')

        init = tf.variables_initializer(self.g_vars + [self.global_step] + self.d_vars +
                                        tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='ADAM_op'))
        saver = tf.train.Saver(max_to_keep=2)
        # get_next_val = self.iterator_test.get_next()
        config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        # config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            sess.run(init)
            self.salmodel.load_weights(os.path.join(os.path.dirname(self.salmodelpath), "mdsem_model_LSUN_weights.h5"))
            self.init_fn(sess)
            # Restore the model to run the model from last checkpoint
            if self.to_restore:
                chkpt_fname = tf.train.latest_checkpoint(self.checkpoint_dir)
                saver.restore(sess, chkpt_fname)
                new_saledit_path = os.path.join(os.path.dirname(self.output_dir), os.path.basename(self.checkpoint_dir))
                saver.save(sess, os.path.join(new_saledit_path, "new_saledit"),
                           global_step=sess.run(self.global_step))

            sess.close()
        tf.reset_default_graph()

    def test(self):
        """Training Function."""
        # Build the network
        self.model_setup_test_HR(0)
        # self.model_setup(0)
        init = tf.variables_initializer(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='ADAM'))
        saver = tf.train.Saver(max_to_keep=2)
        config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        # config.gpu_options.allow_growth = True
        with tf.keras.backend.get_session() as sess:
            sess.run(init)
            chkpt_fname = tf.train.latest_checkpoint(self.checkpoint_dir)
            saver.restore(sess, chkpt_fname)
            new_saledit_path = os.path.join(os.path.dirname(self.output_dir), os.path.basename(self.checkpoint_dir))
            saver.save(sess, os.path.join(new_saledit_path, "test_saledit"),
                       global_step=sess.run(self.global_step))
            epoch_id = sess.run(self.global_step)

            imsaving_path = os.path.join(self.checkpoint_dir, 'results', 'epoch%d' % (epoch_id),
                                         'ims')
            dictsaving_path = os.path.join(self.checkpoint_dir, 'results', 'epoch%d' % (epoch_id),
                                         'params')
            if not os.path.exists(imsaving_path):
                os.makedirs(imsaving_path)
            if not os.path.exists(dictsaving_path):
                os.makedirs(dictsaving_path)

            sess.run(self.Giterator.initializer)
            webpage = html.HTML(os.path.dirname(imsaving_path),
                                'Experiment = %s, Phase = %s, Epoch = %s' % ('exp', 'test', epoch_id))

            mean_inc = []
            fgparams_dict_all_inc = {key: [] for key in list(self.fgparams[0].keys())}
            bgparams_dict_all_inc = {key: [] for key in list(self.bgparams[0].keys())}

            try:
                with tqdm(total=int(self.max_images_G)) as pbar:
                    while True:
                        im, fake_ims_hr_inc, mask, mask_lr, salinit, salfinals_inc, fgparams_inc, bgparams_inc = sess.run(
                            [self.im_hr,
                             self.imfinals,
                             self.mask_hr,
                             self.bin_mask,
                             self.sal_init,
                             self.salfinals,
                             self.fgparams,
                             self.bgparams,
                             ], feed_dict={self.isTrain: False}
                        )
                        im_dict = {}
                        im_dict['fg'] = fgparams_inc[0]
                        im_dict['bg'] = bgparams_inc[0]
                        with open(os.path.join(dictsaving_path, 'im%d.pickle'%self.num_fake_inputs),'wb') as pick:
                            pickle.dump(im_dict, pick)

                        for key, value in fgparams_inc[0].items():
                            fgparams_dict_all_inc[key].append(fgparams_inc[0][key])
                        for key, value in bgparams_inc[0].items():
                            bgparams_dict_all_inc[key].append(bgparams_inc[0][key])

                        scaling_weights = (np.size(mask_lr[0, :, :, :])) / np.sum(mask_lr, axis=(1, 2, 3))
                        mean_sal_init = np.mean(salinit * mask_lr, axis=(1, 2, 3)) * scaling_weights
                        mean_salfinals_inc = [np.mean(salfinal * mask_lr, axis=(1, 2, 3)) * scaling_weights
                                              for salfinal in salfinals_inc]

                        mean_inc.append(mean_salfinals_inc[0] - mean_sal_init)
                        all_dict = collections.OrderedDict()
                        all_dict['im%d' % self.num_fake_inputs] = (im + 1) * 0.5
                        all_dict['mask%d.png' % self.num_fake_inputs] = mask

                        suffix = '%d_%.3f' % (self.num_fake_inputs, mean_sal_init)
                        all_dict_bis = collections.OrderedDict()
                        all_dict_bis['im%d' % self.num_fake_inputs] = (im + 1) * 0.5
                        all_dict_bis['salinit' + suffix + '.png'] = salinit
                        # fake_ims_hr = [apply_transformations_np((im+1)*0.5, mask, fgparams[k], bgparams[k]) for k in range(len(fgparams))]

                        for k in range(len(fake_ims_hr_inc)):
                            txt_params = ''
                            txt_params = print_params(fgparams_inc[k], txt_params)
                            txt_params = print_params(bgparams_inc[k], txt_params)
                            suffix = '%d_%.3f' % (self.num_fake_inputs, mean_salfinals_inc[k] - mean_sal_init)
                            suffixsal = '%d_%.3f.png' % (
                                self.num_fake_inputs, mean_salfinals_inc[k] - mean_sal_init) + 'separator' + txt_params
                            all_dict['imfinal_inc%d' % (k) + suffix] = fake_ims_hr_inc[k]
                            all_dict_bis['salfinal_inc%d' % (k) + suffixsal] = salfinals_inc[k]

                        saveImages_bis_bis(webpage, all_dict, imsaving_path)
                        saveImages_bis_bis(webpage, all_dict_bis, imsaving_path)
                        self.num_fake_inputs += 1
                        pbar.update(self.batch_size)
            except tf.errors.OutOfRangeError:
                pass
        webpage.save()
        sess.close()
        tf.reset_default_graph()
        f = open(os.path.join(os.path.dirname(imsaving_path), 'mean_sals.txt'), 'w')
        f.write('Mean saliency increase %.3f\n' % np.mean(mean_inc))
        f.write('===================================\n\n')
        stds_list_fg = []
        stds_list_bg = []
        for key, value in fgparams_dict_all_inc.items():
            if 'color' in key:
                stacked_color_fg_inc = np.concatenate(fgparams_dict_all_inc['color'], axis=0).squeeze()
                stacked_color_fg_inc = np.reshape(stacked_color_fg_inc, [-1, 24])
                f.write('FG %s : %.3f (%.3f)\n' % (
                    key, np.mean(stacked_color_fg_inc), np.std(stacked_color_fg_inc)))
                stds_list_fg.append(np.std(stacked_color_fg_inc))

                stacked_color_bg_inc = np.concatenate(bgparams_dict_all_inc['color'], axis=0).squeeze()
                stacked_color_bg_inc = np.reshape(stacked_color_bg_inc, [-1, 24])
                f.write('BG %s : %.3f (%.3f)\n' % (
                    key, np.mean(stacked_color_bg_inc), np.std(stacked_color_bg_inc)))
                stds_list_bg.append(np.std(stacked_color_bg_inc))

                fgparams_dict_all_inc['color'] = sum(fgparams_dict_all_inc['color']) / len(
                    fgparams_dict_all_inc['color'])
                bgparams_dict_all_inc['color'] = sum(bgparams_dict_all_inc['color']) / len(
                    bgparams_dict_all_inc['color'])
            elif 'tone' in key:
                stacked_tone_fg_inc = np.concatenate(fgparams_dict_all_inc['tone'], axis=0).squeeze()
                stacked_tone_fg_inc = np.reshape(stacked_tone_fg_inc, [-1, 8])
                f.write('FG %s : %.3f (%.3f)\n' % (
                    key, np.mean(stacked_tone_fg_inc), np.std(stacked_tone_fg_inc)))
                stds_list_fg.append(np.std(stacked_tone_fg_inc))

                stacked_tone_bg_inc = np.concatenate(bgparams_dict_all_inc['tone'], axis=0).squeeze()
                stacked_tone_bg_inc = np.reshape(stacked_tone_bg_inc, [-1, 8])
                f.write('BG %s : %.3f (%.3f)\n' % (
                    key, np.mean(stacked_tone_bg_inc), np.std(stacked_tone_bg_inc)))
                stds_list_bg.append(np.std(stacked_tone_bg_inc))

                fgparams_dict_all_inc['tone'] = sum(fgparams_dict_all_inc['tone']) / len(fgparams_dict_all_inc['tone'])
                bgparams_dict_all_inc['tone'] = sum(bgparams_dict_all_inc['tone']) / len(bgparams_dict_all_inc['tone'])
            else:
                stacked_fg_inc = np.stack(fgparams_dict_all_inc[key]).squeeze()
                f.write('FG %s : %.3f (%.3f)\n' % (key, np.mean(stacked_fg_inc), np.std(stacked_fg_inc)))
                stds_list_fg.append(np.std(stacked_fg_inc))
                stacked_bg_inc = np.stack(bgparams_dict_all_inc[key]).squeeze()
                f.write('BG %s : %.3f (%.3f)\n' % (key, np.mean(stacked_bg_inc), np.std(stacked_bg_inc)))
                stds_list_bg.append(np.std(stacked_bg_inc))

                fgparams_dict_all_inc[key] = sum(fgparams_dict_all_inc[key]) / len(fgparams_dict_all_inc[key])
                bgparams_dict_all_inc[key] = sum(bgparams_dict_all_inc[key]) / len(bgparams_dict_all_inc[key])

            f.write('===================\n')
        f.write('Mean FG std over all parameters: %.3f\n' % np.mean(stds_list_fg))
        f.write('Mean BG std over all parameters: %.3f\n' % np.mean(stds_list_bg))

        with open(os.path.join(os.path.dirname(imsaving_path), 'mean_fg_params.pickle'), 'wb') as handle:
            pickle.dump(fgparams_dict_all_inc, handle)
        with open(os.path.join(os.path.dirname(imsaving_path), 'mean_bg_params.pickle'), 'wb') as handle:
            pickle.dump(bgparams_dict_all_inc, handle)

class adv_increase_multistyle_sftmx:
    """The trainer class"""

    def __init__(self, g_model, d_model, salmodel_path, Giterator, print_freq, log_dir, to_restore,
                 base_lr, max_step, checkpoint_dir, max_images_G, batch_size, args):

        # self.sess = tf.InteractiveSession(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
        # K.set_session(self.sess)

        self.print_freq = print_freq
        self.LAMBDAsal = args.LAMBDAsal
        self.LAMBDAFM = args.LAMBDAFM
        self.LAMBDA_r = args.LAMBDA_r
        self.LAMBDAD = args.LAMBDAD
        self.LAMBDA_p = args.LAMBDA_p
        self.output_dir = log_dir
        self.images_dir = os.path.join(self.output_dir, 'imgs')
        self.num_imgs_to_save = 20
        self.to_restore = to_restore
        self.base_lr = base_lr
        self.base_lrd = args.lrd
        self.max_step = max_step
        self.nb_its = args.nb_its
        self.checkpoint_dir = checkpoint_dir
        self.Giterator = Giterator
        self.max_images_G = max_images_G
        
        self.start_decay_p = args.startdecay
        self.batch_size = batch_size
        self.nb_iterations = max_images_G / self.batch_size
        self.salmodelpath = salmodel_path
        self.nb_gpus = args.nb_gpu
        self.gmodel = g_model
        self.dmodel = d_model
        self.lr = tf.placeholder(tf.float32, name='lr')
        self.lrd = tf.placeholder(tf.float32, name='lrd')
        self.isTrain = tf.placeholder(tf.bool, shape=(), name='istrain')
        self.args = args
        tf.keras.backend.set_learning_phase(0)  # this line is what allows us to freeze the graph without errors
        with tf.variable_scope('snet', reuse=False):
            self.salmodel = tf.keras.models.load_model(salmodel_path)
        print("Loaded sal model from disk")

    def model_setup(self, gpu):
        """
        This function sets up the model to train.
        """

        self.g_model = self.gmodel(self.args)
        # self.d_model = self.dmodel(NF=self.args.ndf, n_scale=self.args.n_scale, training=self.isTrain, n_dis=self.args.n_dis,
        #                            do_norm=self.args.donorm)
        self.d_model = self.dmodel(self.args)
        self.im, self.mask, self.s = self.Giterator.get_next()
        gk = gaussian_kernel()
        self.mask = apply_gaussian_kernel(self.mask, gk)
        means = np.array([103.939, 116.779, 123.68])
        means = means[None, None, None, :]
        sizes = self.im.get_shape().as_list()
        means = np.tile(means, [1, sizes[1], sizes[2], 1])
        means = means[:, :, :, ::-1]
        self.bin_mask = (self.mask + 1) * 0.5
        self.global_step = tf.train.get_or_create_global_step()
        self.num_fake_inputs = 0
        with tf.variable_scope('snet', reuse=True):
            self.sal_init = get_sal_keras_sftmax(self.salmodel, self.im, means, gk)

        self.z1 = tf.random_normal(shape=[tf.shape(self.im)[0], self.args.zdim], name='z1')
        self.z2 = tf.random_normal(shape=[tf.shape(self.im)[0], self.args.zdim], name='z2')

        inputs = {
            'im': self.im,
            'masks': self.mask,
            's': self.s,
            'salinit': self.sal_init,
            'z': self.z1}

        with tf.variable_scope('gen', reuse=gpu>0):
            self.genim_list, self.fg_params, self.bg_params, self.zrecon = self.g_model.get_outputs(inputs)

        inputs['z'] = self.z2
        with tf.variable_scope('gen', reuse=True):
            self.genim_list_z2, _, _, _ = self.g_model.get_outputs(inputs)

        with tf.variable_scope('disc', reuse=gpu > 0):
            self.prob_im_is_real, self.features_im_real, self.prob_im_final_is_real, self.features_im_final = \
                self.d_model.get_outputs(self.im, self.genim_list)

        with tf.variable_scope('snet', reuse=True):
            self.sal_final = get_sal_keras_sftmax(self.salmodel, self.genim_list[-1], means, gk)

        # slim = tf.contrib.slim
        # vgg_vars = slim.get_variables_to_restore(include=['vgg_16'], exclude=self.exclude)
        # self.init_fn = slim.assign_from_checkpoint_fn('./vgg_16.ckpt', vgg_vars)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr, beta1=0.5)
        self.optimizerD = tf.train.AdamOptimizer(learning_rate=self.lrd, beta1=0.5)

    def model_setup_test_HR(self, gpu):
        """
        This function sets up the model to train.
        """

        self.g_model = self.gmodel(self.args)
        self.d_model = self.dmodel(self.args)
        # self.vgg16 = vgg.vgg_16
        self.im, self.mask, self.im_hr, self.mask_hr, self.s = self.Giterator.get_next()
        gk = gaussian_kernel()
        self.mask = apply_gaussian_kernel(self.mask, gk)
        self.mask_hr = apply_gaussian_kernel(self.mask_hr, gk)
        means = np.array([103.939, 116.779, 123.68])
        means = means[None, None, None, :]
        sizes = self.im.get_shape().as_list()
        means = np.tile(means, [1, sizes[1], sizes[2], 1])
        means = means[:, :, :, ::-1]
        self.bin_mask = (self.mask + 1) * 0.5
        self.global_step = tf.train.get_or_create_global_step()
        self.num_fake_inputs = 0
        with tf.variable_scope('snet', reuse=True):
            sal_init = get_sal_keras_sftmax(self.salmodel, self.im, means, gk)
            self.sal_init = get_sal_keras(self.salmodel, self.im, means, gk)

        self.nb_zs = 10
        self.salfinals = []
        self.imfinals = []
        self.fgparams = []
        self.bgparams = []
        self.probs_ims_are_real = []
        self.probs_imfinals_are_real = []
        self.features_ims_real = []
        self.features_ims_final = []
        self.f3s = []
        self.losses = []

        for k in range(self.nb_zs):

            self.z = tf.random_normal(shape=[tf.shape(self.im)[0], self.args.zdim], name='z%d'%k)
            inputs = {
                'im': self.im,
                'masks': self.mask,
                's': self.s,
                'salinit': self.sal_init,
                'z': self.z}

            with tf.variable_scope('gen', reuse = k>0):
                im_final, fg_params, bg_params, _ = self.g_model.get_outputs(inputs)
            im_final_hr = self.g_model.get_outputs_hr((self.im_hr + 1) * 0.5, self.mask_hr, fg_params, bg_params)
            im_final_hr = tf.clip_by_value(im_final_hr, -1., 1.)

            self.imfinals.append(im_final_hr)
            self.fgparams.append(fg_params)
            self.bgparams.append(bg_params)
            with tf.variable_scope('snet', reuse=True):
                self.salfinals.append(get_sal_keras(self.salmodel, im_final[-1], means, gk))

        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr, beta1=0.5)

    def viz(self, name):
        im = tf.concat([self.im, tf.image.grayscale_to_rgb(self.mask)]+self.genim_list + self.genim_list_z2, axis=2)
        im = (im + 1.0) * 0.5  # 127.5
        # im = tf.image.hsv_to_rgb(im) * 255
        im = im * 255
        # im = tf.concat([im, tf.clip_by_value(im, 0, 255)], axis=1)
        im = tf.cast(im, tf.uint8)
        return tf.summary.image(name, im, max_outputs=50)

    def compute_losses(self, gpu):

        with tf.name_scope('losses'):
            with tf.name_scope('g_loss'):
                # l2_loss = tf.reduce_mean(tf.square(self.fg_params)) + tf.reduce_mean(tf.square(self.bg_params),
                # name='l2_loss')

                numelmask = tf.reduce_sum(self.bin_mask, axis=[1, 2, 3])
                numelall = tf.ones_like(numelmask) * tf.size(self.bin_mask[0], out_type=tf.float32)
                numelmask = tf.where(tf.equal(numelmask, 0), numelall, numelmask)
                weight_recon_loss = numelall / numelmask

                saldiff = tf.reduce_mean((tf.reduce_mean(self.bin_mask * self.sal_final, axis=[1, 2, 3]) -
                                          tf.reduce_mean(self.bin_mask * self.sal_init,
                                                         axis=[1, 2, 3])) * weight_recon_loss)

                sal_loss = -saldiff

                g_gan_loss = []
                for k in range(len(self.prob_im_final_is_real)):
                    actualk = self.prob_im_final_is_real[k]
                    g_gan_loss += [hinge_lsgan_loss_generator(actualk[j]) for j in range(len(actualk))]

                g_gan_loss = tf.add_n(g_gan_loss) / float(self.args.n_scale)

                zrecon_loss = tf.reduce_mean(tf.abs(self.z1-self.zrecon))

                g_loss = g_gan_loss + self.LAMBDAsal * sal_loss + self.LAMBDA_r * zrecon_loss

            d_gan_loss = []
            with tf.name_scope('d_loss'):
                for k in range(len(self.prob_im_final_is_real)):
                    actualfk = self.prob_im_final_is_real[k]
                    d_gan_loss += [hinge_lsgan_loss_discriminator(self.prob_im_is_real[j], actualfk[j]) for j in
                                   range(len(actualfk))]

                d_gan_loss = tf.add_n(d_gan_loss) / float(self.args.n_scale)
                d_loss = d_gan_loss

        self.losses = {'gan_loss': g_gan_loss,
                       'sal_loss': self.LAMBDAsal * sal_loss,
                       'z_recon_loss': zrecon_loss,
                       'd_loss': d_loss}

        self.model_vars = tf.trainable_variables()

        if gpu == 0:
            for var in self.model_vars:
                print(var.name)

        self.d_vars = [var for var in self.model_vars if 'disc' in var.name]
        self.g_vars = [var for var in self.model_vars if 'gen' in var.name]
        with tf.variable_scope('ADAM_op', reuse=gpu > 0):
            Ggrads = self.optimizer.compute_gradients(g_loss, self.g_vars)
            Dgrads = self.optimizerD.compute_gradients(d_loss, self.d_vars)

        # Summary variables for tensorboard
        g_loss_summ = tf.summary.merge([tf.summary.scalar("loss_all", g_loss),
                                        tf.summary.scalar("g_gan_loss", g_gan_loss),
                                        tf.summary.scalar("zrecon_loss", zrecon_loss),
                                        tf.summary.scalar("sal_loss", self.LAMBDAsal * sal_loss),
                                        tf.summary.scalar("sal_diff", self.LAMBDAsal * saldiff),
                                        ])
        self.g_loss_summ = [g_loss_summ]
        self.g_loss_summ += [tf.summary.histogram(q + '_fg', v) for q, v in self.fg_params.items()]
        self.g_loss_summ += [tf.summary.histogram(q + '_bg', v) for q, v in self.bg_params.items()]
        self.d_loss_summ = [tf.summary.scalar("d_loss", d_loss)]

        return Ggrads, Dgrads

    def average_gradients(self, tower_grads):
        """Calculate the average gradient for each shared variable across all towers.
        Note that this function provides a synchronization point across all towers.
        Args:
          tower_grads: List of lists of (gradient, variable) tuples. The outer list
            is over individual gradients. The inner list is over the gradient
            calculation for each tower.
        Returns:
           List of pairs of (gradient, variable) where the gradient has been averaged
           across all towers.
        """
        average_grads = []
        for grad_and_vars in zip(*tower_grads):
            # Note that each grad_and_vars looks like the following:
            #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
            grads = []
            for g, var in grad_and_vars:
                # Add 0 dimension to the gradients to represent the tower.
                if g == None:
                    g = tf.zeros_like(var)

                expanded_g = tf.expand_dims(g, 0)

                # Append on a 'tower' dimension which we will average over below.
                grads.append(expanded_g)

            # Average over the 'tower' dimension.
            grad = tf.concat(axis=0, values=grads)
            grad = tf.reduce_mean(grad, 0)

            # Keep in mind that the Variables are redundant because they are shared
            # across towers. So .. we will just return the first tower's pointer to
            # the Variable.
            v = grad_and_vars[0][1]
            grad_and_var = (grad, v)
            average_grads.append(grad_and_var)
        return average_grads

    def train(self):
        """Training Function."""
        # Build the network
        G_tower_grad = []
        D_tower_grad = []
        for i in range(self.nb_gpus):
            print('Building graph for gpu nb %d /n' % i)
            with tf.device('/gpu:%d' % i):
                with tf.name_scope('Tower_%d' % (i)) as scope:
                    self.model_setup(i)
                    # Loss function calculations
                    gradG, gradD = self.compute_losses(i)
                    # print(gradG)
                    # print(gradD)
                    G_tower_grad.append(gradG)
                    D_tower_grad.append(gradD)
                    # tf.get_variable_scope().reuse_variables()

        averageGgrags = self.average_gradients(G_tower_grad)
        averageDgrags = self.average_gradients(D_tower_grad)
        with tf.variable_scope('ADAM_op'):
            self.g_trainer = self.optimizer.apply_gradients(averageGgrags)
            self.d_trainer = self.optimizerD.apply_gradients(averageDgrags)

        for grad, var in averageGgrags:
            if grad is not None:
                self.g_loss_summ.append(tf.summary.histogram(var.op.name + '/gradients', grad))

        for grad, var in averageDgrags:
            if grad is not None:
                self.d_loss_summ.append(tf.summary.histogram(var.op.name + '/gradients', grad))

        self.im_summ = self.viz('viz')

        init = tf.variables_initializer(self.g_vars + [self.global_step] + self.d_vars +
                                        tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='ADAM_op'))
        saver = tf.train.Saver(max_to_keep=2)
        # get_next_val = self.iterator_test.get_next()
        config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        # config.gpu_options.allow_growth = True
        with tf.keras.backend.get_session() as sess:
            sess.run(init)
            # self.salmodel.load_weights(os.path.join(os.path.dirname(self.salmodelpath), "mdsem_model_LSUN_weights.h5"))
            # Restore the model to run the model from last checkpoint
            if self.to_restore:
                chkpt_fname = tf.train.latest_checkpoint(self.checkpoint_dir)
                saver.restore(sess, chkpt_fname)

            writer = tf.summary.FileWriter(self.output_dir)
            writer.add_graph(sess.graph)

            if not os.path.exists(self.output_dir):
                os.makedirs(self.output_dir)

            start_decay = np.floor(self.nb_its * self.start_decay_p / 100.)
            sess.run(self.Giterator.initializer)
            for it in tqdm(range(sess.run(self.global_step), self.nb_its, self.nb_gpus * self.batch_size),
                           desc='iterations'):
                self.base_lr = self.args.lr
                self.base_lrd = self.args.lrd
                if np.mod(it, 5000) == 0:
                    sess.run(tf.assign(self.global_step, it))
                    saver.save(sess, os.path.join(self.output_dir, "saledit"), global_step=it)
                if it > start_decay:
                    self.base_lr -= ((float(it) - start_decay) / (self.nb_its - start_decay)) * self.base_lr
                    self.base_lrd -= ((float(it) - start_decay) / (self.nb_its - start_decay)) * self.base_lrd
                try:
                    _ = sess.run(self.g_trainer, feed_dict={self.lr: self.base_lr, self.isTrain: True})
                    _ = sess.run(self.d_trainer, feed_dict={self.lrd: self.base_lrd, self.isTrain: True})

                    if np.mod(it, self.print_freq) == 0:
                        im_summ, losses, summary_strG, summary_strD = sess.run(
                            [self.im_summ, self.losses, self.g_loss_summ, self.d_loss_summ],
                            feed_dict={self.lr: self.base_lr, self.lrd: self.base_lrd, self.isTrain: True})
                        logging = ['it%d, ' % it]
                        logging += ['lr: %.6f, ' % self.base_lr]
                        logging += ['lrd: %.6f, ' % self.base_lrd]
                        logging += [h + ': %.3f, ' % losses[h] for h in list(losses.keys())]
                        print(''.join(logging))
                        with open(os.path.join(self.output_dir, 'logs.txt'), "a") as log_file:
                            log_file.write('%s\n' % (''.join(logging)))
                        [writer.add_summary(summary_strG[j], self.num_fake_inputs)
                         for j in range(len(summary_strG))]
                        [writer.add_summary(summary_strD[j], self.num_fake_inputs)
                         for j in range(len(summary_strD))]
                        writer.add_summary(im_summ, it)
                    writer.flush()
                    self.num_fake_inputs += self.nb_gpus
                    # pbar.update(1)

                except tf.errors.OutOfRangeError:
                    sess.run(self.Giterator.initializer)
            sess.run(tf.assign(self.global_step, it))
            saver.save(sess, os.path.join(self.output_dir, "saledit"), global_step=it)
            # pbar.update(self.batch_size * self.nb_gpus)

        sess.close()
        tf.reset_default_graph()

    def resave(self):
        """Training Function."""
        # Build the network
        G_tower_grad = []
        D_tower_grad = []
        for i in range(self.nb_gpus):
            print('Building graph for gpu nb %d /n' % i)
            with tf.device('/gpu:%d' % i):
                with tf.name_scope('Tower_%d' % (i)) as scope:
                    self.model_setup(i)
                    # Loss function calculations
                    gradG, gradD = self.compute_losses(i)
                    # print(gradG)
                    # print(gradD)
                    G_tower_grad.append(gradG)
                    D_tower_grad.append(gradD)
                    # tf.get_variable_scope().reuse_variables()

        averageGgrags = self.average_gradients(G_tower_grad)
        averageDgrags = self.average_gradients(D_tower_grad)
        with tf.variable_scope('ADAM_op'):
            self.g_trainer = self.optimizer.apply_gradients(averageGgrags)
            self.d_trainer = self.optimizerD.apply_gradients(averageDgrags)

        for grad, var in averageGgrags:
            if grad is not None:
                self.g_loss_summ.append(tf.summary.histogram(var.op.name + '/gradients', grad))

        for grad, var in averageDgrags:
            if grad is not None:
                self.d_loss_summ.append(tf.summary.histogram(var.op.name + '/gradients', grad))

        self.im_summ = self.viz('viz')

        init = tf.variables_initializer(self.g_vars + [self.global_step] + self.d_vars +
                                        tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='ADAM_op'))
        saver = tf.train.Saver(max_to_keep=2)
        # get_next_val = self.iterator_test.get_next()
        config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        # config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            sess.run(init)
            self.salmodel.load_weights(os.path.join(os.path.dirname(self.salmodelpath), "mdsem_model_LSUN_weights.h5"))
            self.init_fn(sess)
            # Restore the model to run the model from last checkpoint
            if self.to_restore:
                chkpt_fname = tf.train.latest_checkpoint(self.checkpoint_dir)
                saver.restore(sess, chkpt_fname)
                new_saledit_path = os.path.join(os.path.dirname(self.output_dir), os.path.basename(self.checkpoint_dir))
                saver.save(sess, os.path.join(new_saledit_path, "new_saledit"),
                           global_step=sess.run(self.global_step))

            sess.close()
        tf.reset_default_graph()

    def test(self):
        """Training Function."""
        # Build the network
        self.model_setup_test_HR(0)
        # self.model_setup(0)
        init = tf.variables_initializer(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='ADAM'))
        saver = tf.train.Saver(max_to_keep=2)
        config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        # config.gpu_options.allow_growth = True
        with tf.keras.backend.get_session() as sess:
            sess.run(init)
            chkpt_fname = tf.train.latest_checkpoint(self.checkpoint_dir)
            saver.restore(sess, chkpt_fname)

            epoch_id = sess.run(self.global_step)

            imsaving_path = os.path.join(self.checkpoint_dir, 'results', 'epoch%d' % (epoch_id),
                                         'ims')
            # totar = os.path.join(self.checkpoint_dir, 'results')
            if not os.path.exists(imsaving_path):
                os.makedirs(imsaving_path)

            sess.run(self.Giterator.initializer)
            webpage = html.HTML(os.path.dirname(imsaving_path),
                                'Experiment = %s, Phase = %s, Epoch = %s' % ('exp', 'test', epoch_id))

            mean_inc = []
            fgparams_dict_all_inc = {key: [] for key in list(self.fgparams[0].keys())}
            bgparams_dict_all_inc = {key: [] for key in list(self.bgparams[0].keys())}

            try:
                with tqdm(total=int(self.max_images_G)) as pbar:
                    while True:
                        im, fake_ims_hr_inc, mask, mask_lr, salinit, salfinals_inc, fgparams_inc, bgparams_inc = sess.run(
                            [self.im_hr,
                             self.imfinals,
                             self.mask_hr,
                             self.bin_mask,
                             self.sal_init,
                             self.salfinals,
                             self.fgparams,
                             self.bgparams,
                             ], feed_dict={self.isTrain: False}
                        )
                        for key, value in fgparams_inc[0].items():
                            fgparams_dict_all_inc[key].append(fgparams_inc[0][key])
                        for key, value in bgparams_inc[0].items():
                            bgparams_dict_all_inc[key].append(bgparams_inc[0][key])

                        scaling_weights = (np.size(mask_lr[0, :, :, :])) / np.sum(mask_lr, axis=(1, 2, 3))
                        mean_sal_init = np.mean(salinit * mask_lr, axis=(1, 2, 3)) * scaling_weights
                        mean_salfinals_inc = [np.mean(salfinal * mask_lr, axis=(1, 2, 3)) * scaling_weights
                                              for salfinal in salfinals_inc]

                        mean_inc.append(mean_salfinals_inc[0] - mean_sal_init)
                        all_dict = collections.OrderedDict()
                        all_dict['im%d' % self.num_fake_inputs] = (im + 1) * 0.5
                        all_dict['mask%d.png' % self.num_fake_inputs] = mask

                        suffix = '%d_%.2f' % (self.num_fake_inputs, mean_sal_init)
                        all_dict_bis = collections.OrderedDict()
                        all_dict_bis['im%d' % self.num_fake_inputs] = (im + 1) * 0.5
                        all_dict_bis['salinit' + suffix + '.png'] = salinit
                        # fake_ims_hr = [apply_transformations_np((im+1)*0.5, mask, fgparams[k], bgparams[k]) for k in range(len(fgparams))]

                        for k in range(len(fake_ims_hr_inc)):
                            txt_params = ''
                            txt_params = print_params(fgparams_inc[k], txt_params)
                            txt_params = print_params(bgparams_inc[k], txt_params)
                            suffix = '%d_%.3f' % (self.num_fake_inputs, mean_salfinals_inc[k] - mean_sal_init)
                            suffixsal = '%d_%.3f.png' % (
                                self.num_fake_inputs, mean_salfinals_inc[k] - mean_sal_init) + 'separator' + txt_params
                            all_dict['imfinal_inc%d' % (k) + suffix] = fake_ims_hr_inc[k]
                            all_dict_bis['salfinal_inc%d' % (k) + suffixsal] = salfinals_inc[k]

                        saveImages_bis_bis(webpage, all_dict, imsaving_path)
                        saveImages_bis_bis(webpage, all_dict_bis, imsaving_path)
                        self.num_fake_inputs += 1
                        pbar.update(self.batch_size)

            except tf.errors.OutOfRangeError:
                pass
        webpage.save()
        sess.close()
        tf.reset_default_graph()
        f = open(os.path.join(os.path.dirname(imsaving_path), 'mean_sals.txt'), 'w')
        f.write('Mean saliency increase %.3f\n' % np.mean(mean_inc))
        f.write('===================================\n\n')
        stds_list_fg = []
        stds_list_bg = []
        for key, value in fgparams_dict_all_inc.items():
            if 'color' in key:
                stacked_color_fg_inc = np.concatenate(fgparams_dict_all_inc['color'], axis=0).squeeze()
                stacked_color_fg_inc = np.reshape(stacked_color_fg_inc, [-1, 24])
                f.write('FG %s : %.3f (%.3f)\n' % (
                    key, np.mean(stacked_color_fg_inc), np.std(stacked_color_fg_inc)))
                stds_list_fg.append(np.std(stacked_color_fg_inc))

                stacked_color_bg_inc = np.concatenate(bgparams_dict_all_inc['color'], axis=0).squeeze()
                stacked_color_bg_inc = np.reshape(stacked_color_bg_inc, [-1, 24])
                f.write('BG %s : %.3f (%.3f)\n' % (
                    key, np.mean(stacked_color_bg_inc), np.std(stacked_color_bg_inc)))
                stds_list_bg.append(np.std(stacked_color_bg_inc))

                fgparams_dict_all_inc['color'] = sum(fgparams_dict_all_inc['color']) / len(
                    fgparams_dict_all_inc['color'])
                bgparams_dict_all_inc['color'] = sum(bgparams_dict_all_inc['color']) / len(
                    bgparams_dict_all_inc['color'])
            elif 'tone' in key:
                stacked_tone_fg_inc = np.concatenate(fgparams_dict_all_inc['tone'], axis=0).squeeze()
                stacked_tone_fg_inc = np.reshape(stacked_tone_fg_inc, [-1, 8])
                f.write('FG %s : %.3f (%.3f)\n' % (
                    key, np.mean(stacked_tone_fg_inc), np.std(stacked_tone_fg_inc)))
                stds_list_fg.append(np.std(stacked_tone_fg_inc))

                stacked_tone_bg_inc = np.concatenate(bgparams_dict_all_inc['tone'], axis=0).squeeze()
                stacked_tone_bg_inc = np.reshape(stacked_tone_bg_inc, [-1, 8])
                f.write('BG %s : %.3f (%.3f)\n' % (
                    key, np.mean(stacked_tone_bg_inc), np.std(stacked_tone_bg_inc)))
                stds_list_bg.append(np.std(stacked_tone_bg_inc))

                fgparams_dict_all_inc['tone'] = sum(fgparams_dict_all_inc['tone']) / len(fgparams_dict_all_inc['tone'])
                bgparams_dict_all_inc['tone'] = sum(bgparams_dict_all_inc['tone']) / len(bgparams_dict_all_inc['tone'])
            else:
                stacked_fg_inc = np.stack(fgparams_dict_all_inc[key]).squeeze()
                f.write('FG %s : %.3f (%.3f)\n' % (key, np.mean(stacked_fg_inc), np.std(stacked_fg_inc)))
                stds_list_fg.append(np.std(stacked_fg_inc))
                stacked_bg_inc = np.stack(bgparams_dict_all_inc[key]).squeeze()
                f.write('BG %s : %.3f (%.3f)\n' % (key, np.mean(stacked_bg_inc), np.std(stacked_bg_inc)))
                stds_list_bg.append(np.std(stacked_bg_inc))

                fgparams_dict_all_inc[key] = sum(fgparams_dict_all_inc[key]) / len(fgparams_dict_all_inc[key])
                bgparams_dict_all_inc[key] = sum(bgparams_dict_all_inc[key]) / len(bgparams_dict_all_inc[key])

            f.write('===================\n')
        f.write('Mean FG std over all parameters: %.3f\n' % np.mean(stds_list_fg))
        f.write('Mean BG std over all parameters: %.3f\n' % np.mean(stds_list_bg))

        with open(os.path.join(os.path.dirname(imsaving_path), 'mean_fg_params.pickle'), 'wb') as handle:
            pickle.dump(fgparams_dict_all_inc, handle)
        with open(os.path.join(os.path.dirname(imsaving_path), 'mean_bg_params.pickle'), 'wb') as handle:
            pickle.dump(bgparams_dict_all_inc, handle)

