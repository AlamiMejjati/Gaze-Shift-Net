from layers import *
from operations import *


class increase_multistyle:
    """ Model for increasing saliency with multi-style,
    all parametric transformations are applied.
    The initial saliency is also taken as input .
    another input s is added, which indicated the sense of the translation"""

    def __init__(self, args, ksize=25):
        self.NF = args.ngf
        self.fc_dim = args.fc_dim
        self.donorm = args.donormG
        self.ksize = ksize
        f1 = (1 / 8) * np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        f2 = (1 / 8) * np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
        self.tf1 = tf.expand_dims(tf.stack([tf.constant(f1, dtype=tf.float32)] * 3, axis=-1), -1)
        self.tf2 = tf.expand_dims(tf.stack([tf.constant(f2, dtype=tf.float32)] * 3, axis=-1), -1)
        self.L = 8

    def get_outputs(self, inputs):
        im = inputs['im']
        m = inputs['masks']
        bin_mask = (m + 1) * 0.5
        bin_im = (im + 1) * 0.5
        z = inputs['z']

        msize = m.get_shape().as_list()
        z_ = tf.tile(z[:, None, None, :], [1, msize[1], msize[2], 1])
        mz = tf.concat([m,z_], axis=-1)
        x = tf.concat([im, mz], axis=-1)
        xall = self.get_sharedEnc(x, mz, name='SharedEnc')

        fg_params_inc = self.paramspitter(xall, 'inc_fg_param_spitter')
        bg_params_inc = self.paramspitter(xall, 'inc_bg_param_spitter')
        im_increase_list = self.apply_params_inc(bin_im, bin_mask, fg_params_inc, bg_params_inc)

        all_params_fg = tf.concat([fg_params_inc['sharp'], fg_params_inc['exposure'], fg_params_inc['contrast'],
                                   tf.layers.flatten(fg_params_inc['tone']),
                                   tf.layers.flatten(fg_params_inc['color'])], axis=-1)
        all_params_bg = tf.concat([bg_params_inc['sharp'], bg_params_inc['exposure'], bg_params_inc['contrast'],
                                   tf.layers.flatten(bg_params_inc['tone']),
                                   tf.layers.flatten(bg_params_inc['color'])], axis=-1)

        all_params = tf.concat([all_params_fg, all_params_bg], axis=-1)
        zrecon = self.styleEncoder(all_params, z.get_shape().as_list()[-1])

        return [im_increase_list[-1]], fg_params_inc, bg_params_inc, zrecon

    def get_outputs_hr(self, im, m, fg_params, bg_params):
        im_increase_list = self.apply_params_inc(im, m, fg_params, bg_params)
        return im_increase_list[-1]

    def apply_params_inc(self, bin_im, bin_mask, fg_params, bg_params):
        # Apply sharpening#
        with tf.name_scope('sharpening_station'):
            fg_im = apply_sharpening(bin_im, fg_params['sharp'], self.tf1, self.tf2, 1)
            bg_im = apply_sharpening(bin_im, bg_params['sharp'], self.tf1, self.tf2, 0)
            im_sharp = tf.identity(fg_im * bin_mask + bg_im * (1 - bin_mask), name='im_sharp')

        # Apply exposure #
        with tf.name_scope('exposure_station'):
            fg_im = apply_exposure(im_sharp, fg_params['exposure'])
            bg_im = apply_exposure(im_sharp, bg_params['exposure'])
            im_exp = tf.identity(fg_im * bin_mask + bg_im * (1 - bin_mask), name='im_exp')

        # Apply contrast #
        with tf.name_scope('contrast_station'):
            fg_im = apply_contrast(im_exp, fg_params['contrast'])
            bg_im = apply_contrast(im_exp, bg_params['contrast'])
            im_cont = tf.identity(fg_im * bin_mask + bg_im * (1 - bin_mask), name='im_cont')

        # Apply tone curve adjustement#
        with tf.name_scope('tone_adjustment_station'):
            fg_im = apply_tone_curve_adjustment(im_cont, fg_params['tone'])
            bg_im = apply_tone_curve_adjustment(im_cont, bg_params['tone'])
            im_tone = tf.identity(fg_im * bin_mask + bg_im * (1 - bin_mask), name='im_tone')

        # Apply color curve adjustement#
        with tf.name_scope('color_adjustment_station'):
            fg_im = apply_color_curve_adjustment(im_tone, fg_params['color'])
            bg_im = apply_color_curve_adjustment(im_tone, bg_params['color'])
            im_color = tf.identity(fg_im * bin_mask + bg_im * (1 - bin_mask), name='im_color')

        # # Apply blur
        # with tf.name_scope('blur_station'):
        #     bg_im = apply_blur(im_color, bg_params['blur'], self.ksize)
        #     im_blur = tf.identity(im_color * bin_mask + bg_im * (1 - bin_mask), name='im_blur')

        im_list = [im_sharp, im_exp, im_cont, im_tone, im_color]
        # im_list = [im_gamma, im_sharp, im_wb]
        im_list = [(k - 0.5) * 2 for k in im_list]

        return im_list

    def apply_params_dec(self, bin_im, bin_mask, fg_params, bg_params):
        # Apply Gamma#

        fg_im = apply_gamma(bin_im, fg_params['gamma'])
        bg_im = apply_gamma(bin_im, bg_params['gamma'])
        im_gamma = tf.identity(fg_im * bin_mask + bg_im * (1 - bin_mask), name='im_gamma')

        # Apply sharpening#

        fg_im = apply_sharpening_fg(im_gamma, fg_params['sharp'], self.tf1, self.tf2, 1)
        bg_im = apply_sharpening_bg(im_gamma, bg_params['sharp'], self.tf1, self.tf2, 0)
        im_sharp = tf.identity(fg_im * bin_mask + bg_im * (1 - bin_mask), name='im_sharp')

        # Apply WB#

        fg_im = apply_WB(im_sharp, fg_params['WB'])
        bg_im = apply_WB(im_sharp, bg_params['WB'])
        im_wb = tf.identity(fg_im * bin_mask + bg_im * (1 - bin_mask), name='im_wb')

        # Apply exposure #

        fg_im = apply_exposure(im_wb, fg_params['exposure'])
        bg_im = apply_exposure(im_wb, bg_params['exposure'])
        im_exp = tf.identity(fg_im * bin_mask + bg_im * (1 - bin_mask), name='im_exp')

        # Apply contrast #

        fg_im = apply_contrast(im_exp, fg_params['contrast'])
        bg_im = apply_contrast(im_exp, bg_params['contrast'])
        im_cont = tf.identity(fg_im * bin_mask + bg_im * (1 - bin_mask), name='im_cont')

        # Apply saturation #

        fg_im = apply_saturation(im_cont, fg_params['saturation'])
        bg_im = apply_saturation(im_cont, bg_params['saturation'])
        im_sat = tf.identity(fg_im * bin_mask + bg_im * (1 - bin_mask), name='im_sat')

        # Apply tone curve adjustement#

        fg_im = apply_tone_curve_adjustment(im_sat, fg_params['tone'])
        bg_im = apply_tone_curve_adjustment(im_sat, bg_params['tone'])
        im_tone = tf.identity(fg_im * bin_mask + bg_im * (1 - bin_mask), name='im_tone')

        # Apply color curve adjustement#

        fg_im = apply_color_curve_adjustment(im_tone, fg_params['color'])
        bg_im = apply_color_curve_adjustment(im_tone, bg_params['color'])
        im_color = tf.identity(fg_im * bin_mask + bg_im * (1 - bin_mask), name='im_color')

        im_list = [im_gamma, im_sharp, im_wb, im_exp, im_cont, im_sat, im_tone, im_color]
        im_list = [(k - 0.5) * 2 for k in im_list]

        return im_list

    def get_sharedEnc(self, x, sm, name='SharedEnc'):
        with tf.variable_scope(name):
            x = general_conv2d(x, nf=self.NF, ks=7, s=2, name='c1', do_norm=self.donorm, padding='SAME')
            x = tf.concat([x, down_sample_avg(sm)], axis=-1)
            x = general_conv2d(x, nf=self.NF * 2, ks=3, s=2, name='c2', do_norm=self.donorm, padding='SAME')
            x = tf.concat([x, down_sample_avg(sm, 4)], axis=-1)
            x = general_conv2d(x, nf=self.NF * 4, ks=3, s=2, name='c3', do_norm=self.donorm, padding='SAME')
            x = tf.concat([x, down_sample_avg(sm, 8)], axis=-1)
            x = general_conv2d(x, nf=self.NF * 8, ks=3, s=2, name='c4', do_norm=self.donorm, padding='SAME')
            x = tf.concat([x, down_sample_avg(sm, 16)], axis=-1)
            x = general_conv2d(x, nf=self.NF * 16, ks=3, s=2, name='c5', do_norm=False, padding='SAME')
            x = tf.reduce_mean(x, [1, 2])
        return x

    def build_resblock(self, x, chs, name='resblock'):
        with tf.variable_scope(name):
            input = x
            x = general_conv2d(x, nf=chs, ks=3, s=1, name='c1', do_norm=self.donorm, padding='same')
            x = general_conv2d(x, nf=chs, ks=3, s=1, name='c2', do_norm=self.donorm, padding='same')
            return x + input

    def paramspitter_shared(self, x, size_rep, name='paramspitter_shared'):
        with tf.variable_scope(name):
            x = tf.layers.dense(x, size_rep * 2, activation=tf.nn.leaky_relu, name='dense0')
            x = tf.layers.dense(x, size_rep * 2, activation=tf.nn.leaky_relu, name='dense1')
            x = tf.layers.dense(x, size_rep * 2, activation=tf.nn.leaky_relu, name='dense2')
            x = tf.layers.dense(x, size_rep * 2, activation=tf.nn.leaky_relu, name='dense3')

        return x

    def MLP(self, x, chs, name='MLP'):
        with tf.variable_scope(name):
            x = tf.layers.dense(x, chs, activation=tf.nn.leaky_relu, name='fc0')
            x = tf.layers.dense(x, chs, activation=tf.nn.leaky_relu, name='fc1')
            x = tf.layers.dense(x, chs, activation=tf.nn.leaky_relu, name='fc2')
            mu = tf.layers.dense(x, chs, activation=None, name='fcmu')
            sigma = tf.layers.dense(x, chs, activation=None, name='fcsigma')
        return mu, sigma

    def styleEncoder(self, x, chs, name='styleEncoder'):
        with tf.variable_scope(name):
            x = tf.layers.dense(x, self.fc_dim * 2, activation=tf.nn.leaky_relu, name='fc0')
            x = tf.layers.dense(x, self.fc_dim * 2, activation=tf.nn.leaky_relu, name='fc1')
            x = tf.layers.dense(x, self.fc_dim * 2, activation=tf.nn.leaky_relu, name='fc2')
            x = tf.layers.dense(x, self.fc_dim * 2, activation=tf.nn.leaky_relu, name='fc3')
            x = tf.layers.dense(x, chs, activation=None, name='fcrecon')
        return x

    def paramspitter(self, x, name='paramspitter'):
        with tf.variable_scope(name):
            # x = tf.reduce_mean(x, axis=[1, 2], name='avg_pooling')
            # x = tf.layers.flatten(x)
            x = tf.layers.dense(x, self.fc_dim, activation=tf.nn.leaky_relu, name='dense0')
            x = tf.layers.dense(x, self.fc_dim, activation=tf.nn.leaky_relu, name='dense1')

            # gammaparam = 0*tf.layers.dense(x, 1, activation=tf.nn.sigmoid, name='gammaprediction') + 1
            # blurparam = 0*tf.layers.dense(x, 1, activation=tf.nn.sigmoid, name='blurprediction')
            sharpparam = 2 * tf.layers.dense(x, 1, activation=tf.nn.tanh, name='sharpprediction')
            # wbparam = 0*tf.layers.dense(x, 1, activation=tf.nn.sigmoid, name='wbprediction')
            exposureparam = 3 * tf.layers.dense(x, 1, activation=tf.nn.tanh, name='exposureprediction')
            # satparam = 0*tf.layers.dense(x, 1, activation=tf.nn.tanh, name='satprediction')
            contparam = tf.layers.dense(x, 1, activation=tf.nn.tanh, name='contprediction')
            toneparam = tf.layers.dense(x, self.L, activation=tf.nn.sigmoid, name='toneprediction') * 3
            colorparam = tf.layers.dense(x, self.L * 3, activation=tf.nn.sigmoid, name='colorprediction') * 3
            colorparam = tf.reshape(colorparam[:, None, :, None], [-1, 1, self.L, 3])
            toneparam = toneparam[:, None, :, None]

            # dict_params = {'gamma': gammaparam, 'sharp': sharpparam, 'WB': wbparam, 'exposure': exposureparam,
            #              'contrast': contparam, 'saturation':satparam, 'tone':toneparam, 'color':colorparam,
            #              'blur':blurparam}
            dict_params = {'sharp': sharpparam, 'exposure': exposureparam,
                           'contrast': contparam, 'tone': toneparam, 'color': colorparam}
        return dict_params

class increase:
    """ Model for increasing saliency with multi-style,
    all parametric transformations are applied.
    The initial saliency is also taken as input .
    another input s is added, which indicated the sense of the translation"""

    def __init__(self, args, ksize=25):
        self.NF = args.ngf
        self.fc_dim = args.fc_dim
        self.donorm = args.donormG
        self.ksize = ksize
        f1 = (1 / 8) * np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        f2 = (1 / 8) * np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
        self.tf1 = tf.expand_dims(tf.stack([tf.constant(f1, dtype=tf.float32)] * 3, axis=-1), -1)
        self.tf2 = tf.expand_dims(tf.stack([tf.constant(f2, dtype=tf.float32)] * 3, axis=-1), -1)
        self.L = 8

    def get_outputs(self, inputs):
        im = inputs['im']
        m = inputs['masks']
        bin_mask = (m + 1) * 0.5
        bin_im = (im + 1) * 0.5
        x = tf.concat([im, m], axis=-1)
        xall = self.get_sharedEnc(x, m, name='SharedEnc')
        fg_params_inc = self.paramspitter(xall, 'inc_fg_param_spitter')
        bg_params_inc = self.paramspitter(xall, 'inc_bg_param_spitter')
        im_increase_list = self.apply_params_inc(bin_im, bin_mask, fg_params_inc, bg_params_inc)

        return [im_increase_list[-1]], fg_params_inc, bg_params_inc

    def get_outputs_hr(self, im, m, fg_params, bg_params):
        im_increase_list = self.apply_params_inc(im, m, fg_params, bg_params)
        return im_increase_list[-1]

    def apply_params_inc(self, bin_im, bin_mask, fg_params, bg_params):

        # Apply sharpening#
        with tf.name_scope('sharpening_station'):
            fg_im = apply_sharpening(bin_im, fg_params['sharp'], self.tf1, self.tf2, 1)
            bg_im = apply_sharpening(bin_im, bg_params['sharp'], self.tf1, self.tf2, 0)
            im_sharp = tf.identity(fg_im * bin_mask + bg_im * (1 - bin_mask), name='im_sharp')

        # Apply exposure #
        with tf.name_scope('exposure_station'):
            fg_im = apply_exposure(im_sharp, fg_params['exposure'])
            bg_im = apply_exposure(im_sharp, bg_params['exposure'])
            im_exp = tf.identity(fg_im * bin_mask + bg_im * (1 - bin_mask), name='im_exp')

        # Apply contrast #
        with tf.name_scope('contrast_station'):
            fg_im = apply_contrast(im_exp, fg_params['contrast'])
            bg_im = apply_contrast(im_exp, bg_params['contrast'])
            im_cont = tf.identity(fg_im * bin_mask + bg_im * (1 - bin_mask), name='im_cont')


        # Apply tone curve adjustement#
        with tf.name_scope('tone_adjustment_station'):
            fg_im = apply_tone_curve_adjustment(im_cont, fg_params['tone'])
            bg_im = apply_tone_curve_adjustment(im_cont, bg_params['tone'])
            im_tone = tf.identity(fg_im * bin_mask + bg_im * (1 - bin_mask), name='im_tone')

        # Apply color curve adjustement#
        with tf.name_scope('color_adjustment_station'):
            fg_im = apply_color_curve_adjustment(im_tone, fg_params['color'])
            bg_im = apply_color_curve_adjustment(im_tone, bg_params['color'])
            im_color = tf.identity(fg_im * bin_mask + bg_im * (1 - bin_mask), name='im_color')

        # # Apply blur
        # with tf.name_scope('blur_station'):
        #     bg_im = apply_blur(im_color, bg_params['blur'], self.ksize)
        #     im_blur = tf.identity(im_color * bin_mask + bg_im * (1 - bin_mask), name='im_blur')

        im_list = [im_sharp, im_exp, im_cont, im_tone, im_color]
        # im_list = [im_gamma, im_sharp, im_wb]
        im_list = [(k-0.5)*2 for k in im_list]

        return im_list

    def apply_params_dec(self, bin_im, bin_mask, fg_params, bg_params):
        # Apply Gamma#

        fg_im = apply_gamma(bin_im, fg_params['gamma'])
        bg_im = apply_gamma(bin_im, bg_params['gamma'])
        im_gamma = tf.identity(fg_im * bin_mask + bg_im * (1 - bin_mask), name='im_gamma')

        # Apply sharpening#

        fg_im = apply_sharpening_fg(im_gamma, fg_params['sharp'], self.tf1, self.tf2, 1)
        bg_im = apply_sharpening_bg(im_gamma, bg_params['sharp'], self.tf1, self.tf2, 0)
        im_sharp = tf.identity(fg_im * bin_mask + bg_im * (1 - bin_mask), name='im_sharp')

        # Apply WB#

        fg_im = apply_WB(im_sharp, fg_params['WB'])
        bg_im = apply_WB(im_sharp, bg_params['WB'])
        im_wb = tf.identity(fg_im * bin_mask + bg_im * (1 - bin_mask), name='im_wb')

        # Apply exposure #

        fg_im = apply_exposure(im_wb, fg_params['exposure'])
        bg_im = apply_exposure(im_wb, bg_params['exposure'])
        im_exp = tf.identity(fg_im * bin_mask + bg_im * (1 - bin_mask), name='im_exp')

        # Apply contrast #

        fg_im = apply_contrast(im_exp, fg_params['contrast'])
        bg_im = apply_contrast(im_exp, bg_params['contrast'])
        im_cont = tf.identity(fg_im * bin_mask + bg_im * (1 - bin_mask), name='im_cont')

        # Apply saturation #

        fg_im = apply_saturation(im_cont, fg_params['saturation'])
        bg_im = apply_saturation(im_cont, bg_params['saturation'])
        im_sat = tf.identity(fg_im * bin_mask + bg_im * (1 - bin_mask), name='im_sat')

        # Apply tone curve adjustement#

        fg_im = apply_tone_curve_adjustment(im_sat, fg_params['tone'])
        bg_im = apply_tone_curve_adjustment(im_sat, bg_params['tone'])
        im_tone = tf.identity(fg_im * bin_mask + bg_im * (1 - bin_mask), name='im_tone')

        # Apply color curve adjustement#

        fg_im = apply_color_curve_adjustment(im_tone, fg_params['color'])
        bg_im = apply_color_curve_adjustment(im_tone, bg_params['color'])
        im_color = tf.identity(fg_im * bin_mask + bg_im * (1 - bin_mask), name='im_color')


        im_list = [im_gamma, im_sharp, im_wb, im_exp, im_cont, im_sat, im_tone, im_color]
        im_list = [(k-0.5)*2 for k in im_list]

        return im_list

    def get_sharedEnc(self, x, sm, name='SharedEnc'):
        with tf.variable_scope(name):
            x = general_conv2d(x, nf=self.NF, ks=7, s=2, name='c1', do_norm=self.donorm, padding='SAME')
            x = tf.concat([x, down_sample_avg(sm)], axis=-1)
            x = general_conv2d(x, nf=self.NF * 2, ks=3, s=2, name='c2', do_norm=self.donorm, padding='SAME')
            x = tf.concat([x, down_sample_avg(sm,4)], axis=-1)
            x = general_conv2d(x, nf=self.NF * 4, ks=3, s=2, name='c3', do_norm=self.donorm, padding='SAME')
            x = tf.concat([x, down_sample_avg(sm, 8)], axis=-1)
            x = general_conv2d(x, nf=self.NF * 8, ks=3, s=2, name='c4', do_norm=self.donorm, padding='SAME')
            x = tf.concat([x, down_sample_avg(sm, 16)], axis=-1)
            x = general_conv2d(x, nf=self.NF * 16, ks=3, s=2, name='c5', do_norm=False, padding='SAME')
            x = tf.reduce_mean(x, [1,2])
        return x

    def build_resblock(self, x, chs, name='resblock'):
        with tf.variable_scope(name):
            input = x
            x = general_conv2d(x, nf=chs, ks=3, s=1, name='c1', do_norm=self.donorm, padding='same')
            x = general_conv2d(x, nf=chs, ks=3, s=1, name='c2', do_norm=self.donorm, padding='same')
            return x+input

    def paramspitter_shared(self, x, size_rep, name='paramspitter_shared'):
        with tf.variable_scope(name):

            x = tf.layers.dense(x, size_rep*2, activation=tf.nn.leaky_relu, name='dense0')
            x = tf.layers.dense(x, size_rep*2, activation=tf.nn.leaky_relu, name='dense1')
            x = tf.layers.dense(x, size_rep*2, activation=tf.nn.leaky_relu, name='dense2')
            x = tf.layers.dense(x, size_rep*2, activation=tf.nn.leaky_relu, name='dense3')

        return x

    def MLP(self, x, chs, name='MLP'):
        with tf.variable_scope(name):
            x = tf.layers.dense(x, self.fc_dim, activation=tf.nn.leaky_relu, name='fc0')
            x = tf.layers.dense(x, self.fc_dim, activation=tf.nn.leaky_relu, name='fc1')
            x = tf.layers.dense(x, self.fc_dim, activation=tf.nn.leaky_relu, name='fc2')
            x = tf.layers.dense(x, self.fc_dim, activation=tf.nn.leaky_relu, name='fc3')
            mu = tf.layers.dense(x, chs, activation=None, name='fcmu')
            sigma = tf.layers.dense(x, chs, activation=None, name='fcsigma')
        return mu, sigma

    def styleEncoder(self, x, chs, name='styleEncoder', do_norm=False):
        with tf.variable_scope(name):
            x = tf.layers.dense(x, self.fc_dim, activation=tf.nn.leaky_relu, name='fc0')
            x = tf.layers.dense(x, self.fc_dim, activation=tf.nn.leaky_relu, name='fc1')
            x = tf.layers.dense(x, self.fc_dim, activation=tf.nn.leaky_relu, name='fc2')
            x = tf.layers.dense(x, self.fc_dim, activation=tf.nn.leaky_relu, name='fc3')
            x = tf.layers.dense(x, chs, activation=None, name='fcrecon')
        return x

    def paramspitter(self, x, name='paramspitter'):
        with tf.variable_scope(name):
            # x = tf.reduce_mean(x, axis=[1, 2], name='avg_pooling')
            # x = tf.layers.flatten(x)
            x = tf.layers.dense(x, self.fc_dim, activation=tf.nn.leaky_relu, name='dense0')
            x = tf.layers.dense(x, self.fc_dim, activation=tf.nn.leaky_relu, name='dense1')

            # gammaparam = 0*tf.layers.dense(x, 1, activation=tf.nn.sigmoid, name='gammaprediction') + 1
            # blurparam = 0*tf.layers.dense(x, 1, activation=tf.nn.sigmoid, name='blurprediction')
            sharpparam = 2*tf.layers.dense(x, 1, activation=tf.nn.tanh, name='sharpprediction')
            # wbparam = 0*tf.layers.dense(x, 1, activation=tf.nn.sigmoid, name='wbprediction')
            exposureparam = 3*tf.layers.dense(x, 1, activation=tf.nn.tanh, name='exposureprediction')
            # satparam = 0*tf.layers.dense(x, 1, activation=tf.nn.tanh, name='satprediction')
            contparam = tf.layers.dense(x, 1, activation=tf.nn.tanh, name='contprediction')
            toneparam = tf.layers.dense(x, self.L, activation=tf.nn.sigmoid, name='toneprediction')*3
            colorparam = tf.layers.dense(x, self.L*3, activation=tf.nn.sigmoid, name='colorprediction')*3
            colorparam = tf.reshape(colorparam[:,None,:,None], [-1, 1, self.L, 3])
            toneparam = toneparam[:, None, :, None]

            # dict_params = {'gamma': gammaparam, 'sharp': sharpparam, 'WB': wbparam, 'exposure': exposureparam,
            #              'contrast': contparam, 'saturation':satparam, 'tone':toneparam, 'color':colorparam,
            #              'blur':blurparam}
            dict_params = {'sharp': sharpparam, 'exposure': exposureparam,
                         'contrast': contparam, 'tone':toneparam, 'color':colorparam}
        return dict_params

class incdec:
    """ Model for increasing saliency with multi-style,
    all parametric transformations are applied.
    The initial saliency is also taken as input .
    another input s is added, which indicated the sense of the translation"""

    def __init__(self, args, ksize=25):
        self.NF = args.ngf
        self.fc_dim = args.fc_dim
        self.donorm = args.donormG
        self.ksize = ksize
        f1 = (1 / 8) * np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        f2 = (1 / 8) * np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
        self.tf1 = tf.expand_dims(tf.stack([tf.constant(f1, dtype=tf.float32)] * 3, axis=-1), -1)
        self.tf2 = tf.expand_dims(tf.stack([tf.constant(f2, dtype=tf.float32)] * 3, axis=-1), -1)
        self.L = 8

    def get_outputs(self, inputs):
        im = inputs['im']
        m = inputs['masks']
        bin_mask = (m + 1) * 0.5
        bin_im = (im + 1) * 0.5
        x = tf.concat([im, m], axis=-1)
        xall = self.get_sharedEnc(x, m, name='SharedEnc')
        fg_params_inc = self.paramspitter(xall, 'inc_fg_param_spitter')
        bg_params_inc = self.paramspitter(xall, 'inc_bg_param_spitter')
        fg_params_dec = self.paramspitter(xall, 'dec_fg_param_spitter')
        bg_params_dec = self.paramspitter(xall, 'dec_bg_param_spitter')

        im_increase_list_inc = self.apply_params(bin_im, bin_mask, fg_params_inc, bg_params_inc)
        im_increase_list_dec = self.apply_params(bin_im, bin_mask, fg_params_dec, bg_params_dec)

        return [im_increase_list_inc[-1], im_increase_list_dec[-1]], fg_params_inc, bg_params_inc, fg_params_dec, bg_params_dec

    def get_outputs_hr(self, im, m, fg_params, bg_params):
        im_increase_list = self.apply_params(im, m, fg_params, bg_params)
        return im_increase_list[-1]

    def apply_params(self, bin_im, bin_mask, fg_params, bg_params):

        # Apply sharpening#
        with tf.name_scope('sharpening_station'):
            fg_im = apply_sharpening(bin_im, fg_params['sharp'], self.tf1, self.tf2, 1)
            bg_im = apply_sharpening(bin_im, bg_params['sharp'], self.tf1, self.tf2, 0)
            im_sharp = tf.identity(fg_im * bin_mask + bg_im * (1 - bin_mask), name='im_sharp')

        # Apply exposure #
        with tf.name_scope('exposure_station'):
            fg_im = apply_exposure(im_sharp, fg_params['exposure'])
            bg_im = apply_exposure(im_sharp, bg_params['exposure'])
            im_exp = tf.identity(fg_im * bin_mask + bg_im * (1 - bin_mask), name='im_exp')

        # Apply contrast #
        with tf.name_scope('contrast_station'):
            fg_im = apply_contrast(im_exp, fg_params['contrast'])
            bg_im = apply_contrast(im_exp, bg_params['contrast'])
            im_cont = tf.identity(fg_im * bin_mask + bg_im * (1 - bin_mask), name='im_cont')


        # Apply tone curve adjustement#
        with tf.name_scope('tone_adjustment_station'):
            fg_im = apply_tone_curve_adjustment(im_cont, fg_params['tone'])
            bg_im = apply_tone_curve_adjustment(im_cont, bg_params['tone'])
            im_tone = tf.identity(fg_im * bin_mask + bg_im * (1 - bin_mask), name='im_tone')

        # Apply color curve adjustement#
        with tf.name_scope('color_adjustment_station'):
            fg_im = apply_color_curve_adjustment(im_tone, fg_params['color'])
            bg_im = apply_color_curve_adjustment(im_tone, bg_params['color'])
            im_color = tf.identity(fg_im * bin_mask + bg_im * (1 - bin_mask), name='im_color')

        # # Apply blur
        # with tf.name_scope('blur_station'):
        #     bg_im = apply_blur(im_color, bg_params['blur'], self.ksize)
        #     im_blur = tf.identity(im_color * bin_mask + bg_im * (1 - bin_mask), name='im_blur')

        im_list = [im_sharp, im_exp, im_cont, im_tone, im_color]
        # im_list = [im_gamma, im_sharp, im_wb]
        im_list = [(k-0.5)*2 for k in im_list]

        return im_list

    def get_sharedEnc(self, x, sm, name='SharedEnc'):
        with tf.variable_scope(name):
            x = general_conv2d(x, nf=self.NF, ks=7, s=2, name='c1', do_norm=self.donorm, padding='SAME')
            x = tf.concat([x, down_sample_avg(sm)], axis=-1)
            x = general_conv2d(x, nf=self.NF * 2, ks=3, s=2, name='c2', do_norm=self.donorm, padding='SAME')
            x = tf.concat([x, down_sample_avg(sm,4)], axis=-1)
            x = general_conv2d(x, nf=self.NF * 4, ks=3, s=2, name='c3', do_norm=self.donorm, padding='SAME')
            x = tf.concat([x, down_sample_avg(sm, 8)], axis=-1)
            x = general_conv2d(x, nf=self.NF * 8, ks=3, s=2, name='c4', do_norm=self.donorm, padding='SAME')
            x = tf.concat([x, down_sample_avg(sm, 16)], axis=-1)
            x = general_conv2d(x, nf=self.NF * 16, ks=3, s=2, name='c5', do_norm=False, padding='SAME')
            x = tf.reduce_mean(x, [1,2])
        return x

    def build_resblock(self, x, chs, name='resblock'):
        with tf.variable_scope(name):
            input = x
            x = general_conv2d(x, nf=chs, ks=3, s=1, name='c1', do_norm=self.donorm, padding='same')
            x = general_conv2d(x, nf=chs, ks=3, s=1, name='c2', do_norm=self.donorm, padding='same')
            return x+input

    def paramspitter_shared(self, x, size_rep, name='paramspitter_shared'):
        with tf.variable_scope(name):

            x = tf.layers.dense(x, size_rep*2, activation=tf.nn.leaky_relu, name='dense0')
            x = tf.layers.dense(x, size_rep*2, activation=tf.nn.leaky_relu, name='dense1')
            x = tf.layers.dense(x, size_rep*2, activation=tf.nn.leaky_relu, name='dense2')
            x = tf.layers.dense(x, size_rep*2, activation=tf.nn.leaky_relu, name='dense3')

        return x

    def MLP(self, x, chs, name='MLP'):
        with tf.variable_scope(name):
            x = tf.layers.dense(x, self.fc_dim, activation=tf.nn.leaky_relu, name='fc0')
            x = tf.layers.dense(x, self.fc_dim, activation=tf.nn.leaky_relu, name='fc1')
            x = tf.layers.dense(x, self.fc_dim, activation=tf.nn.leaky_relu, name='fc2')
            x = tf.layers.dense(x, self.fc_dim, activation=tf.nn.leaky_relu, name='fc3')
            mu = tf.layers.dense(x, chs, activation=None, name='fcmu')
            sigma = tf.layers.dense(x, chs, activation=None, name='fcsigma')
        return mu, sigma

    def styleEncoder(self, x, chs, name='styleEncoder', do_norm=False):
        with tf.variable_scope(name):
            x = tf.layers.dense(x, self.fc_dim, activation=tf.nn.leaky_relu, name='fc0')
            x = tf.layers.dense(x, self.fc_dim, activation=tf.nn.leaky_relu, name='fc1')
            x = tf.layers.dense(x, self.fc_dim, activation=tf.nn.leaky_relu, name='fc2')
            x = tf.layers.dense(x, self.fc_dim, activation=tf.nn.leaky_relu, name='fc3')
            x = tf.layers.dense(x, chs, activation=None, name='fcrecon')
        return x

    def paramspitter(self, x, name='paramspitter'):
        with tf.variable_scope(name):
            # x = tf.reduce_mean(x, axis=[1, 2], name='avg_pooling')
            # x = tf.layers.flatten(x)
            x = tf.layers.dense(x, self.fc_dim, activation=tf.nn.leaky_relu, name='dense0')
            x = tf.layers.dense(x, self.fc_dim, activation=tf.nn.leaky_relu, name='dense1')

            # gammaparam = 0*tf.layers.dense(x, 1, activation=tf.nn.sigmoid, name='gammaprediction') + 1
            # blurparam = 0*tf.layers.dense(x, 1, activation=tf.nn.sigmoid, name='blurprediction')
            sharpparam = 2*tf.layers.dense(x, 1, activation=tf.nn.tanh, name='sharpprediction')
            # wbparam = 0*tf.layers.dense(x, 1, activation=tf.nn.sigmoid, name='wbprediction')
            exposureparam = 3*tf.layers.dense(x, 1, activation=tf.nn.tanh, name='exposureprediction')
            # satparam = 0*tf.layers.dense(x, 1, activation=tf.nn.tanh, name='satprediction')
            contparam = tf.layers.dense(x, 1, activation=tf.nn.tanh, name='contprediction')
            toneparam = tf.layers.dense(x, self.L, activation=tf.nn.sigmoid, name='toneprediction')*3
            colorparam = tf.layers.dense(x, self.L*3, activation=tf.nn.sigmoid, name='colorprediction')*3
            colorparam = tf.reshape(colorparam[:,None,:,None], [-1, 1, self.L, 3])
            toneparam = toneparam[:, None, :, None]

            # dict_params = {'gamma': gammaparam, 'sharp': sharpparam, 'WB': wbparam, 'exposure': exposureparam,
            #              'contrast': contparam, 'saturation':satparam, 'tone':toneparam, 'color':colorparam,
            #              'blur':blurparam}
            dict_params = {'sharp': sharpparam, 'exposure': exposureparam,
                         'contrast': contparam, 'tone':toneparam, 'color':colorparam}
        return dict_params

class MSD_global:

    def __init__(self, args):
        self.NF = args.ndf
        self.n_scale = args.n_scale
        self.n_dis = 4
        self.do_norm = args.donormD
    def get_outputs(self, im, im_final):
        prob_im_is_real, feats_real = self.get_disc(im, reuse=False)
        if isinstance(im_final, list):
            prob_im_final_is_real = []
            feats_fake = []
            for j in im_final:
                pf_is_real, f_fake = self.get_disc(j, reuse=True)
                prob_im_final_is_real.append(pf_is_real)
                feats_fake.append(f_fake)
        else:
            prob_im_final_is_real, feats_fake = self.get_disc(im_final, reuse=True)
        return prob_im_is_real, feats_real, prob_im_final_is_real, feats_fake

    def get_disc(self, x_init, reuse, name="disc"):
        D_logit = []
        features = []
        with tf.variable_scope(name, reuse=reuse):
            for scale in range(self.n_scale):
                channel = self.NF
                x = conv(x_init, channel, kernel=4, stride=2, pad=1, pad_type='reflect',
                         scope='ms_' + str(scale) + 'conv_0')
                x = tf.nn.leaky_relu(x, 0.2)
                features.append(x)
                if self.do_norm:
                    x = instance_norm(x, name='ms_' + str(scale)+'IN_0')
                for i in range(1, self.n_dis):
                    x = conv(x, channel * 2, kernel=4, stride=2, pad=1, pad_type='reflect',
                             scope='ms_' + str(scale) + 'conv_' + str(i))
                    x = tf.nn.leaky_relu(x, 0.2)
                    features.append(x)
                    if self.do_norm:
                        x = instance_norm(x, name='ms_' + str(scale)+'IN_%d'%i)
                    if channel < 256:
                        channel = channel * 2

                x = tf.layers.flatten(x, name='flatten')
                x = tf.layers.dense(x, 128, activation=tf.nn.leaky_relu, name='ms_' + str(scale) + 'fc1') #this was added recently
                x = tf.layers.dense(x, 1, activation=None, name='ms_' + str(scale) + 'D_logit')

                D_logit.append(x[:, None, None, :])
                x_init = down_sample(x_init)

            return D_logit, features

class MSD_FM:

    def __init__(self, NF, n_scale, n_dis, do_norm=False):
        self.NF = NF
        self.n_scale = n_scale
        self.n_dis = n_dis
        self.do_norm = do_norm

    def get_outputs(self, im, im_final):
        prob_im_is_real, feats_real = self.get_disc(im, reuse=False)
        if isinstance(im_final, list):
            prob_im_final_is_real = []
            feats_fake = []
            for j in im_final:
                pf_is_real, f_fake = self.get_disc(j, reuse=True)
                prob_im_final_is_real.append(pf_is_real)
                feats_fake.append(f_fake)
        else:
            prob_im_final_is_real, feats_fake = self.get_disc(im_final, reuse=True)
        return prob_im_is_real, feats_real, prob_im_final_is_real, feats_fake

    def get_disc(self, x_init, reuse, name="disc"):
        D_logit = []
        features = []
        with tf.variable_scope(name, reuse=reuse):
            for scale in range(self.n_scale):
                with tf.variable_scope('ms_%d' %scale, reuse=reuse):
                    xs, fs = self.D_body(x_init, reuse)
                    D_logit.append(xs)
                    features.append(fs)
            return D_logit, features

    def D_body(self, xinput, reuse, name='dbody'):
        features = []
        with tf.variable_scope(name, reuse=reuse):
            x = general_conv2d(xinput, nf=self.NF, ks=7, s=1, name='c1', do_norm=False, padding='SAME')
            features.append(x)
            x = general_conv2d(x, nf=self.NF * 2, ks=5, s=2, name='c2', do_norm=self.do_norm, padding='SAME')
            features.append(x)
            x = general_conv2d(x, nf=self.NF * 4, ks=5, s=2, name='c3', do_norm=self.do_norm, padding='SAME')
            features.append(x)
            x = general_conv2d(x, nf=self.NF * 4, ks=5, s=2, name='c4', do_norm=self.do_norm, padding='SAME')
            features.append(x)
            x = general_conv2d(x, nf=1, ks=5, s=2, name='c5', activation=None, do_norm=False, padding='SAME')
            return x, features
