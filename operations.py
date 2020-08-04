from layers import batch_conv2d
from utils import *

#some parametric transformations are from https://github.com/yuanming-hu/exposure


def apply_color_curve_adjustment(im, color_param):
    # with tf.name_scope('color_adjustment_station'):
    L = color_param.get_shape().as_list()[2]
    color_curve_sum = tf.reduce_sum(color_param, axis=2, keepdims=True) + 1e-9
    total_image = im * 0
    color_param = tf.split(color_param, L, 2)
    # color_param = [color_param[k][:,:,None,:] for k in range(L)]
    for i in range(L):
        # color_curve_i = color_param[:, :, i, :]
        total_image += tf.clip_by_value(im - 1.0 * i / L, 0, 1.0 / L) * color_param[i]
    total_image *= L / color_curve_sum
    return total_image

def apply_color_curve_adjustment_2Dmap(im, color_param):
    # with tf.name_scope('color_adjustment_station'):
    L = color_param.get_shape().as_list()[3]
    color_curve_sum = tf.reduce_sum(color_param, axis=-2) + 1e-9
    total_image = im * 0
    color_param = tf.split(color_param, L, axis=-2)
    # color_param = [color_param[k][:,:,None,:] for k in range(L)]
    for i in range(L):
        # color_curve_i = color_param[:, :, i, :]
        total_image += tf.clip_by_value(im - 1.0 * i / L, 0, 1.0 / L) * tf.squeeze(color_param[i], axis=-2)
    total_image *= L / color_curve_sum
    return total_image

def apply_tone_curve_adjustment(im, tone_param):
    # with tf.name_scope('tone_adjustment_station'):
    L = tone_param.get_shape().as_list()[2]
    tone_curve_sum = tf.reduce_sum(tone_param, axis=2, keepdims=True) + 1e-9
    total_image = im * 0
    tone_param = tf.split(tone_param, L, 2)
    # tone_param = [tone_param[k][:,:,None,:] for k in range(L)]
    for i in range(L):
        total_image += tf.clip_by_value(im - 1.0 * i / L, 0, 1.0 / L) * tone_param[i]
    total_image *= L / tone_curve_sum
    return total_image

def apply_tone_curve_adjustment_2Dmap(im, tone_param):
    # with tf.name_scope('tone_adjustment_station'):
    L = tone_param.get_shape().as_list()[3]
    tone_curve_sum = tf.reduce_sum(tone_param, axis=-1, keepdims=True) + 1e-9
    total_image = im * 0
    tone_param = tf.split(tone_param, L, axis=-1)
    # tone_param = [tone_param[k][:,:,None,:] for k in range(L)]
    for i in range(L):
        total_image += tf.clip_by_value(im - 1.0 * i / L, 0, 1.0 / L) * tone_param[i]
    total_image *= L / tone_curve_sum
    return total_image

def apply_BnW(im, BnW_param):
    # with tf.name_scope('BnW_station'):
    BnW_param = BnW_param[:, :, None, None]
    luminance = rgb2lum(im)
    BnW_im = lerp(im, luminance, BnW_param)
    return BnW_im

def apply_saturation(im, saturation_param):
    # with tf.name_scope('saturation_station'):
    saturation_param = saturation_param[:, :, None, None]
    im = tf.clip_by_value(im, 0, 1)
    hsv = tf.image.rgb_to_hsv(im)
    s = hsv[:, :, :, 1:2]
    v = hsv[:, :, :, 2:3]
    enhanced_s = s + (1 - s) * (0.5 - tf.abs(0.5 - v)) * 0.8
    hsv1 = tf.concat([hsv[:, :, :, 0:1], enhanced_s, hsv[:, :, :, 2:]], axis=3)
    full_color = tf.image.hsv_to_rgb(hsv1)
    saturated_im = lerp(im, full_color, saturation_param)
    return saturated_im

def apply_saturation_2Dmap(im, saturation_param):
    # with tf.name_scope('saturation_station'):
    # saturation_param = saturation_param[:, :, None, None]
    im = tf.clip_by_value(im, 0, 1)
    hsv = tf.image.rgb_to_hsv(im)
    s = hsv[:, :, :, 1:2]
    v = hsv[:, :, :, 2:3]
    enhanced_s = s + (1 - s) * (0.5 - tf.abs(0.5 - v)) * 0.8
    hsv1 = tf.concat([hsv[:, :, :, 0:1], enhanced_s, hsv[:, :, :, 2:]], axis=3)
    full_color = tf.image.hsv_to_rgb(hsv1)
    saturated_im = lerp(im, full_color, saturation_param)
    return saturated_im

def apply_saturation_bis(im, saturation_param):
    # with tf.name_scope('saturation_station'):
    saturation_param = saturation_param[:, :, None, None]
    im = tf.clip_by_value(im, 0, 1)
    hsv = tf.image.rgb_to_hsv(im)
    s = hsv[:, :, :, 1:2]
    v = hsv[:, :, :, 2:3]
    enhanced_s = s + 0.3
    hsv1 = tf.concat([hsv[:, :, :, 0:1], enhanced_s, hsv[:, :, :, 2:]], axis=3)
    hsv1 = tf.clip_by_value(hsv1, 0,1)
    full_color = tf.image.hsv_to_rgb(hsv1)
    saturated_im = lerp(im, full_color, saturation_param)
    return saturated_im

def apply_contrast(im, contrast_param):
    # with tf.name_scope('contrast_station'):
    contrast_param = contrast_param[:, :, None, None]
    luminance = tf.clip_by_value(rgb2lum(im), 0, 1)
    contrast_lum = -tf.cos(np.pi * luminance) * 0.5 + 0.5
    contrast_image = im / (luminance + 1e-6) * contrast_lum
    return lerp(im, contrast_image, contrast_param)

def apply_contrast_2Dmap(im, contrast_param):
    # with tf.name_scope('contrast_station'):
    # contrast_param = contrast_param[:, :, None, None]
    luminance = tf.clip_by_value(rgb2lum(im), 0, 1)
    contrast_lum = -tf.cos(np.pi * luminance) * 0.5 + 0.5
    contrast_image = im / (luminance + 1e-6) * contrast_lum
    return lerp(im, contrast_image, contrast_param)

def apply_WB(im, whiteBalanceparam):
    # with tf.name_scope('whitebalance_station'):
    whiteBalanceparam = whiteBalanceparam[:, :, None, None]
    rgb_means = tf.reduce_mean(im, axis=(1, 2), keepdims=True) + 1e-9
    balancing_vec = 0.5 / (rgb_means)
    balancing_mat = tf.tile(balancing_vec, [1, tf.shape(im)[1], tf.shape(im)[2], 1])
    WB_im = im * balancing_mat
    return lerp(im, WB_im, whiteBalanceparam)

def apply_WB_2Dmap(im, whiteBalanceparam):
    # with tf.name_scope('whitebalance_station'):
    # whiteBalanceparam = whiteBalanceparam[:, :, None, None]
    rgb_means = tf.reduce_mean(im, axis=(1, 2), keepdims=True) + 1e-9
    balancing_vec = 0.5 / (rgb_means)
    balancing_mat = tf.tile(balancing_vec, [1, tf.shape(im)[1], tf.shape(im)[2], 1])
    WB_im = im * balancing_mat
    return lerp(im, WB_im, whiteBalanceparam)

def apply_WB_(im, whiteBalanceparam):
    # with tf.name_scope('whitebalance__station'):
    # whiteBalanceparam = whiteBalanceparam[:, :, None, None]
    rgb_means = tf.reduce_mean(im, axis=(1, 2), keepdims=True) + 1e-9
    balancing_vec = whiteBalanceparam / (rgb_means)
    balancing_mat = tf.tile(balancing_vec, [1, tf.shape(im)[1], tf.shape(im)[2], 1])
    WB_im = im * balancing_mat
    return WB_im #lerp(im, WB_im, whiteBalanceparam)

def apply_exposure(im, exposureparam):
    # with tf.name_scope('exposure_station'):
    exposureparam = exposureparam[:, :, None, None]
    exposed_im = im * tf.exp(exposureparam * np.log(2))
    return exposed_im

def apply_exposure_gradients(im, fg_params, margins, shape1, shape2):

    exp_concat = tf.concat([fg_params['exposure_start'], fg_params['exposure_end'], margins], axis=-1)

    def create_map(plop):
        param_start, param_end, margins = tf.split(plop, [1, 1, 4], axis=-1)
        param_start = tf.squeeze(param_start)
        param_end = tf.squeeze(param_end)
        margins = tf.cast(tf.squeeze(margins), tf.int32)
        p = tf.linspace(0., 1., margins[1] - margins[0])
        grad_vec = param_start * p + param_end * (1 - p)
        grad_mat = tf.tile(grad_vec[:, None], [1, shape2])
        grad_mat = tf.pad(grad_mat, [[margins[0], shape1 - margins[1]], [0, 0]])
        return grad_mat

    f = lambda x: create_map(x)

    exposure_map = tf.map_fn(f, exp_concat)
    exposed_im = apply_exposure_2Dmap(im, exposure_map[:,:,:,None])

    return exposed_im, exposure_map

def apply_exposure_gradients_xy(im, fg_params, margins, shape1, shape2):

    exp_concat_x = tf.concat([fg_params['exposureparam_startx'], fg_params['exposureparam_endx'], margins], axis=-1)
    exp_concat_y = tf.concat([fg_params['exposureparam_starty'], fg_params['exposureparam_endy'], margins], axis=-1)
    def create_map_y(plop):
        param_start, param_end, margins = tf.split(plop, [1, 1, 4], axis=-1)
        param_start = tf.squeeze(param_start)
        param_end = tf.squeeze(param_end)
        margins = tf.cast(tf.squeeze(margins), tf.int32)
        p = tf.linspace(0., 1., margins[1] - margins[0])
        grad_vec = param_start * p + param_end * (1 - p)
        grad_mat = tf.tile(grad_vec[:, None], [1, shape2])
        grad_mat = tf.pad(grad_mat, [[margins[0], shape1 - margins[1]], [0, 0]])
        return grad_mat

    def create_map_x(plop):
        param_start, param_end, margins = tf.split(plop, [1, 1, 4], axis=-1)
        param_start = tf.squeeze(param_start)
        param_end = tf.squeeze(param_end)
        margins = tf.cast(tf.squeeze(margins), tf.int32)
        p = tf.linspace(0., 1., margins[3] - margins[2])
        grad_vec = param_start * p + param_end * (1 - p)
        grad_mat = tf.tile(grad_vec[None, :], [shape1, 1])
        grad_mat = tf.pad(grad_mat, [[0, 0], [margins[2], shape2 - margins[3]]])
        return grad_mat

    fy = lambda x: create_map_y(x)
    fx = lambda x: create_map_x(x)

    exposure_map_y = tf.map_fn(fy, exp_concat_y)
    exposure_map_x = tf.map_fn(fx, exp_concat_x)

    exposure_map = 0.5*(exposure_map_x + exposure_map_y)#tf.sqrt(tf.pow(exposure_map_x,2) + tf.pow(exposure_map_y,2))
    exposed_im = apply_exposure_2Dmap(im, exposure_map[:,:,:,None])

    return exposed_im, exposure_map

def apply_exposure_2Dmap(im, exposureparam):
    # with tf.name_scope('exposure_station'):
    # exposureparam = exposureparam[:, :, None, None]
    exposed_im = im * tf.exp(exposureparam * np.log(2))
    return exposed_im

def apply_affine(im, biasparam, slopeparam):
    with tf.name_scope('affine_station'):
        biasparam = biasparam[:, :, None, None]
        biasparam = tf.tile(biasparam, [1, tf.shape(im)[1], tf.shape(im)[2], tf.shape(im)[3]])
        slopeparam = slopeparam[:, :, None, None]
        slopeparam = tf.tile(slopeparam, [1, tf.shape(im)[1], tf.shape(im)[2], tf.shape(im)[3]])
        affine_im = slopeparam*im + biasparam
    return affine_im

def apply_slope(im, slopeparam):
    with tf.name_scope('affine_station'):
        slopeparam = slopeparam[:, :, None, None]
        slopeparam = tf.tile(slopeparam, [1, tf.shape(im)[1], tf.shape(im)[2], tf.shape(im)[3]])
        slope_im = slopeparam*im
    return slope_im

def apply_blur(im, blurparam, ks):

    # with tf.name_scope('blurring_station'):
    # im = tf.clip_by_value(im, 0, 1)
    blur_vec = [tf.exp(tf.div(-tf.pow(float(k) - ks, 2), 2 * tf.pow(blurparam, 2) + 1e-7)) for k in
                range(ks * 2 + 1)]
    blur_vec = tf.concat(blur_vec, axis=-1)
    blur_vec = tf.div(blur_vec, tf.reduce_sum(blur_vec, axis=-1, keepdims=True) + 1e-7)
    blur_vec = blur_vec[:, :, None]
    blur_kernel = tf.matmul(blur_vec, blur_vec, transpose_b=True)
    blur_kernel = tf.stack([blur_kernel] * 3, axis=-1)
    blur_kernel = blur_kernel[:, :, :, :, tf.newaxis]
    #####
    # blur_kernel = tf.squeeze(blur_kernel, axis=-1)
    # blurred_im = tf.nn.conv2d(im, blur_kernel[0], [1,1,1,1], 'SAME')
    #####
    blurred_im = batch_conv2d(im, blur_kernel)

    return blurred_im

# def apply_blur(im, blurparam, ks):
#
#     # with tf.name_scope('blurring_station'):
#     d = tf.distributions.Normal(tf.zeros_like(blurparam), blurparam)
#
#     blur_vecs = d.prob(tf.range(start=-ks, limit=ks + 1, dtype=tf.float32))
#     blur_vecs = blur_vecs[:, :, None]
#
#     blur_kernel = tf.matmul(blur_vecs, blur_vecs, transpose_b=True)
#     blur_kernel = tf.stack([blur_kernel] * 3, axis=-1)
#     blur_kernel = blur_kernel[:, :, :, :, tf.newaxis]
#
#     # gauss_kernel = tf.einsum('i,j->ij',
#     #                          vals,
#     #                          vals)
#
#     blur_kernel = blur_kernel / (tf.reduce_sum(blur_kernel)+1e-9)
#     #####
#     # blur_kernel = tf.squeeze(blur_kernel, axis=-1)
#     # blurred_im = tf.nn.conv2d(im, blur_kernel[0], [1,1,1,1], 'SAME')
#     #####
#     blurred_im = batch_conv2d(im, blur_kernel)
#
#     return blurred_im

def apply_gamma(im, gammaparam):

    # with tf.name_scope('gamma_station'):
    gammaparam = gammaparam[:, :, None, None]
    gammaparam = tf.tile(gammaparam, [1, tf.shape(im)[1], tf.shape(im)[2], tf.shape(im)[3]])
    gamma_im = tf.pow(im + 1e-7, gammaparam)

    return gamma_im

def apply_gamma_2Dmap(im, gammaparam):

    # with tf.name_scope('gamma_station'):
    # gammaparam = gammaparam[:, :, None, None]
    # gammaparam = tf.tile(gammaparam, [1, tf.shape(im)[1], tf.shape(im)[2], tf.shape(im)[3]])
    gamma_im = tf.pow(im + 1e-7, gammaparam)

    return gamma_im

def apply_bias(im, biasparam):

    # with tf.name_scope('bias_station'):
    biasparam = biasparam[:, :, None, None]
    biasparam = tf.tile(biasparam, [1, tf.shape(im)[1], tf.shape(im)[2], tf.shape(im)[3]])
    bias_im = im + biasparam
    return bias_im

def apply_sharpening(im, sharpparam, tf1, tf2, direction):

    # with tf.name_scope('sharpening_station'):
    im1 = tf.nn.conv2d(im, tf1, strides=[1, 1, 1, 1], padding='SAME')
    im2 = tf.nn.conv2d(im, tf2, strides=[1, 1, 1, 1], padding='SAME')
    imedges = tf.sqrt(tf.pow(im1, 2) + tf.pow(im2,2) + 1e-7)
    # imedges = tf.pow(im1, 2) + tf.pow(im2, 2)
    # imedges= im1+im2
    sharpparam = sharpparam[:, :, None, None]
    sharpparam = tf.tile(sharpparam, [1, tf.shape(im)[1], tf.shape(im)[2], tf.shape(im)[3]])
    sharp_im = im + sharpparam * imedges * im

    return sharp_im

def apply_sharpening_fg(im, sharpparam, tf1, tf2, direction):

    # with tf.name_scope('sharpening_station_fg'):
    im1 = tf.nn.conv2d(im, tf1, strides=[1, 1, 1, 1], padding='SAME')
    im2 = tf.nn.conv2d(im, tf2, strides=[1, 1, 1, 1], padding='SAME')
    imedges = tf.sqrt(tf.pow(im1, 2) + tf.pow(im2,2) + 1e-7)
    # imedges = tf.pow(im1, 2) + tf.pow(im2, 2)
    # imedges= im1+im2
    sharpparam = sharpparam[:, :, None, None]
    sharpparam = tf.tile(sharpparam, [1, tf.shape(im)[1], tf.shape(im)[2], tf.shape(im)[3]])
    sharp_im = im + sharpparam * imedges * im

    return sharp_im

def apply_sharpening_bg(im, sharpparam, tf1, tf2, direction):

    # with tf.name_scope('sharpening_station_fg'):
    im1 = tf.nn.conv2d(im, tf1, strides=[1, 1, 1, 1], padding='SAME')
    im2 = tf.nn.conv2d(im, tf2, strides=[1, 1, 1, 1], padding='SAME')
    imedges = tf.sqrt(tf.pow(im1, 2) + tf.pow(im2,2) + 1e-7)
    # imedges = tf.pow(im1, 2) + tf.pow(im2, 2)
    # imedges= im1+im2
    sharpparam = sharpparam[:, :, None, None]
    sharpparam = tf.tile(sharpparam, [1, tf.shape(im)[1], tf.shape(im)[2], tf.shape(im)[3]])
    sharp_im = im - sharpparam * imedges * im

    return sharp_im

def apply_sharpening_fg_2Dmap(im, sharpparam, tf1, tf2, direction):

    # with tf.name_scope('sharpening_station_fg'):
    im1 = tf.nn.conv2d(im, tf1, strides=[1, 1, 1, 1], padding='SAME')
    im2 = tf.nn.conv2d(im, tf2, strides=[1, 1, 1, 1], padding='SAME')
    imedges = tf.sqrt(tf.pow(im1, 2) + tf.pow(im2,2) + 1e-7)
    # imedges = tf.pow(im1, 2) + tf.pow(im2, 2)
    # imedges= im1+im2
    # sharpparam = sharpparam[:, :, None, None]
    # sharpparam = tf.tile(sharpparam, [1, tf.shape(im)[1], tf.shape(im)[2], tf.shape(im)[3]])
    sharp_im = im + sharpparam * imedges * im

    return sharp_im

def apply_sharpening_bg_2Dmap(im, sharpparam, tf1, tf2, direction):

    # with tf.name_scope('sharpening_station_fg'):
    im1 = tf.nn.conv2d(im, tf1, strides=[1, 1, 1, 1], padding='SAME')
    im2 = tf.nn.conv2d(im, tf2, strides=[1, 1, 1, 1], padding='SAME')
    imedges = tf.sqrt(tf.pow(im1, 2) + tf.pow(im2,2) + 1e-7)
    # imedges = tf.pow(im1, 2) + tf.pow(im2, 2)
    # imedges= im1+im2
    # sharpparam = sharpparam[:, :, None, None]
    # sharpparam = tf.tile(sharpparam, [1, tf.shape(im)[1], tf.shape(im)[2], tf.shape(im)[3]])
    sharp_im = im - sharpparam * imedges * im

    return sharp_im

def apply_transformations(im, mask, params_fg, params_bg, tf1, tf2, ksize):


    fg_im = apply_gamma(im, params_fg['gamma_fg'])
    bg_im = apply_gamma(im, params_bg['gamma_bg'])

    # Apply sharpening#

    fg_im = apply_sharpening_fg(fg_im, params_fg['sharp_fg'], tf1, tf2, 1)
    bg_im = apply_sharpening_bg(bg_im, params_bg['sharp_bg'], tf1, tf2, 0)

    # Apply bias#

    fg_im = apply_bias(fg_im, params_fg['bias_fg'])
    bg_im = apply_bias(bg_im, params_bg['bias_bg'])

    # Apply WB#

    fg_im = apply_WB(fg_im, params_fg['WB_fg'])
    bg_im = apply_WB(bg_im, params_bg['WB_bg'])

    # Apply exposure #

    fg_im = apply_exposure(fg_im, params_fg['exposure_fg'])
    bg_im = apply_exposure(bg_im, params_bg['exposure_bg'])

    # Apply contrast #

    fg_im = apply_contrast(fg_im, params_fg['contrast_fg'])
    bg_im = apply_contrast(bg_im, params_bg['contrast_bg'])

    # Apply saturation #

    fg_im = apply_saturation(fg_im, params_fg['saturation_fg'])
    bg_im = apply_saturation(bg_im, params_bg['saturation_bg'])

    # Apply BnW adjustement#

    fg_im = apply_BnW(fg_im, params_fg['BnW_fg'])
    bg_im = apply_BnW(bg_im, params_bg['BnW_bg'])

    # Apply tone curve adjustement#

    fg_im = apply_tone_curve_adjustment(fg_im, params_fg['toneAdjustement_fg'])
    bg_im = apply_tone_curve_adjustment(bg_im, params_bg['toneAdjustement_bg'])

    # Apply color curve adjustement#

    fg_im = apply_color_curve_adjustment(fg_im, params_fg['colorAdjustement_fg'])
    bg_im = apply_color_curve_adjustment(bg_im, params_bg['colorAdjustement_bg'])

    # Apply blur
    bg_im = apply_blur(bg_im, params_bg['blur_bg']+1e-9, ksize)

    final_im = fg_im * mask + bg_im * (1 - mask)
    final_im = tf.clip_by_value(final_im, 0, 1)
    final_im = (final_im - 0.5) * 2
    # final_im = (final_im - 0.5) * 2
    return final_im

def apply_transformations_seqgan(bin_im, bin_mask, fg_params, bg_params, tf1, tf2, ksize):
    with tf.name_scope('gamma_station'):
        fg_im = apply_gamma(bin_im, fg_params['gamma'])
        bg_im = apply_gamma(bin_im, bg_params['gamma'])
        im_gamma = tf.identity(fg_im * bin_mask + bg_im * (1 - bin_mask), name='im_gamma')

    # Apply sharpening#
    with tf.name_scope('sharpening_station'):
        fg_im = apply_sharpening_fg(im_gamma, fg_params['sharp'], tf1, tf2, 1)
        bg_im = apply_sharpening_bg(im_gamma, bg_params['sharp'], tf1, tf2, 0)
        im_sharp = tf.identity(fg_im * bin_mask + bg_im * (1 - bin_mask), name='im_sharp')

    # Apply WB#
    with tf.name_scope('WB_station'):
        fg_im = apply_WB(im_sharp, fg_params['WB'])
        bg_im = apply_WB(im_sharp, bg_params['WB'])
        im_wb = tf.identity(fg_im * bin_mask + bg_im * (1 - bin_mask), name='im_wb')

    # Apply exposure #
    with tf.name_scope('exposure_station'):
        fg_im = apply_exposure(im_wb, fg_params['exposure'])
        bg_im = apply_exposure(im_wb, bg_params['exposure'])
        im_exp = tf.identity(fg_im * bin_mask + bg_im * (1 - bin_mask), name='im_exp')

    # Apply contrast #
    with tf.name_scope('contrast_station'):
        fg_im = apply_contrast(im_exp, fg_params['contrast'])
        bg_im = apply_contrast(im_exp, bg_params['contrast'])
        im_cont = tf.identity(fg_im * bin_mask + bg_im * (1 - bin_mask), name='im_cont')

    # Apply saturation #
    with tf.name_scope('saturation_station'):
        fg_im = apply_saturation(im_cont, fg_params['saturation'])
        bg_im = apply_saturation(im_cont, bg_params['saturation'])
        im_sat = tf.identity(fg_im * bin_mask + bg_im * (1 - bin_mask), name='im_sat')

    # Apply tone curve adjustement#
    with tf.name_scope('tone_adjustment_station'):
        fg_im = apply_tone_curve_adjustment(im_sat, fg_params['tone'])
        bg_im = apply_tone_curve_adjustment(im_sat, bg_params['tone'])
        im_tone = tf.identity(fg_im * bin_mask + bg_im * (1 - bin_mask), name='im_tone')

    # Apply color curve adjustement#
    with tf.name_scope('color_adjustment_station'):
        fg_im = apply_color_curve_adjustment(im_tone, fg_params['color'])
        bg_im = apply_color_curve_adjustment(im_tone, bg_params['color'])
        im_color = tf.identity(fg_im * bin_mask + bg_im * (1 - bin_mask), name='im_color')

    # Apply blur
    with tf.name_scope('blur_station'):
        bg_im = apply_blur(im_color, bg_params['blur'], ksize)
        im_blur = tf.identity(im_color * bin_mask + bg_im * (1 - bin_mask), name='im_blur')

    final_im = tf.clip_by_value(im_blur, 0, 1)
    final_im = (final_im - 0.5) * 2
    # final_im = (final_im - 0.5) * 2
    return final_im

def apply_transformations_exp_gradients(bin_im, bin_mask, fg_params, bg_params, margins):

    # Apply exposure #
    with tf.name_scope('exposure_station'):
        fg_im, _ = apply_exposure_gradients(bin_im, fg_params, margins, tf.shape(bin_im)[1], tf.shape(bin_im)[2])
        bg_im = apply_exposure(bin_im, bg_params['exposure'])
        im_exp = tf.identity(fg_im * bin_mask + bg_im * (1 - bin_mask), name='im_exp')

    final_im = tf.clip_by_value(im_exp, 0, 1)
    final_im = (final_im - 0.5) * 2
    # final_im = (final_im - 0.5) * 2
    return final_im

def apply_transformations_exp_gradients_xy(bin_im, bin_mask, fg_params, bg_params, margins, tf1, tf2, ksize):

    # Apply sharpening#
    with tf.name_scope('sharpening_station'):
        fg_im = apply_sharpening_fg(bin_im, fg_params['sharp'], tf1, tf2, 1)
        bg_im = apply_sharpening_bg(bin_im, bg_params['sharp'], tf1, tf2, 0)
        im_sharp = tf.identity(fg_im * bin_mask + bg_im * (1 - bin_mask), name='im_sharp')


    # Apply exposure #
    with tf.name_scope('exposure_station'):
        fg_im, _ = apply_exposure_gradients_xy(im_sharp, fg_params, margins, tf.shape(bin_im)[1], tf.shape(bin_im)[2])
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


    final_im = tf.clip_by_value(im_color, 0, 1)
    final_im = (final_im - 0.5) * 2
    # final_im = (final_im - 0.5) * 2
    return final_im

def apply_transformations_seqgan_noblur(bin_im, bin_mask, fg_params, bg_params, tf1, tf2, ksize):
    with tf.name_scope('gamma_station'):
        fg_im = apply_gamma(bin_im, fg_params['gamma'])
        bg_im = apply_gamma(bin_im, bg_params['gamma'])
        im_gamma = tf.identity(fg_im * bin_mask + bg_im * (1 - bin_mask), name='im_gamma')

    # Apply sharpening#
    with tf.name_scope('sharpening_station'):
        fg_im = apply_sharpening_fg(im_gamma, fg_params['sharp'], tf1, tf2, 1)
        bg_im = apply_sharpening_bg(im_gamma, bg_params['sharp'], tf1, tf2, 0)
        im_sharp = tf.identity(fg_im * bin_mask + bg_im * (1 - bin_mask), name='im_sharp')

    # Apply WB#
    with tf.name_scope('WB_station'):
        fg_im = apply_WB(im_sharp, fg_params['WB'])
        bg_im = apply_WB(im_sharp, bg_params['WB'])
        im_wb = tf.identity(fg_im * bin_mask + bg_im * (1 - bin_mask), name='im_wb')

    # Apply exposure #
    with tf.name_scope('exposure_station'):
        fg_im = apply_exposure(im_wb, fg_params['exposure'])
        bg_im = apply_exposure(im_wb, bg_params['exposure'])
        im_exp = tf.identity(fg_im * bin_mask + bg_im * (1 - bin_mask), name='im_exp')

    # Apply contrast #
    with tf.name_scope('contrast_station'):
        fg_im = apply_contrast(im_exp, fg_params['contrast'])
        bg_im = apply_contrast(im_exp, bg_params['contrast'])
        im_cont = tf.identity(fg_im * bin_mask + bg_im * (1 - bin_mask), name='im_cont')

    # Apply saturation #
    with tf.name_scope('saturation_station'):
        fg_im = apply_saturation(im_cont, fg_params['saturation'])
        bg_im = apply_saturation(im_cont, bg_params['saturation'])
        im_sat = tf.identity(fg_im * bin_mask + bg_im * (1 - bin_mask), name='im_sat')

    # Apply tone curve adjustement#
    with tf.name_scope('tone_adjustment_station'):
        fg_im = apply_tone_curve_adjustment(im_sat, fg_params['tone'])
        bg_im = apply_tone_curve_adjustment(im_sat, bg_params['tone'])
        im_tone = tf.identity(fg_im * bin_mask + bg_im * (1 - bin_mask), name='im_tone')

    # Apply color curve adjustement#
    with tf.name_scope('color_adjustment_station'):
        fg_im = apply_color_curve_adjustment(im_tone, fg_params['color'])
        bg_im = apply_color_curve_adjustment(im_tone, bg_params['color'])
        im_color = tf.identity(fg_im * bin_mask + bg_im * (1 - bin_mask), name='im_color')

    final_im = tf.clip_by_value(im_color, 0, 1)
    final_im = (final_im - 0.5) * 2
    # final_im = (final_im - 0.5) * 2
    return final_im

def apply_transformations_sharp_exp_cont_tone_color_blur(bin_im, bin_mask, fg_params, bg_params, tf1, tf2, ksize):

    # Apply sharpening#
    with tf.name_scope('sharpening_station'):
        fg_im = apply_sharpening_fg(bin_im, fg_params['sharp'], tf1, tf2, 1)
        bg_im = apply_sharpening_bg(bin_im, bg_params['sharp'], tf1, tf2, 0)
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

    # Apply blur
    with tf.name_scope('blur_station'):
        bg_im = apply_blur(im_color, bg_params['blur'], ksize)
        im_blur = tf.identity(im_color * bin_mask + bg_im * (1 - bin_mask), name='im_blur')

    final_im = tf.clip_by_value(im_blur, 0, 1)
    final_im = (final_im - 0.5) * 2
    # final_im = (final_im - 0.5) * 2
    return final_im

def apply_transformations_sharp_exp_cont_tone_color(bin_im, bin_mask, fg_params, bg_params, tf1, tf2):

    # Apply sharpening#
    with tf.name_scope('sharpening_station'):
        fg_im = apply_sharpening(bin_im, fg_params['sharp'], tf1, tf2, 1)
        bg_im = apply_sharpening(bin_im, bg_params['sharp'], tf1, tf2, 0)
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

    final_im = tf.clip_by_value(im_color, 0, 1)
    # final_im = (final_im - 0.5) * 2
    # final_im = (final_im - 0.5) * 2
    return final_im

def apply_transformations_sharp_exp_cont_tone_color_bis(bin_im, bin_mask, fg_params, bg_params, tf1, tf2, ksize):

    # Apply sharpening#
    with tf.name_scope('sharpening_station'):
        fg_im = apply_sharpening(bin_im, fg_params['sharp'], tf1, tf2, 1)
        bg_im = apply_sharpening(bin_im, bg_params['sharp'], tf1, tf2, 0)
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


    final_im = tf.clip_by_value(im_color, 0, 1)
    final_im = (final_im - 0.5) * 2
    # final_im = (final_im - 0.5) * 2
    return final_im

def apply_transformations_seqgan_only_tone_color(bin_im, bin_mask, fg_params, bg_params, tf1, tf2, ksize):

    # Apply tone curve adjustement#
    with tf.name_scope('tone_adjustment_station'):
        fg_im = apply_tone_curve_adjustment(bin_im, fg_params['tone'])
        bg_im = apply_tone_curve_adjustment(bin_im, bg_params['tone'])
        im_tone = tf.identity(fg_im * bin_mask + bg_im * (1 - bin_mask), name='im_tone')

    # Apply color curve adjustement#
    with tf.name_scope('color_adjustment_station'):
        fg_im = apply_color_curve_adjustment(im_tone, fg_params['color'])
        bg_im = apply_color_curve_adjustment(im_tone, bg_params['color'])
        im_color = tf.identity(fg_im * bin_mask + bg_im * (1 - bin_mask), name='im_color')

    final_im = tf.clip_by_value(im_color, 0, 1)
    final_im = (final_im - 0.5) * 2
    # final_im = (final_im - 0.5) * 2
    return final_im

def apply_transformations_predict_neuron(im, bin_mask, fg_params, bg_params):

    fg_im = im[:, :, :, :, None] * fg_params['w1'][:, None, None, None] + fg_params['b1'][:, None, None, None]
    fg_im = tf.nn.leaky_relu(fg_im)
    bg_im = im[:, :, :, :, None] * bg_params['w1'][:, None, None, None] + bg_params['b1'][:, None, None, None]
    bg_im = tf.nn.leaky_relu(bg_im)

    fg_im = tf.reduce_sum(fg_im * fg_params['w2'][:, None, None, None], axis=-1) + fg_params['b2'][:, None, None]
    bg_im = tf.reduce_sum(bg_im * bg_params['w2'][:, None, None, None], axis=-1) + bg_params['b2'][:, None, None]
    final_im = fg_im * bin_mask + bg_im * (1 - bin_mask)

    final_im = tf.clip_by_value(final_im, -1, 1)
    # final_im = (final_im - 0.5) * 2
    return final_im

def apply_transformations_onebranch(bin_im, bin_mask, fg_params, tf1, tf2, ksize):
    with tf.name_scope('gamma_station'):
        fg_im = apply_gamma(bin_im, fg_params['gamma'])
        im_gamma = tf.identity(fg_im * bin_mask + bin_im * (1 - bin_mask), name='im_gamma')

    # Apply sharpening#
    with tf.name_scope('sharpening_station'):
        fg_im = apply_sharpening_fg(im_gamma, fg_params['sharp'], tf1, tf2, 1)
        im_sharp = tf.identity(fg_im * bin_mask + bin_im * (1 - bin_mask), name='im_sharp')

    # Apply WB#
    with tf.name_scope('WB_station'):
        fg_im = apply_WB(im_sharp, fg_params['WB'])
        im_wb = tf.identity(fg_im * bin_mask + bin_im * (1 - bin_mask), name='im_wb')

    # Apply exposure #
    with tf.name_scope('exposure_station'):
        fg_im = apply_exposure(im_wb, fg_params['exposure'])
        im_exp = tf.identity(fg_im * bin_mask + bin_im * (1 - bin_mask), name='im_exp')

    # Apply contrast #
    with tf.name_scope('contrast_station'):
        fg_im = apply_contrast(im_exp, fg_params['contrast'])
        im_cont = tf.identity(fg_im * bin_mask + bin_im * (1 - bin_mask), name='im_cont')

    # Apply saturation #
    with tf.name_scope('saturation_station'):
        fg_im = apply_saturation(im_cont, fg_params['saturation'])
        im_sat = tf.identity(fg_im * bin_mask + bin_im * (1 - bin_mask), name='im_sat')

    # Apply tone curve adjustement#
    with tf.name_scope('tone_adjustment_station'):
        fg_im = apply_tone_curve_adjustment(im_sat, fg_params['tone'])
        im_tone = tf.identity(fg_im * bin_mask + bin_im * (1 - bin_mask), name='im_tone')

    # Apply color curve adjustement#
    with tf.name_scope('color_adjustment_station'):
        fg_im = apply_color_curve_adjustment(im_tone, fg_params['color'])
        im_color = tf.identity(fg_im * bin_mask + bin_im * (1 - bin_mask), name='im_color')

    final_im = tf.clip_by_value(im_color, 0, 1)
    final_im = (final_im - 0.5) * 2
    # final_im = (final_im - 0.5) * 2
    return final_im

def apply_transformations_seqgan_noblur_notone_nocolor(bin_im, bin_mask, fg_params, bg_params, tf1, tf2, ksize):
    with tf.name_scope('gamma_station'):
        fg_im = apply_gamma(bin_im, fg_params['gamma'])
        bg_im = apply_gamma(bin_im, bg_params['gamma'])
        im_gamma = tf.identity(fg_im * bin_mask + bg_im * (1 - bin_mask), name='im_gamma')

    # Apply sharpening#
    with tf.name_scope('sharpening_station'):
        fg_im = apply_sharpening_fg(im_gamma, fg_params['sharp'], tf1, tf2, 1)
        bg_im = apply_sharpening_bg(im_gamma, bg_params['sharp'], tf1, tf2, 0)
        im_sharp = tf.identity(fg_im * bin_mask + bg_im * (1 - bin_mask), name='im_sharp')

    # Apply WB#
    with tf.name_scope('WB_station'):
        fg_im = apply_WB(im_sharp, fg_params['WB'])
        bg_im = apply_WB(im_sharp, bg_params['WB'])
        im_wb = tf.identity(fg_im * bin_mask + bg_im * (1 - bin_mask), name='im_wb')

    # Apply exposure #
    with tf.name_scope('exposure_station'):
        fg_im = apply_exposure(im_wb, fg_params['exposure'])
        bg_im = apply_exposure(im_wb, bg_params['exposure'])
        im_exp = tf.identity(fg_im * bin_mask + bg_im * (1 - bin_mask), name='im_exp')

    # Apply contrast #
    with tf.name_scope('contrast_station'):
        fg_im = apply_contrast(im_exp, fg_params['contrast'])
        bg_im = apply_contrast(im_exp, bg_params['contrast'])
        im_cont = tf.identity(fg_im * bin_mask + bg_im * (1 - bin_mask), name='im_cont')

    # Apply saturation #
    with tf.name_scope('saturation_station'):
        fg_im = apply_saturation(im_cont, fg_params['saturation'])
        bg_im = apply_saturation(im_cont, bg_params['saturation'])
        im_sat = tf.identity(fg_im * bin_mask + bg_im * (1 - bin_mask), name='im_sat')

    final_im = tf.clip_by_value(im_sat, 0, 1)
    final_im = (final_im - 0.5) * 2
    # final_im = (final_im - 0.5) * 2
    return final_im

def apply_transformations_seqgan_noblur_2Dmap(bin_im, fg_params, tf1, tf2, ksize):
    with tf.name_scope('gamma_station'):
        im_gamma = apply_gamma_2Dmap(bin_im, fg_params['gamma'])

    # Apply sharpening#
    with tf.name_scope('sharpening_station'):
        im_sharp = apply_sharpening_fg_2Dmap(im_gamma, fg_params['sharp'], tf1, tf2, 1)

    # Apply WB#
    with tf.name_scope('WB_station'):
        im_wb = apply_WB_2Dmap(im_sharp, fg_params['WB'])

    # Apply exposure #
    with tf.name_scope('exposure_station'):
        im_exp = apply_exposure_2Dmap(im_wb, fg_params['exposure'])

    # Apply contrast #
    with tf.name_scope('contrast_station'):
        im_cont = apply_contrast_2Dmap(im_exp, fg_params['contrast'])

    # Apply saturation #
    with tf.name_scope('saturation_station'):
        im_sat = apply_saturation_2Dmap(im_cont, fg_params['saturation'])

    # Apply tone curve adjustement#
    with tf.name_scope('tone_adjustment_station'):
        im_tone = apply_tone_curve_adjustment_2Dmap(im_sat, fg_params['tone'])

    # Apply color curve adjustement#
    with tf.name_scope('color_adjustment_station'):
        im_color = apply_color_curve_adjustment_2Dmap(im_tone, fg_params['color'])


    final_im = tf.clip_by_value(im_color, 0, 1)
    final_im = (final_im - 0.5) * 2
    # final_im = (final_im - 0.5) * 2
    return final_im

def apply_transformations_seqgan_blurfirst(im, mask, params_fg, params_bg, tf1, tf2, ksize):


    bg_im = apply_blur(im, params_bg['blur_bg']+1e-9, ksize)
    blur_im = im * mask + bg_im * (1 - mask)

    fg_im = apply_gamma(blur_im, params_fg['gamma_fg'])
    bg_im = apply_gamma(blur_im, params_bg['gamma_bg'])
    gamma_im = fg_im * mask + bg_im * (1 - mask)
    # Apply sharpening#

    fg_im = apply_sharpening_fg(gamma_im, params_fg['sharp_fg'], tf1, tf2, 1)
    bg_im = apply_sharpening_bg(gamma_im, params_bg['sharp_bg'], tf1, tf2, 0)
    sharp_im = fg_im * mask + bg_im * (1 - mask)

    # Apply WB#

    fg_im = apply_WB(sharp_im, params_fg['WB_fg'])
    bg_im = apply_WB(sharp_im, params_bg['WB_bg'])
    wb_im = fg_im * mask + bg_im * (1 - mask)

    # Apply exposure #

    fg_im = apply_exposure(wb_im, params_fg['exposure_fg'])
    bg_im = apply_exposure(wb_im, params_bg['exposure_bg'])
    exposure_im = fg_im * mask + bg_im * (1 - mask)

    # Apply contrast #

    fg_im = apply_contrast(exposure_im, params_fg['contrast_fg'])
    bg_im = apply_contrast(exposure_im, params_bg['contrast_bg'])
    cont_im = fg_im * mask + bg_im * (1 - mask)

    # Apply saturation #

    fg_im = apply_saturation(cont_im, params_fg['saturation_fg'])
    bg_im = apply_saturation(cont_im, params_bg['saturation_bg'])
    sat_im = fg_im * mask + bg_im * (1 - mask)

    # Apply tone curve adjustement#

    fg_im = apply_tone_curve_adjustment(sat_im, params_fg['toneAdjustement_fg'])
    bg_im = apply_tone_curve_adjustment(sat_im, params_bg['toneAdjustement_bg'])
    tone_im = fg_im * mask + bg_im * (1 - mask)

    # Apply color curve adjustement#

    fg_im = apply_color_curve_adjustment(tone_im, params_fg['colorAdjustement_fg'])
    bg_im = apply_color_curve_adjustment(tone_im, params_bg['colorAdjustement_bg'])
    color_im = fg_im * mask + bg_im * (1 - mask)


    final_im = fg_im * mask + bg_im * (1 - mask)
    final_im = tf.clip_by_value(final_im, 0, 1)
    final_im = (final_im - 0.5) * 2
    # final_im = (final_im - 0.5) * 2
    return final_im

def apply_transformations_zbin(im, mask, z, params_fg, params_bg, tf1, tf2, ksize):

    z_i = tf.split(z, 9, axis=-1)
    z_i = [zed[:, :, tf.newaxis, tf.newaxis] for zed in z_i]
    fg_im_ = apply_gamma(im, params_fg['gamma_fg'])
    bg_im_ = apply_gamma(im, params_bg['gamma_bg'])
    fg_im = z_i[0] * fg_im_ + (1 - z_i[0]) * im
    bg_im = z_i[0] * bg_im_ + (1 - z_i[0]) * im
    # Apply sharpening#

    fg_im_ = apply_sharpening_fg(fg_im, params_fg['sharp_fg'], tf1, tf2, 1)
    bg_im_ = apply_sharpening_bg(bg_im, params_bg['sharp_bg'], tf1, tf2, 0)
    fg_im = z_i[1] * fg_im_ + (1 - z_i[1]) * fg_im
    bg_im = z_i[1] * bg_im_ + (1 - z_i[1]) * bg_im

    # Apply bias#

    # fg_im_ = apply_bias(fg_im, params_fg['bias_fg'])
    # bg_im_ = apply_bias(bg_im, params_bg['bias_bg'])
    # fg_im = z_i[2] * fg_im_ + (1 - z_i[2]) * fg_im
    # bg_im = z_i[2] * bg_im_ + (1 - z_i[2]) * bg_im

    # Apply WB#

    fg_im_ = apply_WB(fg_im, params_fg['WB_fg'])
    bg_im_ = apply_WB(bg_im, params_bg['WB_bg'])
    fg_im = z_i[2] * fg_im_ + (1 - z_i[2]) * fg_im
    bg_im = z_i[2] * bg_im_ + (1 - z_i[2]) * bg_im

    # Apply exposure #

    fg_im_ = apply_exposure(fg_im, params_fg['exposure_fg'])
    bg_im_ = apply_exposure(bg_im, params_bg['exposure_bg'])
    fg_im = z_i[3] * fg_im_ + (1 - z_i[3]) * fg_im
    bg_im = z_i[3] * bg_im_ + (1 - z_i[3]) * bg_im

    # Apply contrast #

    fg_im_ = apply_contrast(fg_im, params_fg['contrast_fg'])
    bg_im_ = apply_contrast(bg_im, params_bg['contrast_bg'])
    fg_im = z_i[4] * fg_im_ + (1 - z_i[4]) * fg_im
    bg_im = z_i[4] * bg_im_ + (1 - z_i[4]) * bg_im
    # Apply saturation #

    fg_im_ = apply_saturation(fg_im, params_fg['saturation_fg'])
    bg_im_ = apply_saturation(bg_im, params_bg['saturation_bg'])
    fg_im = z_i[5] * fg_im_ + (1 - z_i[5]) * fg_im
    bg_im = z_i[5] * bg_im_ + (1 - z_i[5]) * bg_im
    # Apply BnW adjustement#

    # fg_im_ = apply_BnW(fg_im, params_fg['BnW_fg'])
    # bg_im_ = apply_BnW(bg_im, params_bg['BnW_bg'])
    # fg_im = z_i[7] * fg_im_ + (1 - z_i[7]) * fg_im
    # bg_im = z_i[7] * bg_im_ + (1 - z_i[7]) * bg_im
    # Apply tone curve adjustement#

    fg_im_ = apply_tone_curve_adjustment(fg_im, params_fg['toneAdjustement_fg'])
    bg_im_ = apply_tone_curve_adjustment(bg_im, params_bg['toneAdjustement_bg'])
    fg_im = z_i[6] * fg_im_ + (1 - z_i[6]) * fg_im
    bg_im = z_i[6] * bg_im_ + (1 - z_i[6]) * bg_im
    # Apply color curve adjustement#

    fg_im_ = apply_color_curve_adjustment(fg_im, params_fg['colorAdjustement_fg'])
    bg_im_ = apply_color_curve_adjustment(bg_im, params_bg['colorAdjustement_bg'])
    fg_im = z_i[7] * fg_im_ + (1 - z_i[7]) * fg_im
    bg_im = z_i[7] * bg_im_ + (1 - z_i[7]) * bg_im
    # Apply blur
    bg_im_ = apply_blur(bg_im, params_bg['blur_bg']+1e-9, ksize)
    bg_im = z_i[8] * bg_im_ + (1 - z_i[8]) * bg_im

    final_im = fg_im * mask + bg_im * (1 - mask)
    final_im = tf.clip_by_value(final_im, 0, 1)
    final_im = (final_im - 0.5) * 2
    # final_im = (final_im - 0.5) * 2
    return final_im

def apply_transformations_dualbranch_zbin(im, mask, z, params_fg, params_bg, tf1, tf2, ksize, direction):

    z_i = tf.split(z, 9, axis=-1)
    z_i = [zed[:, :, tf.newaxis, tf.newaxis] for zed in z_i]
    fg_im_ = apply_gamma(im, params_fg['gamma'])
    bg_im_ = apply_gamma(im, params_bg['gamma'])
    fg_im = z_i[0] * fg_im_ + (1 - z_i[0]) * im
    bg_im = z_i[0] * bg_im_ + (1 - z_i[0]) * im
    # Apply sharpening#

    fg_im_ = apply_sharpening_fg(fg_im, params_fg['sharp'], tf1, tf2, 1)
    bg_im_ = apply_sharpening_bg(bg_im, params_bg['sharp'], tf1, tf2, 0)
    fg_im = z_i[1] * fg_im_ + (1 - z_i[1]) * fg_im
    bg_im = z_i[1] * bg_im_ + (1 - z_i[1]) * bg_im

    # Apply bias#

    # fg_im_ = apply_bias(fg_im, params_fg['bias_fg'])
    # bg_im_ = apply_bias(bg_im, params_bg['bias_bg'])
    # fg_im = z_i[2] * fg_im_ + (1 - z_i[2]) * fg_im
    # bg_im = z_i[2] * bg_im_ + (1 - z_i[2]) * bg_im

    # Apply WB#

    fg_im_ = apply_WB(fg_im, params_fg['WB'])
    bg_im_ = apply_WB(bg_im, params_bg['WB'])
    fg_im = z_i[2] * fg_im_ + (1 - z_i[2]) * fg_im
    bg_im = z_i[2] * bg_im_ + (1 - z_i[2]) * bg_im

    # Apply exposure #

    fg_im_ = apply_exposure(fg_im, params_fg['exposure'])
    bg_im_ = apply_exposure(bg_im, params_bg['exposure'])
    fg_im = z_i[3] * fg_im_ + (1 - z_i[3]) * fg_im
    bg_im = z_i[3] * bg_im_ + (1 - z_i[3]) * bg_im

    # Apply contrast #

    fg_im_ = apply_contrast(fg_im, params_fg['contrast'])
    bg_im_ = apply_contrast(bg_im, params_bg['contrast'])
    fg_im = z_i[4] * fg_im_ + (1 - z_i[4]) * fg_im
    bg_im = z_i[4] * bg_im_ + (1 - z_i[4]) * bg_im
    # Apply saturation #

    fg_im_ = apply_saturation(fg_im, params_fg['saturation'])
    bg_im_ = apply_saturation(bg_im, params_bg['saturation'])
    fg_im = z_i[5] * fg_im_ + (1 - z_i[5]) * fg_im
    bg_im = z_i[5] * bg_im_ + (1 - z_i[5]) * bg_im
    # Apply BnW adjustement#

    # fg_im_ = apply_BnW(fg_im, params_fg['BnW_fg'])
    # bg_im_ = apply_BnW(bg_im, params_bg['BnW_bg'])
    # fg_im = z_i[7] * fg_im_ + (1 - z_i[7]) * fg_im
    # bg_im = z_i[7] * bg_im_ + (1 - z_i[7]) * bg_im
    # Apply tone curve adjustement#

    fg_im_ = apply_tone_curve_adjustment(fg_im, params_fg['tone'])
    bg_im_ = apply_tone_curve_adjustment(bg_im, params_bg['tone'])
    fg_im = z_i[6] * fg_im_ + (1 - z_i[6]) * fg_im
    bg_im = z_i[6] * bg_im_ + (1 - z_i[6]) * bg_im
    # Apply color curve adjustement#

    fg_im_ = apply_color_curve_adjustment(fg_im, params_fg['color'])
    bg_im_ = apply_color_curve_adjustment(bg_im, params_bg['color'])
    fg_im = z_i[7] * fg_im_ + (1 - z_i[7]) * fg_im
    bg_im = z_i[7] * bg_im_ + (1 - z_i[7]) * bg_im
    # Apply blur
    if direction == 'inc':
        bg_im_ = apply_blur(bg_im, params_bg['blur']+1e-9, ksize)
        bg_im = z_i[8] * bg_im_ + (1 - z_i[8]) * bg_im

    final_im = fg_im * mask + bg_im * (1 - mask)
    final_im = tf.clip_by_value(final_im, 0, 1)
    final_im = (final_im - 0.5) * 2
    # final_im = (final_im - 0.5) * 2
    return final_im

def apply_transformations_seqgan_dualbranch(im, mask, z, params_fg, params_bg, tf1, tf2, ksize, direction):

    z_i = tf.split(z, 9, axis=-1)
    z_i = [zed[:, :, tf.newaxis, tf.newaxis] for zed in z_i]
    fg_im_ = apply_gamma(im, params_fg['gamma'])
    bg_im_ = apply_gamma(im, params_bg['gamma'])
    fg_im = z_i[0] * fg_im_ + (1 - z_i[0]) * im
    bg_im = z_i[0] * bg_im_ + (1 - z_i[0]) * im
    gamma_im = fg_im * mask + bg_im * (1 - mask)
    # Apply sharpening#

    fg_im_ = apply_sharpening_fg(gamma_im, params_fg['sharp'], tf1, tf2, 1)
    bg_im_ = apply_sharpening_bg(gamma_im, params_bg['sharp'], tf1, tf2, 0)
    fg_im = z_i[1] * fg_im_ + (1 - z_i[1]) * gamma_im
    bg_im = z_i[1] * bg_im_ + (1 - z_i[1]) * gamma_im
    sharp_im = fg_im * mask + bg_im * (1 - mask)

    # Apply WB#

    fg_im_ = apply_WB(sharp_im, params_fg['WB'])
    bg_im_ = apply_WB(sharp_im, params_bg['WB'])
    fg_im = z_i[2] * fg_im_ + (1 - z_i[2]) * sharp_im
    bg_im = z_i[2] * bg_im_ + (1 - z_i[2]) * sharp_im
    wb_im = fg_im * mask + bg_im * (1 - mask)

    # Apply exposure #

    fg_im_ = apply_exposure(wb_im, params_fg['exposure'])
    bg_im_ = apply_exposure(wb_im, params_bg['exposure'])
    fg_im = z_i[3] * fg_im_ + (1 - z_i[3]) * wb_im
    bg_im = z_i[3] * bg_im_ + (1 - z_i[3]) * wb_im
    exp_im = fg_im * mask + bg_im * (1 - mask)

    # Apply contrast #

    fg_im_ = apply_contrast(exp_im, params_fg['contrast'])
    bg_im_ = apply_contrast(exp_im, params_bg['contrast'])
    fg_im = z_i[4] * fg_im_ + (1 - z_i[4]) * exp_im
    bg_im = z_i[4] * bg_im_ + (1 - z_i[4]) * exp_im
    cont_im = fg_im * mask + bg_im * (1 - mask)
    # Apply saturation #

    fg_im_ = apply_saturation(cont_im, params_fg['saturation'])
    bg_im_ = apply_saturation(cont_im, params_bg['saturation'])
    fg_im = z_i[5] * fg_im_ + (1 - z_i[5]) * cont_im
    bg_im = z_i[5] * bg_im_ + (1 - z_i[5]) * cont_im
    sat_im = fg_im * mask + bg_im * (1 - mask)
    # Apply tone curve adjustement#

    fg_im_ = apply_tone_curve_adjustment(sat_im, params_fg['tone'])
    bg_im_ = apply_tone_curve_adjustment(sat_im, params_bg['tone'])
    fg_im = z_i[6] * fg_im_ + (1 - z_i[6]) * sat_im
    bg_im = z_i[6] * bg_im_ + (1 - z_i[6]) * sat_im
    tone_im = fg_im * mask + bg_im * (1 - mask)
    # Apply color curve adjustement#

    fg_im_ = apply_color_curve_adjustment(tone_im, params_fg['color'])
    bg_im_ = apply_color_curve_adjustment(tone_im, params_bg['color'])
    fg_im = z_i[7] * fg_im_ + (1 - z_i[7]) * tone_im
    bg_im = z_i[7] * bg_im_ + (1 - z_i[7]) * tone_im
    color_im = fg_im * mask + bg_im * (1 - mask)
    # Apply blur
    if direction == 'inc':
        bg_im_ = apply_blur(color_im, params_bg['blur']+1e-9, ksize)
        bg_im = z_i[8] * bg_im_ + (1 - z_i[8]) * color_im

    final_im = fg_im * mask + bg_im * (1 - mask)
    final_im = tf.clip_by_value(final_im, 0, 1)
    final_im = (final_im - 0.5) * 2
    # final_im = (final_im - 0.5) * 2
    return final_im