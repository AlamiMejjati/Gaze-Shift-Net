# from layers import batch_conv2d
from utils import *
import cv2
from scipy import ndimage
from matplotlib import colors

def apply_color_curve_adjustment(im, color_param):
    L = color_param.shape[-2]
    color_curve_sum = np.sum(color_param, axis=2, keepdims=True) + 1e-9
    total_image = im * 0
    for i in range(L):
        color_curve_i = color_param[:, :, i, :]
        total_image += np.clip(im - 1.0 * i / L, 0, 1.0 / L) * color_curve_i[:, None, :, :]
    total_image *= L / color_curve_sum
    return total_image

def apply_tone_curve_adjustment(im, tone_param):
    L = tone_param.shape[-2]
    tone_curve_sum = np.sum(tone_param, axis=2, keepdims=True) + 1e-9
    total_image = im * 0
    for i in range(L):
        total_image += np.clip(im - 1.0 * i / L, 0, 1.0 / L) * tone_param[:, :, i, :]
    total_image *= L / tone_curve_sum
    return total_image

def apply_BnW(im, BnW_param):
    luminance = rgb2lum(im)
    BnW_im = lerp(im, luminance, BnW_param)
    return BnW_im

def apply_saturation(im, saturation_param):
    im = np.clip(im.squeeze(), 0, 1)
    hsv = colors.rgb_to_hsv(im)
    s = hsv[:, :, 1:2]
    v = hsv[:, :, 2:3]
    enhanced_s = s + (1 - s) * (0.5 - np.abs(0.5 - v)) * 0.8
    hsv1 = np.concatenate([hsv[:, :, 0:1], enhanced_s, hsv[:, :, 2:]], axis=2)
    full_color = colors.hsv_to_rgb(hsv1)
    saturated_im = lerp(im, full_color, saturation_param)
    return np.expand_dims(saturated_im, axis=0)

def apply_contrast(im, contrast_param):
    luminance = np.clip(rgb2lum(im), 0, 1)
    contrast_lum = -np.cos(np.pi * luminance) * 0.5 + 0.5
    contrast_image = im / (luminance + 1e-6) * contrast_lum
    return lerp(im, contrast_image, contrast_param)

def apply_WB(im, whiteBalanceparam):
    rgb_means = np.mean(im, axis=(1, 2), keepdims=True) + 1e-9
    balancing_vec = 0.5 / rgb_means
    balancing_mat = np.tile(balancing_vec, [1, im.shape[1], im.shape[2], 1])
    WB_im = im * balancing_mat
    return lerp(im, WB_im, whiteBalanceparam)

def apply_exposure(im, exposureparam):
    exposed_im = im * np.exp(exposureparam * np.log(2))
    return exposed_im

def apply_affine(im, biasparam, slopeparam):
    biasparam = biasparam[:, :, None, None]
    biasparam = np.tile(biasparam, [1, im.shape[1], im.shape[2], im.shape[3]])
    slopeparam = slopeparam[:, :, None, None]
    slopeparam = np.tile(slopeparam, [1, im.shape[1], im.shape[2], im.shape[3]])
    affine_im = slopeparam*im + biasparam
    return affine_im

def apply_slope(im, slopeparam):
    slopeparam = slopeparam[:, :, None, None]
    slopeparam = np.tile(slopeparam, [1, im.shape[1], im.shape[2], im.shape[3]])
    slope_im = slopeparam*im
    return slope_im

def apply_blur(im, blurparam, ks=51):

    blurred_im = cv2.GaussianBlur(im.squeeze(), (ks, ks), sigmaX=blurparam, sigmaY=blurparam,
                                  borderType=cv2.BORDER_DEFAULT)

    return blurred_im

# def apply_blur(im, blurparam, ks=51):
#
#     im = np.clip(im, 0, 1)
#     blur_vec = [np.exp((-(float(k) - ks)**2)/ (2 * blurparam**2 + 1e-7)) for k in range(ks * 2 + 1)]
#     blur_vec = np.array(blur_vec)
#     blur_vec = blur_vec/(np.sum(blur_vec, axis=-1, keepdims=True)+1e-7)
#     blur_kernel = np.matmul(blur_vec, np.transpose(blur_vec))
#     blur_kernel = np.stack([blur_kernel] * 3, axis=-1)
#     blur_kernel = blur_kernel[None,:,:]
#     blurred_im = ndimage.convolve(im, blur_kernel)/np.size(blur_kernel)
#
#     return blurred_im

def apply_gamma(im, gammaparam):

    gamma_im = im**gammaparam

    return gamma_im

def apply_bias(im, biasparam):
    bias_im = im + biasparam
    return bias_im

def apply_sharpening_fg(im, sharpparam):
    tf1 = (1 / 8) * np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    tf2 = (1 / 8) * np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    im1 = ndimage.convolve(im, np.expand_dims(np.stack([tf1]*3, axis=-1), axis=0))
    im2 = ndimage.convolve(im, np.expand_dims(np.stack([tf2]*3, axis=-1), axis=0))
    imedges = np.sqrt(im1**2 + im2**2)

    sharp_im = im + sharpparam * imedges * im

    return sharp_im

def apply_sharpening_bg(im, sharpparam):

    tf1 = (1 / 8) * np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    tf2 = (1 / 8) * np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    im1 = ndimage.convolve(im, np.expand_dims(np.stack([tf1]*3, axis=-1), axis=0))
    im2 = ndimage.convolve(im, np.expand_dims(np.stack([tf2]*3, axis=-1), axis=0))
    imedges = np.sqrt(im1**2 + im2**2 + 1e-7)

    sharp_im = im - sharpparam * imedges * im

    return sharp_im

def apply_sharpening(im, sharpparam):
    tf1 = (1 / 8) * np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    tf2 = (1 / 8) * np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    im1 = ndimage.convolve(im, np.expand_dims(np.stack([tf1]*3, axis=-1), axis=0))
    im2 = ndimage.convolve(im, np.expand_dims(np.stack([tf2]*3, axis=-1), axis=0))
    imedges = np.sqrt(im1**2 + im2**2)

    sharp_im = im + sharpparam * imedges * im

    return sharp_im

def apply_transformations_np(im, mask, params_fg, params_bg):

    fg_im = apply_gamma(im, params_fg['gamma'])
    bg_im = apply_gamma(im, params_bg['gamma'])

    # Apply sharpening#

    fg_im = apply_sharpening_fg(fg_im, params_fg['sharp'])
    bg_im = apply_sharpening_bg(bg_im, params_bg['sharp'])

    # Apply WB#

    fg_im = apply_WB(fg_im, params_fg['WB'])
    bg_im = apply_WB(bg_im, params_bg['WB'])

    # Apply exposure #

    fg_im = apply_exposure(fg_im, params_fg['exposure'])
    bg_im = apply_exposure(bg_im, params_bg['exposure'])

    # Apply contrast #

    fg_im = apply_contrast(fg_im, params_fg['contrast'])
    bg_im = apply_contrast(bg_im, params_bg['contrast'])

    # Apply saturation #

    fg_im = apply_saturation(fg_im, params_fg['saturation'])
    bg_im = apply_saturation(bg_im, params_bg['saturation'])


    # Apply tone curve adjustement#

    fg_im = apply_tone_curve_adjustment(fg_im, params_fg['tone'])
    bg_im = apply_tone_curve_adjustment(bg_im, params_bg['tone'])

    # Apply color curve adjustement#

    fg_im = apply_color_curve_adjustment(fg_im, params_fg['color'])
    bg_im = apply_color_curve_adjustment(bg_im, params_bg['color'])

    # Apply blur
    bg_im = apply_blur(bg_im, params_bg['blur']+1e-9)

    final_im = fg_im * mask + bg_im * (1 - mask)
    final_im = np.clip(final_im, 0, 1)
    # final_im = (final_im - 0.5) * 2
    return final_im

def apply_transformations_np_gui(im, mask, params_fg, params_bg):


    # Apply sharpening#

    fg_im = apply_sharpening_fg(im, params_fg['sharp_fg'])
    bg_im = apply_sharpening_bg(im, params_bg['sharp_bg'])
    im_sharp = fg_im * mask + bg_im * (1 - mask)
    # Apply exposure #

    fg_im = apply_exposure(im_sharp, params_fg['exposure_fg'])
    bg_im = apply_exposure(im_sharp, params_bg['exposure_bg'])
    im_exp = fg_im * mask + bg_im * (1 - mask)

    # Apply contrast #

    fg_im = apply_contrast(im_exp, params_fg['contrast_fg'])
    bg_im = apply_contrast(im_exp, params_bg['contrast_bg'])
    im_cont = fg_im * mask + bg_im * (1 - mask)

    # Apply tone curve adjustement#

    fg_im = apply_tone_curve_adjustment(im_cont, params_fg['tone_fg'])
    bg_im = apply_tone_curve_adjustment(im_cont, params_bg['tone_bg'])
    im_tone = fg_im * mask + bg_im * (1 - mask)

    # Apply color curve adjustement#

    fg_im = apply_color_curve_adjustment(im_tone, params_fg['color_fg'])
    bg_im = apply_color_curve_adjustment(im_tone, params_bg['color_bg'])
    im_color = fg_im * mask + bg_im * (1 - mask)

    # Apply blur
    # bg_im = apply_blur(bg_im, params_bg['blur_bg']+1e-9)

    final_im = np.clip(im_color, 0, 1)
    # final_im = (final_im - 0.5) * 2
    return final_im

def apply_transformations_np_nobias(im, mask, params_fg, params_bg):

    fg_im = apply_gamma(im, params_fg['gamma_fg'])
    bg_im = apply_gamma(im, params_bg['gamma_bg'])

    # Apply sharpening#

    fg_im = apply_sharpening_fg(fg_im, params_fg['sharp_fg'])
    bg_im = apply_sharpening_bg(bg_im, params_bg['sharp_bg'])

    # Apply bias#

    # fg_im = apply_bias(fg_im, params_fg['bias_fg'])
    # bg_im = apply_bias(bg_im, params_bg['bias_bg'])

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
    bg_im = apply_blur(bg_im, params_bg['blur_bg']+1e-9)

    final_im = fg_im * mask + bg_im * (1 - mask)
    final_im = np.clip(final_im, 0, 1)
    # final_im = (final_im - 0.5) * 2
    return final_im

def apply_transformations_np_decrease(im, mask, params_fg, params_bg):

    fg_im = apply_gamma(im, params_fg['gamma_fg'])
    bg_im = apply_gamma(im, params_bg['gamma_bg'])

    # Apply sharpening#

    fg_im = apply_sharpening_fg(fg_im, params_fg['sharp_fg'])
    bg_im = apply_sharpening_bg(bg_im, params_bg['sharp_bg'])

    # Apply bias#

    # fg_im = apply_bias(fg_im, params_fg['bias_fg'])
    # bg_im = apply_bias(bg_im, params_bg['bias_bg'])

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
    # bg_im = apply_blur(bg_im, params_bg['blur_bg']+1e-9)

    final_im = fg_im * mask + bg_im * (1 - mask)
    final_im = np.clip(final_im, 0, 1)
    # final_im = (final_im - 0.5) * 2
    return final_im

def get_margins(m):
    tmp_x = np.nonzero(np.sum(m, axis=0))
    tmp_y = np.nonzero(np.sum(m, axis=1))
    tmp_x = tmp_x[0]
    tmp_y = tmp_y[0]
    try:
        margins_x = [np.min(tmp_x), np.max(tmp_x)]
        margins_y = [np.min(tmp_y), np.max(tmp_y)]
        cont = False
    except:
        margins_x = 0
        margins_y = 0
        cont = True

    return margins_x, margins_y, cont