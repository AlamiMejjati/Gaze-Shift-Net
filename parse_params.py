import numpy as np

params = '''WB_fg: 0.434

gamma_fg: 1.012

contrast_fg: 0.834

color_fg: 0.978 0.930 1.027 1.064 0.928 1.039 0.987 0.956 1.006 1.031 1.015 1.001 1.038 1.025 1.001 0.981 1.069 0.984 1.033 1.065 0.975 0.942 1.046 0.934

exposure_fg: 0.080

tone_fg: 0.600 1.256 1.390 1.166 1.716 1.507 1.522 1.014

saturation_fg: -0.451

sharp_fg: 0.596

WB_bg: 0.034

gamma_bg: 0.954

contrast_bg: -0.012

color_bg: 0.931 1.070 0.973 0.964 1.002 0.924 0.982 0.985 1.075 1.023 0.966 0.997 1.043 1.012 0.976 1.056 0.951 1.005 0.973 1.045 0.980 1.024 1.026 1.028

exposure_bg: -0.045

tone_bg: 1.593 0.746 1.136 1.092 0.881 0.663 0.935 1.210

saturation_bg: 0.313

sharp_bg: 0.003

blur_bg: 0.285'''


# im_id = 94
im_id = 1
im_name = 'im%d.jpg'%im_id
mask_name = 'mask%d.jpg'%im_id

param_strings= params.split ('\n\n')
params_fg_end = {}
params_bg_end = {}
for param in param_strings:
    param_i = param.split(': ')
    if 'fg' in param_i[0]:
        param_i[0] = param_i[0].split('_')[0]
        if ('color' in param_i[0]) or ('tone' in param_i[0]):
            listparamsi = param_i[1].split(' ')
            listparamsi = [float(k) for k in listparamsi]
            if 'color' in param_i[0]:
                ca = np.array(listparamsi)[None, None, :, None]
                params_fg_end[param_i[0]] = np.reshape(ca, [1, 1, -1, 3])
            elif 'tone' in param_i[0]:
                ta = np.array(listparamsi)[None, None, :, None]
                params_fg_end[param_i[0]] = ta
        else:
            params_fg_end[param_i[0]] = np.expand_dims(np.expand_dims(float(param_i[1]), axis=0), axis=0)

    if 'bg' in param_i[0]:
        param_i[0] = param_i[0].split('_')[0]
        if ('color' in param_i[0]) or ('tone' in param_i[0]):
            listparamsi = param_i[1].split(' ')
            listparamsi = [float(k) for k in listparamsi]
            if 'color' in param_i[0]:
                ca = np.array(listparamsi)[None, None, :, None]
                params_bg_end[param_i[0]] = np.reshape(ca, [1, 1, -1, 3])
            elif 'tone' in param_i[0]:
                ta = np.array(listparamsi)[None, None, :, None]
                params_bg_end[param_i[0]] = ta
        else:
            params_bg_end[param_i[0]] = np.expand_dims(np.expand_dims(float(param_i[1]), axis=0), axis=0)


