import tkinter
import tensorflow as tf
from operations import apply_transformations_sharp_exp_cont_tone_color
import cv2
from PIL import Image, ImageTk
import io
import PySimpleGUI as sg
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, FigureCanvasAgg
from matplotlib.figure import Figure
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--frozen_model", default="results/frozen_model.pb", type=str,
                    help="Frozen model file to import")
args = parser.parse_args()

tf.enable_eager_execution()
# sg.theme('DarkBlue1')
# sg.ChangeLookAndFeel('Material2')
sg.ChangeLookAndFeel('Black')
sg.SetOptions(scrollbar_color='#D8D7D7')
fontSize = 14
L = 8
SIZE_X = 1
SIZE_Y = 1
UIfont = 'courier'
NUMBER_MARKER_FREQUENCY = 8

iminit = './datasets/example_data/ims'
minit = './datasets/example_data/masks'
frozen_model_filename = args.frozen_model
layout = [[sg.Text("Upload image:")], [sg.Input(key='im'), sg.FilesBrowse(initial_folder=iminit)],[sg.Text("Upload mask:")],[sg.Input(key='mask'), sg.FilesBrowse(initial_folder=minit)], [sg.OK(), sg.Cancel()]]
window0 = sg.Window('Select files').Layout(layout)
event, values = window0.Read()
im_name = os.path.basename(values['im'])
mask_name = os.path.basename(values['mask'])
window0.close()
# im_name = '/home/yam28/Documents/phdYoop/datasets/patchbased/saliency_shift_18/018_in.jpg'
# mask_name = '/home/yam28/Documents/phdYoop/datasets/patchbased/saliency_shift_18/masks/018_mask.jpg'
# values= {'im':im_name, 'mask':mask_name}
# im_id = int(im_name.split('.')[0][2:])


def draw_axis(graph):
    graph.draw_line((0, 0), (0, 1))                # axis lines
    graph.draw_line((0, 0), (1, 0))

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

def get_img_data(f, maxsize=(200, 200), first=True):
    """Generate image data using PIL
    """
    img = Image.open(f)
    img.thumbnail(maxsize)
    if first:                     # tkinter is inactive the first time
        bio = io.BytesIO()
        img.save(bio, format="PNG")
        del img
        return bio.getvalue()
    return ImageTk.PhotoImage(img)

def iterator(im_path, m_path):
    np_im_hr = cv2.imread(im_path)[:,:,::-1]/127.5 -1
    np_im = cv2.resize(np_im_hr, (320,240))
    np_im_hr = np_im_hr[None, :,:,:]
    np_im = np_im[None, :,:,:]
    np_m_hr = cv2.imread(m_path, cv2.IMREAD_GRAYSCALE)/255.
    # np_m_hr = 1- np_m_hr
    np_m = (cv2.resize(np_m_hr, (320, 240))-0.5)*2
    np_m = np_m[None, :, :, None]
    np_m_hr = np_m_hr[None, :, :, None]
    return np_im, np_m

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

graph = load_graph(frozen_model_filename)

im = graph.get_tensor_by_name('IteratorGetNext:0')
mask = graph.get_tensor_by_name('IteratorGetNext:1')

fgparams = {'exposure': graph.get_tensor_by_name('gen/inc_fg_param_spitter/mul_1:0'),
            'contrast': graph.get_tensor_by_name('gen/inc_fg_param_spitter/contprediction/Tanh:0'),
            'sharp': graph.get_tensor_by_name('gen/inc_fg_param_spitter/mul:0'),
            'tone': graph.get_tensor_by_name('gen/inc_fg_param_spitter/strided_slice_1:0'),
            'color': graph.get_tensor_by_name('gen/inc_fg_param_spitter/Reshape:0')}
bgparams = {'exposure': graph.get_tensor_by_name('gen/inc_bg_param_spitter/mul_1:0'),
            'contrast': graph.get_tensor_by_name('gen/inc_bg_param_spitter/contprediction/Tanh:0'),
            'sharp': graph.get_tensor_by_name('gen/inc_bg_param_spitter/mul:0'),
            'tone': graph.get_tensor_by_name('gen/inc_bg_param_spitter/strided_slice_1:0'),
            'color': graph.get_tensor_by_name('gen/inc_bg_param_spitter/Reshape:0')}

with tf.Session(graph=graph) as sess:
    np_im, np_m = iterator(values['im'], values['mask'])
    params_fg_end, params_bg_end = sess.run([fgparams, bgparams], feed_dict={im: np_im, mask: np_m})

params_fg_begin, params_bg_begin = initialize_params_gui(params_fg_end, params_bg_end)

im = cv2.imread(values['im'])  # create PIL image from frame+
mask = cv2.imread(values['mask'], cv2.IMREAD_GRAYSCALE)

if (im.shape[0]>1000) or (im.shape[1]>1000):
    mask = cv2.resize(mask, (int(im.shape[1] / 4), int(im.shape[0] / 4)))
    im = cv2.resize(im, (int(im.shape[1]/4), int(im.shape[0]/4)))

imgbytes = cv2.imencode('.png', im)[1].tobytes()

# im = Image.open(testpath)
displayImage = sg.Image(data=imgbytes,key="imageContainer", pad=((0,0),(0,0)))
Titlesal = sg.Text("Attention Shifter", size=(20, 0), font=(UIfont, 14), pad=((0,0),(0,0)))
sharpness_text = sg.Text("S", size=(1, 0), font=(UIfont, 10))
exposure_text = sg.Text("E", size=(1, 0), font=(UIfont, 10))
cont_text = sg.Text("C", size=(1, 0), font=(UIfont, 10))
empty_param = sg.Text(" ", size=(1, 0), font=(UIfont, 10))
tone_color_text = sg.Text("Tone & Color Curves", size=(20, 0), font=(UIfont, 10))
empty_text = sg.Text(" ", size=(20, 0), font=(UIfont, 10))

main_slider = sg.Slider(range=(0,1), orientation='h', resolution=0.01, size=(int(mask.shape[1]/23),10), change_submits=True, key='mainslider',disable_number_display=True)
sharp_slider_fg = sg.Slider(range=(-2,2), orientation='h', resolution=0.01, size=(10,10), change_submits=True, key='sharp_fg',disable_number_display=True)
exposure_slider_fg = sg.Slider(range=(-3,3), orientation='h', resolution=0.01, size=(10,10), change_submits=True, key='exposure_fg',disable_number_display=True)
contrast_slider_fg = sg.Slider(range=(-1,1), orientation='h', resolution=0.01, size=(10,10), change_submits=True, key='contrast_fg',disable_number_display=True)
tone_fg = [[sg.Slider(range=(0,3), orientation='h', resolution=0.01, size=(20,10), change_submits=True, key='tone_fg_%d'%k,disable_number_display=False)] for k in range(L)]
color_fg = [[sg.Slider(range=(0,3), orientation='h', resolution=0.01, size=(6,10), change_submits=True, key='color_fg_%d_%d'%(k,j),disable_number_display=False) for k in range(3)]for j in range(L)]
graph_fg = sg.Graph(canvas_size=(128, 128),
                 graph_bottom_left=(0, 0),
                 graph_top_right=(1, 1),
                 background_color='#D8D7D7',
                 key='graph_fg')

sharp_slider_bg = sg.Slider(range=(-2,2), orientation='h', resolution=0.01, size=(10,10), change_submits=True, key='sharp_bg',disable_number_display=True)
exposure_slider_bg = sg.Slider(range=(-3,3), orientation='h', resolution=0.01, size=(10,10), change_submits=True, key='exposure_bg',disable_number_display=True)
contrast_slider_bg = sg.Slider(range=(-1,1), orientation='h', resolution=0.01, size=(10,10), change_submits=True, key='contrast_bg',disable_number_display=True)
tone_bg = [[sg.Slider(range=(0,3), orientation='h', resolution=0.01, size=(20,10), change_submits=True, key='tone_bg_%d'%k,disable_number_display=False)] for k in range(L)]
color_bg = [[sg.Slider(range=(0,3), orientation='h', resolution=0.01, size=(6,10), change_submits=True, key='color_bg_%d_%d'%(k,j),disable_number_display=False) for k in range(3)]for j in range(L)]
graph_bg = sg.Graph(canvas_size=(128, 128),
                 graph_bottom_left=(0, 0),
                 graph_top_right=(1, 1),
                 background_color='#D8D7D7',
                 key='graph_bg')

layout = [[empty_text],
          [displayImage],
          # [Titlesal],
          [sg.Frame('Saliency Slider',[[main_slider]], font=(UIfont, 10), pad=((int(mask.shape[1]/4),0),(10,10)))],
          [sg.Frame('Foreground Parameters',
              [
                  # [gamma_text, gamma_slider_fg],
                  [sharpness_text, sharp_slider_fg],
                  # [WB_text, wb_slider_fg],
                  [exposure_text, exposure_slider_fg],
                  # [sat_text, saturation_slider_fg],
                  [cont_text, contrast_slider_fg],
                  [tone_color_text],
                  [graph_fg]
                  # [blur_text, blur_slider_fg]
                  # [sg.Frame('Tone adjustement Parameters', tone_fg), sg.Frame('Color adjustement Parameters', color_fg)]
              ], font=(UIfont, 10), pad=((int(mask.shape[1]/4),0),(10,10))),
           sg.Frame('Background Parameters',
                    [
                         # [gamma_text, gamma_slider_bg],
                         [sharpness_text,sharp_slider_bg],
                         # [WB_text, wb_slider_bg],
                         [exposure_text, exposure_slider_bg],
                         # [sat_text, saturation_slider_bg],
                         [cont_text, contrast_slider_bg],
                         [tone_color_text],
                         [graph_bg]
                         # [blur_text, blur_slider_bg]
                         # [sg.Frame('Tone adjustement Parameters', tone_bg), sg.Frame('Color adjustement Parameters', color_bg)]
                    ], font=(UIfont, 10))
           ], [empty_param]
          ]


sz = fontSize
window = sg.Window("Saliency Editor", layout, grab_anywhere=False, finalize=True, font=UIfont)


f1 = (1 / 8) * np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
f2 = (1 / 8) * np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
tf1 = tf.expand_dims(tf.stack([tf.constant(f1, dtype=tf.float32)] * 3, axis=-1), -1)
tf2 = tf.expand_dims(tf.stack([tf.constant(f2, dtype=tf.float32)] * 3, axis=-1), -1)

nb_ticks = 8
aa_init = np.array(np.arange(0, 1, 1/nb_ticks).tolist() + [1,1])
aa = aa_init[:,None]
aa = aa.reshape([int(nb_ticks/2) +1 ,2])
aa = aa[None, :, :, None]
aa_tone = aa.astype(np.float32)
aa_color = np.concatenate([aa,aa,aa], axis=-1).astype(np.float32)


counter= 0
while True:
    event, values = window.Read()
    if event is None:
        break
    graph_fg.erase()
    graph_bg.erase()
    # draw_axis(graph_fg)
    if event=='mainslider' or counter ==0:
        mainval = values['mainslider']
        new_params_fg, new_params_bg = update_params_gui_tf(params_fg_begin, params_bg_begin, params_fg_end, params_bg_end, mainval)
        counter+=1

    bb_tone_fg = apply_tone_curve_adjustment(aa_tone, new_params_fg['tone']).numpy().flatten()
    bb_color_fg = apply_color_curve_adjustment(aa_color, new_params_fg['color']).numpy().reshape([-1,3])
    bb_tone_bg = apply_tone_curve_adjustment(aa_tone, new_params_bg['tone']).numpy().flatten()
    bb_color_bg = apply_color_curve_adjustment(aa_color, new_params_bg['color']).numpy().reshape([-1,3])

    prev_x_tone_fg = prev_y_tone_fg = None
    prev_x_red_fg = prev_y_red_fg = None
    prev_x_blue_fg = prev_y_blue_fg = None
    prev_x_green_fg = prev_y_green_fg = None
    for h in range(len(bb_tone_fg)):
        if prev_x_tone_fg is not None:
            graph_fg.draw_line((aa_init[h], bb_tone_fg[h]), (prev_x_tone_fg, prev_y_tone_fg), color='black', width=2)
            graph_fg.draw_point((aa_init[h], bb_tone_fg[h]), color='black', size=2)
        if prev_x_red_fg is not None:
            graph_fg.draw_line((aa_init[h], bb_color_fg[h,0]), (prev_x_red_fg, prev_y_red_fg), color='red', width=2)
            graph_fg.draw_point((aa_init[h], bb_color_fg[h,0]), color='red', size=2)
        if prev_x_green_fg is not None:
            graph_fg.draw_line((aa_init[h], bb_color_fg[h,1]), (prev_x_green_fg, prev_y_green_fg), color='green', width=2)
            graph_fg.draw_point((aa_init[h], bb_color_fg[h, 1]), color='green', size=2)
        if prev_x_blue_fg is not None:
            graph_fg.draw_line((aa_init[h], bb_color_fg[h,2]), (prev_x_blue_fg, prev_y_blue_fg), color='blue', width=2)
            graph_fg.draw_point((aa_init[h], bb_color_fg[h, 2]), color='blue', size=2)

        prev_x_tone_fg = aa_init[h]
        prev_y_tone_fg = bb_tone_fg[h]
        prev_x_red_fg = aa_init[h]
        prev_y_red_fg = bb_color_fg[h, 0]
        prev_x_green_fg = aa_init[h]
        prev_y_green_fg = bb_color_fg[h, 1]
        prev_x_blue_fg = aa_init[h]
        prev_y_blue_fg = bb_color_fg[h, 2]



    prev_x_tone_bg = prev_y_tone_bg = None
    prev_x_red_bg = prev_y_red_bg = None
    prev_x_blue_bg = prev_y_blue_bg = None
    prev_x_green_bg = prev_y_green_bg = None
    for h in range(len(bb_tone_bg)):
        if prev_x_tone_bg is not None:
            graph_bg.draw_line((aa_init[h], bb_tone_bg[h]), (prev_x_tone_bg, prev_y_tone_bg), color='black', width=2)
            graph_bg.draw_point((aa_init[h], bb_tone_bg[h]), color='black', size=2)
        if prev_x_red_bg is not None:
            graph_bg.draw_line((aa_init[h], bb_color_bg[h,0]), (prev_x_red_bg, prev_y_red_bg), color='red', width=2)
            graph_bg.draw_point((aa_init[h], bb_color_bg[h, 0]), color='red', size=2)
        if prev_x_green_bg is not None:
            graph_bg.draw_line((aa_init[h], bb_color_bg[h,1]), (prev_x_green_bg, prev_y_green_bg), color='green', width=2)
            graph_bg.draw_point((aa_init[h], bb_color_bg[h, 1]), color='green', size=2)
        if prev_x_blue_bg is not None:
            graph_bg.draw_line((aa_init[h], bb_color_bg[h,2]), (prev_x_blue_bg, prev_y_blue_bg), color='blue', width=2)
            graph_bg.draw_point((aa_init[h], bb_color_bg[h, 2]), color='blue', size=2)

        prev_x_tone_bg = aa_init[h]
        prev_y_tone_bg = bb_tone_bg[h]
        prev_x_red_bg = aa_init[h]
        prev_y_red_bg = bb_color_bg[h, 0]
        prev_x_green_bg = aa_init[h]
        prev_y_green_bg = bb_color_bg[h, 1]
        prev_x_blue_bg = aa_init[h]
        prev_y_blue_bg = bb_color_bg[h, 2]


    if event != 'mainslider':
        if 'fg' in event:
            new_params_fg[event.split('_')[0]] = new_params_fg[event.split('_')[0]]*0 + values[event]
        elif 'bg' in event:
            new_params_bg[event.split('_')[0]] = new_params_bg[event.split('_')[0]] * 0 + values[event]

    new_im = apply_transformations_sharp_exp_cont_tone_color(im[None,:,:,::-1]/255., mask[None,:,:,None]/255., new_params_fg, new_params_bg, tf1, tf2)
    # new_im = np.ones_like(img)*mainval*255
    # new_im = (new_im.numpy() +1)*0.5
    new_im = new_im.numpy()
    newimgbytes = cv2.imencode('.png', new_im.squeeze()[:,:,::-1]*255)[1].tobytes()
    image = window.FindElement('imageContainer')
    image.Update(data=newimgbytes)

    for k in list(new_params_fg.keys()):

        if k+ '_fg' in list(window.AllKeysDict.keys()):
            window.FindElement(k+'_fg').Update(new_params_fg[k].numpy().squeeze())

    for k in list(new_params_bg.keys()):

        if k + '_bg' in list(window.AllKeysDict.keys()):
            window.FindElement(k+'_bg').Update(new_params_bg[k].numpy().squeeze())

print("Done.")