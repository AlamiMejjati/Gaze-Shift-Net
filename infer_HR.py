from utils import *
from PIL import Image
import time
import glob
import os

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
    return np_im_hr, np_im, np_m_hr, np_m

if __name__ == '__main__':
    # Let's allow the user to pass the filename as an argument
    parser = argparse.ArgumentParser()
    parser.add_argument("--frozen_model", default="results/frozen_model.pb", type=str,
                        help="Frozen model file to import")
    args = parser.parse_args()


    result_folder = os.path.join(os.path.dirname(args.frozen_model), 'results_from_frozen')
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)
    # We use our "load_graph" function
    graph = load_graph(args.frozen_model)

    # We can verify that we can access the list of operations in the graph
    for op in graph.get_operations():
        print(op.name)
        # prefix/Placeholder/inputs_placeholder
        # ...
        # prefix/Accuracy/predictions

    # We access the input and output nodes
    #TODO: Give your nodes proper names
    im = graph.get_tensor_by_name('IteratorGetNext:0')
    mask = graph.get_tensor_by_name('IteratorGetNext:1')
    im_hr = graph.get_tensor_by_name('IteratorGetNext:2')
    mask_hr = graph.get_tensor_by_name('IteratorGetNext:3')
    # s = graph.get_tensor_by_name('IteratorGetNext:4')
    im_f = graph.get_tensor_by_name('clip_by_value:0')
    # init_sal = graph.get_tensor_by_name('snet_1/downsampler_1/conv2d/Conv2D:0')
    # im_f = graph.get_tensor_by_name('gen/mul_6:0')



    # load image + mask
    m_path = './datasets/example_data/masks'
    im_path = './datasets/example_data/ims'
    ms = sorted(glob.glob(os.path.join(m_path, '*.png')))
    ims = sorted(glob.glob(os.path.join(im_path, '*.png')))



    # We launch a Session
    it=0
    with tf.Session(graph=graph) as sess:
        for mp, imp in zip(ms, ims):
            np_im_hr, np_im, np_m_hr, np_m = iterator(imp, mp)
            im_final = sess.run(im_f, feed_dict={im: np_im, im_hr:np_im_hr, mask_hr:np_m_hr, mask: np_m})
            im_final = (im_final.squeeze()+1)*0.5*255.
            im_final = im_final.astype(np.uint8)
            cv2.imwrite(os.path.join(result_folder, 'im%d.png'%it), im_final[:,:,::-1])
            it+=1




