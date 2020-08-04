from pycocotools.coco import COCO
import numpy as np
import argparse
import os
from PIL import Image
import tensorflow as tf
import tensorflow.keras.backend as K
import os
import cv2
import random
from tqdm import tqdm
import argparse
# os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


np.random.seed(0)

parser = argparse.ArgumentParser()

parser.add_argument(
    '--path', required=True,
    help='the path that contains raw coco JPEG images')
parser.add_argument('--salmodelpath', required=True, help='path for the saliency model')
args = parser.parse_args()
saliency_model_path = args.salmodelpath
dataDir = args.path

mask_dir_inc = './datasets/CoCoClutter/increase/maskdir'
if not os.path.exists(mask_dir_inc):
    os.makedirs(mask_dir_inc)

# seg_dir = './segdir'
# if not os.path.exists(seg_dir):
#     os.makedirs(seg_dir)

mode = 'train'


dataType = mode + '2017'
annFile ='{}/annotations/instances_{}.json'.format(dataDir,dataType)

# initialize COCO api for instance annotations
coco=COCO(annFile)

# display COCO categories and supercategories
cats = coco.loadCats(coco.getCatIds())
nms=[cat['name'] for cat in cats]
print('COCO categories: \n{}\n'.format(' '.join(nms)))

nms_sc = set([cat['supercategory'] for cat in cats])
print('COCO supercategories: \n{}'.format(' '.join(nms_sc)))

# get all images containing given categories, select one at random
catIds = coco.getCatIds(catNms=nms)
nb_cats = max(catIds)
# map_cats = (((np.array(catIds)+100)/nb_cats) *255).astype(np.int).tolist()
map_cats = (np.array(catIds) + (255 - nb_cats)).tolist()

segDict = dict(zip(catIds, map_cats))
imgIds=[]
for i in catIds:
    imgIds += coco.getImgIds(catIds=i)
imgIds = list(dict.fromkeys(imgIds))
nb_images = len(imgIds)
size = 320, 240
bsize = size[0]*2, size[1]*2
counter = 0
model = tf.keras.models.load_model(saliency_model_path)
model._make_predict_function()

# while counter < nb_images:

for counter in tqdm(range(0,nb_images), desc='image'):
    imId = imgIds.pop(np.random.randint(0, len(imgIds)))
    img = coco.loadImgs(imId)[0]
    annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
    anns = coco.loadAnns(annIds)
    counteri = 0
    if len(anns) < 3:
        continue
    name = str(anns[0]['image_id'])
    namei = []
    maski = []
    for i in range(len(anns)):
        mask = coco.annToMask(anns[i])
        if i==0:
            insseg = np.round(mask) * anns[i]['category_id'] #segDict[anns[i]['category_id']]
        else:
            insseg += np.round(mask) * anns[i]['category_id'] #segDict[anns[i]['category_id']]
        ratio = mask.sum()/np.size(mask)
        if (ratio > 0.4):
            continue
        if (ratio < 0.03):
            continue

        minim = cv2.resize(mask, bsize, interpolation=cv2.INTER_NEAREST)
        im = Image.open(os.path.join(dataDir, dataType, img['file_name']))
        if im.mode != 'RGB':
            continue
        im = np.array(im.resize(size))
        im = im[None, :, :, :]
        im[:, :, :, 0] = im[:, :, :, 0] - 103.939
        im[:, :, :, 1] = im[:, :, :, 1] - 116.779
        im[:, :, :, 2] = im[:, :, :, 2] - 123.68
        salmap = model.predict(im)[0, 0, :, :, 0]
        salmap = (salmap - np.min(salmap)) / (np.max(salmap) - np.min(salmap))
        if np.mean(salmap * minim) * (np.size(minim) / np.sum(minim)) > 0.7:
            continue

        namei.append('0'*(12 - len(name)) + name + '_%d.jpg'%counteri)
        maski.append(mask*255)
        counteri += 1

    if len(maski)>1:
        pick = random.randint(0, len(maski)-1)
        Image.fromarray(maski[pick]).save(os.path.join(mask_dir_inc, namei[pick]))
        counter +=1
        # Image.fromarray(insseg).save(os.path.join(seg_dir, '0'*(12 - len(name)) + name +'.jpg'))

