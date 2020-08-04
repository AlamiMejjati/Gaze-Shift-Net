import numpy as np
import os
import cv2
import random
from operations_np import *
# shape1 = 240
# shape2 = 320

class CoCoLoader_rectangle:
    """ Produce images read from a list of files. """

    def __init__(self, files, main_dir, channel=3, resize=None, shuffle=True, random_s=True, shape1=224, shape2=320):
        """
        Args:
            files (list): list of file paths.
            channel (int): 1 or 3. Will convert grayscale to RGB images if channel==3.
                Will produce (h, w, 1) array if channel==1.
            resize (tuple): int or (h, w) tuple. If given, resize the image.
        """
        assert len(files), "No image files given to ImageFromFile!"
        self.files = files
        self.channel = int(channel)
        assert self.channel in [1, 3], self.channel
        self.imread_mode = cv2.IMREAD_GRAYSCALE if self.channel == 1 else cv2.IMREAD_COLOR
        self.resize = resize
        self.shuffle = shuffle
        self.indexes = list(range(len(self.files)))
        self.main_dir = main_dir
        self.random_s = random_s
        self.shape1 = shape1
        self.shape2 = shape2

    def __len__(self):
        return len(self.files)

    def __iter__(self):
        if self.shuffle:
            random.shuffle(self.files)
        for i in self.indexes:
            mf = self.files[i]
            m = cv2.imread(mf, cv2.IMREAD_GRAYSCALE)

            # filename = os.path.splitext(os.path.basename(mf))[0] + '.jpg'
            f = os.path.join(self.main_dir, os.path.basename(mf).split('_')[0]) + '.jpg'
            im = cv2.imread(f, self.imread_mode)
            assert im is not None, f
            if self.channel == 3:
                im = im[:, :, ::-1]
            if self.resize is not None:
                im = cv2.resize(im, tuple(self.resize[::-1]))
            if self.channel == 1:
                im = im[:, :, np.newaxis]


            if self.shuffle:
                imresized = cv2.resize(im, (int(self.shape2 * 1.12), int(self.shape1 * 1.12)))
                m = cv2.resize(m, (int(self.shape2 * 1.12), int(self.shape1 * 1.12)), interpolation=cv2.INTER_NEAREST)
                m = np.expand_dims(m, axis=-1)
                resized = np.concatenate([imresized, m], axis=-1)
                margin1 = np.floor(self.shape1 * 1.12 - self.shape1)
                margin2 = np.floor(self.shape2 * 1.12 - self.shape2)
                x = random.randint(0, margin1)
                y = random.randint(0, margin2)
                cropped = resized[x:x+self.shape1, y:y+self.shape2, :]
                # print('cropped:', np.unique(cropped[:, :, -1]))
                if random.uniform(0, 1) >0.5:
                    cropped = cv2.flip(cropped, 1)
                im = cropped[:, :, :-1]
                m = cropped[:, :, -1]
                # print(np.unique(template))
            else:
                im = cv2.resize(im, (self.shape2, self.shape1))
                m = cv2.resize(m, (self.shape2, self.shape1), interpolation=cv2.INTER_NEAREST)

            m = np.expand_dims(m, axis=-1)
            if self.random_s:
                deltasal = np.expand_dims(round(random.uniform(0.009, 0.1), 2), axis=-1)
            else:
                deltasal = np.expand_dims(0.02, axis=-1)

            minim = cv2.resize(m, (int(self.shape2/2), int(self.shape1/2)), interpolation=cv2.INTER_NEAREST)
            minim = np.expand_dims(minim, axis=-1)
            im = (im / 127.5) - 1
            m = (m / 127.5) - 1
            minim = (minim / 127.5) - 1
            yield (im, m, deltasal)

class CoCoLoader_rectangle_HR:
    """ Produce images read from a list of files. """

    def __init__(self, files, main_dir, channel=3, resize=None, shuffle=True, random_s=True, shape1=224, shape2=320):
        """
        Args:
            files (list): list of file paths.
            channel (int): 1 or 3. Will convert grayscale to RGB images if channel==3.
                Will produce (h, w, 1) array if channel==1.
            resize (tuple): int or (h, w) tuple. If given, resize the image.
        """
        assert len(files), "No image files given to ImageFromFile!"
        self.files = files
        self.channel = int(channel)
        assert self.channel in [1, 3], self.channel
        self.imread_mode = cv2.IMREAD_GRAYSCALE if self.channel == 1 else cv2.IMREAD_COLOR
        self.resize = resize
        self.shuffle = shuffle
        self.indexes = list(range(len(self.files)))
        self.main_dir = main_dir
        self.random_s = random_s
        self.shape1 = shape1
        self.shape2 = shape2

    def __len__(self):
        return len(self.files)

    def __iter__(self):
        if self.shuffle:
            random.shuffle(self.files)
        for i in self.indexes:
            mf = self.files[i]
            m = cv2.imread(mf, cv2.IMREAD_GRAYSCALE)
            m_hr = m

            # filename = os.path.splitext(os.path.basename(mf))[0] + '.jpg'
            f = os.path.join(self.main_dir, os.path.basename(mf).split('_')[0]) + '.jpg'
            im = cv2.imread(f, self.imread_mode)
            im_hr = im

            assert im is not None, f
            if self.channel == 3:
                im = im[:, :, ::-1]
                im_hr = im_hr[:, :, ::-1]
            if self.resize is not None:
                im = cv2.resize(im, tuple(self.resize[::-1]))
            if self.channel == 1:
                im = im[:, :, np.newaxis]


            if self.shuffle:
                imresized = cv2.resize(im, (int(self.shape2 * 1.12), int(self.shape1 * 1.12)))
                m = cv2.resize(m, (int(self.shape2 * 1.12), int(self.shape1 * 1.12)), interpolation=cv2.INTER_NEAREST)
                m = np.expand_dims(m, axis=-1)
                resized = np.concatenate([imresized, m], axis=-1)
                margin1 = np.floor(self.shape1 * 1.12 - self.shape1)
                margin2 = np.floor(self.shape2 * 1.12 - self.shape2)
                x = random.randint(0, margin1)
                y = random.randint(0, margin2)
                cropped = resized[x:x+self.shape1, y:y+self.shape2, :]
                # print('cropped:', np.unique(cropped[:, :, -1]))
                if random.uniform(0, 1) >0.5:
                    cropped = cv2.flip(cropped, 1)
                im = cropped[:, :, :-1]
                m = cropped[:, :, -1]
                # print(np.unique(template))
            else:
                im = cv2.resize(im, (self.shape2, self.shape1))
                m = cv2.resize(m, (self.shape2, self.shape1), interpolation=cv2.INTER_NEAREST)

            m = np.expand_dims(m, axis=-1)
            m_hr = np.expand_dims(m_hr, axis=-1)
            if self.random_s:
                deltasal = np.expand_dims(round(random.uniform(0.009, 0.1), 2), axis=-1)
            else:
                deltasal = np.expand_dims(0.02, axis=-1)

            minim = cv2.resize(m, (int(self.shape2/2), int(self.shape1/2)), interpolation=cv2.INTER_NEAREST)
            minim = np.expand_dims(minim, axis=-1)
            im = (im / 127.5) - 1
            m = (m / 127.5) - 1
            minim = (minim / 127.5) - 1
            im_hr = (im_hr / 127.5) - 1
            m_hr = m_hr / 255.0
            yield (im, m, im_hr, m_hr, deltasal)


