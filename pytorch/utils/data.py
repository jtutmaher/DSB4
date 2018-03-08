import os
import sys
import random
import warnings

import numpy as np
import pandas as pd

from PIL import Image, ImageEnhance

from tqdm import tqdm
from itertools import chain
from skimage.io import imread, imshow, imread_collection, concatenate_images
from skimage.transform import resize

def rle_encoding(x):
    dots = np.where(x.T.flatten() == 1)[0]
    run_lengths = []
    prev = -2
    for b in dots:
        if (b>prev+1): run_lengths.extend((b + 1, 0))
        run_lengths[-1] += 1
        prev = b
    return run_lengths

def prob_to_rles(x, cutoff=0.5):
    lab_img = label(x > cutoff)
    for i in range(1, lab_img.max() + 1):
        yield rle_encoding(lab_img == i)

def load_data(train_dir,test_dir,channels=3):
    warnings.filterwarnings('ignore', category=UserWarning, module='skimage')
    seed = 42
    random.seed = seed
    np.random.seed = seed

    # Get train and test IDs
    train_ids = next(os.walk(train_dir))[1]
    test_ids = next(os.walk(test_dir))[1]

    X_train = []
    Y_train = []
    print('Getting and resizing train images and masks ... ')
    sys.stdout.flush()
    for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):
            path = train_dir + id_
            img = Image.fromarray(imread(path + '/images/' + id_ + '.png')[:,:,:channels])
            w, h = img.size
            X_train.append(img)
            mask = np.zeros((h, w), dtype=np.uint8)
            for mask_file in next(os.walk(path + '/masks/'))[2]:
                    mask_ = imread(path + '/masks/' + mask_file)
                    mask = np.maximum(mask, mask_)
            Y_train.append(Image.fromarray(mask))
    
    # Get and resize test images
    X_test = []
    sizes_test = []
    print('Getting and resizing test images ... ')
    sys.stdout.flush()
    for n, id_ in tqdm(enumerate(test_ids), total=len(test_ids)):
            path = test_dir + id_
            img = Image.open(path + '/images/' + id_ + '.png')
            sizes_test.append([img.size[0], img.size[1]])
            X_test.append(img)

    print('Done!')
    return X_train,Y_train,X_test