import os
import sys
import random
import warnings

import numpy as np
from PIL import Image, ImageEnhance
from skimage.io import imread, imshow, imread_collection, concatenate_images

import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms

from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

img_size=128

# create random crop class that holds parameters until called to switch
class RandomCrop(object):
    def __init__(self, size):
        self.size = size
        self.i = 0
        self.j = 0
        
    def gen_params(self, img_size):
        if self.size >= img_size:
            self.i = 0
            self.j = 0
        else:
            self.i = random.randint(0, img_size-self.size)
            self.j = random.randint(0, img_size-self.size)
    
    def __call__(self, img):
        return img.crop((self.i, self.j, self.i+self.size, self.j+self.size))

class RandomRotateAndFlip(object):
    def __init__(self, max_angle):
        self.max_angle = max_angle
        self.angle = None
        self.flip_horizontal = False
        self.flip_vertical = False
        
    def gen_params(self):
        self.angle = np.random.uniform(-self.max_angle, self.max_angle)
        if np.random.uniform() < 0.5:
            self.flip_horizontal = True
        else:
            self.flip_horizontal = False
        if np.random.uniform() < 0.5:
            self.flip_vertical = True
        else:
            self.flip_vertical = False
        
        
    def __call__(self, img):
        if self.angle is None:
            self.gen_params()
        return_img = img.rotate(self.angle, False, False, None)
        if self.flip_horizontal:
            return_img = return_img.transpose(Image.FLIP_LEFT_RIGHT)
        if self.flip_vertical:
            return_img = return_img.transpose(Image.FLIP_TOP_BOTTOM)
        return return_img

class RandomColorJitter(object):
    def __init__(self, brightness=0.1, contrast=0.1, saturation=0.1):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.brightness_factor = None
        self.contrast_factor = None
        self.saturation_factor = None
    
    def gen_params(self):
        self.brightness_factor = np.random.uniform(max(0, 1.0-self.brightness), 1.0+self.brightness)
        self.contrast_factor = np.random.uniform(max(0, 1.0-self.contrast), 1.0+self.contrast)
        self.saturation_factor = np.random.uniform(max(0, 1.0-self.saturation), 1.0+self.saturation)
        
    def __call__(self, img):
        if self.brightness_factor is None:
            self.gen_params()
        brightness_enhancer = ImageEnhance.Brightness(img)
        return_img = brightness_enhancer.enhance(self.brightness_factor)
        contrast_enhancer = ImageEnhance.Contrast(return_img)
        return_img = contrast_enhancer.enhance(self.contrast_factor)
        color_enhancer = ImageEnhance.Color(return_img)
        return_img = color_enhancer.enhance(self.saturation_factor)
        return return_img

class NucleusDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.crop = RandomCrop(112)
        self.rot_flip = RandomRotateAndFlip(180)
        self.jitter_factor = 0.2
        self.color_jitter = RandomColorJitter(self.jitter_factor, 
                                              self.jitter_factor, 
                                              self.jitter_factor)
        self.X_transforms = transforms.Compose([transforms.Scale(img_size),
                                                self.crop, 
                                                self.rot_flip, 
                                                self.color_jitter, 
                                                transforms.ToTensor(),
                                                transforms.Normalize((0.5, 0.5, 0.5), 
                                                                     (0.5, 0.5, 0.5))])
        self.Y_transforms = transforms.Compose([transforms.Scale(img_size),
                                                self.crop, 
                                                self.rot_flip, 
                                                transforms.ToTensor()])

    def __getitem__(self, index):
        #self.gen_params(self.X[index])
        return {'X': self.X_transforms(self.X[index]), 
                'y': self.Y_transforms(self.y[index])}
    
    def __len__(self):
        return len(self.X)
    
    def gen_params(self, img):
        self.crop.gen_params(img)
        self.rot_flip.gen_params()
        self.color_jitter.gen_params()

class TestDataset(Dataset):
    def __init__(self,X):
        self.X = X
        self.X_transforms = transforms.Compose([transforms.Scale(img_size), 
                                                transforms.ToTensor(),
                                                transforms.Normalize((0.5, 0.5, 0.5), 
                                                                     (0.5, 0.5, 0.5))])

    def __getitem__(self,index):
        return {'X':self.X_transforms(self.X[index])}

    def __len__(self):
        return len(self.X)


def unnorm(x):
    un_x = 255*(x*0.5+0.5)
    return un_x.astype(np.uint8)