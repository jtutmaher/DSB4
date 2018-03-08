import os
import sys
import random
import warnings

import numpy as np
import pandas as pd

#import matplotlib.pyplot as plt

from PIL import Image, ImageEnhance

from tqdm import tqdm
from itertools import chain
from skimage.io import imread, imshow, imread_collection, concatenate_images
from skimage.transform import resize
from skimage.morphology import label

import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms

from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms


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


# Build U-Net model
class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(DownBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=0),
            nn.ReflectionPad2d(kernel_size//2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
            nn.Conv2d(out_channels, out_channels, kernel_size, padding=0),
            nn.ReflectionPad2d(kernel_size//2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )
    
    def forward(self, x):
        c = self.block(x)
        return c

class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=2, stride=2):
        super(UpBlock, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding=0)
        self.conv = DownBlock(2*out_channels, out_channels)
    
    def forward(self, x1, x2):
        x = torch.cat([self.up(x1), x2], dim=1)
        return self.conv(x)
        
class UNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet, self).__init__()
        self.dblock1 = DownBlock(n_channels, 64)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.dblock2 = DownBlock(64, 128)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.dblock3 = DownBlock(128, 256)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.dblock4 = DownBlock(256, 512)
        self.pool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.dblock5 = DownBlock(512, 1024)
        self.ublock5 = UpBlock(1024, 512)
        self.ublock4 = UpBlock(512, 256)
        self.ublock3 = UpBlock(256, 128)
        self.ublock2 = UpBlock(128, 64)
        self.output = nn.Conv2d(64, n_classes, kernel_size=1, groups=1, stride=1)
        
    def forward(self, x):
        c1 = self.dblock1(x)
        p1 = self.pool1(c1)
        c2 = self.dblock2(p1)
        p2 = self.pool1(c2)
        c3 = self.dblock3(p2)
        p3 = self.pool1(c3)
        c4 = self.dblock4(p3)
        p4 = self.pool1(c4)
        c5 = self.dblock5(p4)
        u5 = self.ublock5(c5, c4)
        u4 = self.ublock4(u5, c3)
        u3 = self.ublock3(u4, c2)
        u2 = self.ublock2(u3, c1)

        return self.output(u2)

# Run-length encoding stolen from https://www.kaggle.com/rakhlin/fast-run-length-encoding-python
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

def recon_loss(output, mask):
    probs = nn.Sigmoid()(output)
    intersection = torch.sum(probs[:,1,:,:]*mask.squeeze(1))
    return 1.0 - ((2.0*intersection+1.0)/(torch.sum(probs[:,1,:,:]) + torch.sum(mask) + 1.0))

if __name__=="__main__":

	# Set some parameters
	IMG_CHANNELS = 3
	TRAIN_PATH = '../../data/stage1_train/'
	TEST_PATH = '../../data/stage1_test/'

	warnings.filterwarnings('ignore', category=UserWarning, module='skimage')
	seed = 42
	random.seed = seed
	np.random.seed = seed

	# Get train and test IDs
	train_ids = next(os.walk(TRAIN_PATH))[1]
	test_ids = next(os.walk(TEST_PATH))[1]

	X_train = []
	Y_train = []
	print('Getting and resizing train images and masks ... ')
	sys.stdout.flush()
	for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):
        	path = TRAIN_PATH + id_
        	img = Image.fromarray(imread(path + '/images/' + id_ + '.png')[:,:,:IMG_CHANNELS])
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
        	path = TEST_PATH + id_
        	img = Image.open(path + '/images/' + id_ + '.png')
       		sizes_test.append([img.size[0], img.size[1]])
        	X_test.append(img)

	print('Done!')

	img_size = 128

	train_dataset = NucleusDataset(X_train, Y_train)
	train_dataloader = DataLoader(train_dataset, batch_size=24, shuffle=True)
	model = UNet(3, 2)
	use_cuda = True
	gpu_ids = [0,1]
	optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

	#testdataset = np.concatenate([np.asarray(x) for x in X_test],axis=0)
	#print(type(testdataset))
	#print(testdataset.shape)
	#inf_transforms = transforms.Compose([transforms.ToPILImage(),transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))])
	mean = float(np.mean([np.mean(np.array(x)[:,:,:3]) for x in X_test]))
	std = float(np.std([np.std(np.array(x)[:,:,:3]) for x in X_test]))

	# Predict on train, val and test
	n_epochs = 30
	for _ in tqdm(range(n_epochs)):
		for data in train_dataloader:
			train_dataset.gen_params(img_size)
			optimizer.zero_grad()
			X = Variable(data['X']).cuda()
			target = Variable(data['y']).cuda()
			model.cuda()
			#model = torch.nn.DataParallel(model, device_ids=gpu_ids)
			#cudnn.benchmark = True
			output = model(X)
			loss = recon_loss(output, target)
			print("loss: " + str(loss.data[0]))
			loss.backward()
			optimizer.step()

	#model = load_model('../models/dsbowl2018-1.h5')
	#preds_train = model.forward(X_train[:int(X_train.shape[0]*0.9)])
	#preds_val = model.forward(X_train[int(X_train.shape[0]*0.9):])
	preds_test = []
	for x in X_test:
		x = np.array(x)[:,:,:3].astype('float32')
		x -= mean
		x /= std
		x = np.transpose(x,(2,0,1))
		x = x[np.newaxis,:,:,:]
		tensor = torch.from_numpy(x)
		preds_temp = model(Variable(tensor).cuda())
		preds_test.append(preds_temp)
	#print(np.asarray(X_test).shape
	#testdataset = TestDataset(X_test)

	# Threshold predictions
	#preds_train_t = (preds_train > 0.5).astype(np.uint8)
	#preds_val_t = (preds_val > 0.5).astype(np.uint8)
	preds_test_t = (preds_test > 0.5).astype(np.uint8)

	# Create list of upsampled test masks
	preds_test_upsampled = []
	for i in range(len(preds_test)):
		preds_test_upsampled.append(resize(np.squeeze(preds_test[i]), 
                                       (sizes_test[i][0], sizes_test[i][1]), 
                                       mode='constant', preserve_range=True))

	new_test_ids = []
	rles = []
	for n, id_ in enumerate(test_ids):
    		rle = list(prob_to_rles(preds_test_upsampled[n]))
    		rles.extend(rle)
    		new_test_ids.extend([id_] * len(rle))

	# Create submission DataFrame
	sub = pd.DataFrame()
	sub['ImageId'] = new_test_ids
	sub['EncodedPixels'] = pd.Series(rles).apply(lambda x: ' '.join(str(y) for y in x))
	sub.to_csv('sub-dsbowl2018-1-pytorch.csv', index=False)
