import numpy as np
import cv2
import os
import pathlib
import imageio
from skimage.color import rgb2gray
from skimage.filters import threshold_otsu
from scipy.misc import imfilter
from random import shuffle

def split_data(arr,dim=[4,4]):
	arr = np.array(np.split(arr,dim[0]))
	res = []
	for data in arr:
		res.extend(np.split(data,dim[1],axis=1))
	return np.array(res)

def augment(ims,labels):
	pass

def check_invert_contrast(im):
    hist = np.histogram(im)[0]
    score = np.sum(hist[-3:])
    total = np.sum(hist)
    if score/float(total)>0.15:
        return imfilter((1-im),'edge_enhance')
    else:
        return imfilter(im,'edge_enhance')

def load_train_data(im_dir,dim=(4,4)):
	# Load Images
	print("-- Reading Images")
	training_paths = pathlib.Path(im_dir).glob('*/images/*.png')
	training_sorted = sorted([x for x in training_paths])
	train_ims = np.array([rgb2gray(imageio.imread(str(x)))[:256,:256] for x in training_sorted])
	train_ims = np.array([check_invert_contrast(train_ims[x]) for x in range(train_ims.shape[0])])
	#Augment Code Added 1/16
	train_ims_blurred = np.array([imfilter(train_ims[x],'blur') for x in range(train_ims.shape[0])])
	train_ims = np.concatenate((train_ims,train_ims_blurred),axis=0)
	downsample = np.concatenate([split_data(train_ims[x],dim=dim) for x in range(train_ims.shape[0])],axis=0)
	downsample_ims = downsample.reshape((downsample.shape[0],downsample.shape[1],downsample.shape[2],1))
	print downsample_ims.shape

	# Load Labels
	print("-- Reading Labels")
	label_paths = pathlib.Path(im_dir).glob('*/masks/')
	label_sorted = sorted([x for x in label_paths])
	labels = np.array([np.sum([rgb2gray(imageio.imread(str(x))) for x in pathlib.Path(y).glob('*.png')],axis=0)[:256,:256] for y in label_sorted])
	#For Augment Added 1/16
	labels2 = np.copy(labels)
	labels = np.concatenate((labels,labels2),axis=0)
	downsample_l = np.concatenate([split_data(labels[x],dim=dim) for x in range(labels.shape[0])],axis=0)
	downsample_labs = downsample_l.reshape((downsample_l.shape[0],downsample_l.shape[1],downsample_l.shape[2],1))
	print downsample_labs.shape
	
	# Add Zero Arrays
	add_zero_ims = np.zeros((int(0.15*downsample.shape[0]),downsample.shape[1],downsample.shape[2],1))
	add_zero_labs = np.zeros((int(0.15*downsample.shape[0]),downsample.shape[1],downsample.shape[2],1))
	final_ims = np.concatenate((downsample_ims,add_zero_ims),axis=0).astype('float32')
	final_labs = np.concatenate((downsample_labs,add_zero_labs),axis=0).astype('float32')

	cand = list(np.linspace(0,final_ims.shape[0]-1,final_ims.shape[0]))
	idx = shuffle([int(x) for x in cand])
	shuffle_ims = final_ims[idx,:,:,:]
	shuffle_labs = final_labs[idx,:,:,:]
	return shuffle_ims.reshape(final_ims.shape),shuffle_labs.reshape(final_labs.shape)
	
	#return final_ims,final_labs


if __name__=="__main__":
	DIR = "./data/stage1_train/"
	size = (4,4)
	images, labels = load_train_data(DIR,dim=size)
	print("Saving Data")
	print("Final Size: "+str(images.shape)+", "+str(labels.shape))
	np.savez_compressed("./data.npz",images=images,masks=labels)