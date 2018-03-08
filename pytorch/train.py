import os
import sys
import random
import warnings

import numpy as np

import utils.data as data
import utils.augment as augment
import models.unet as unet

from tqdm import tqdm

import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms

from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms


if __name__=="__main__":
	# Set some parameters
	#IMG_CHANNELS = 3
	TRAIN_PATH = '../data/stage1_train/'
	TEST_PATH = '../data/stage1_test/'

	
	#seed = 42
	#random.seed = seed
	#np.random.seed = seed

	# Get train and test IDs
	X_train,Y_train,X_test = data.load_data(TRAIN_PATH,TEST_PATH)

	img_size = 128

	train_dataset = augment.NucleusDataset(X_train, Y_train)
	train_dataloader = DataLoader(train_dataset, batch_size=12, shuffle=True)
	model = unet.UNet(3, 2)
	use_cuda = False
	gpu_ids = [0,1]
	optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

	mean = float(np.mean([np.mean(np.array(x)[:,:,:3]) for x in X_test]))
	std = float(np.std([np.std(np.array(x)[:,:,:3]) for x in X_test]))

	# Predict on train, val and test
	n_epochs = 1
	for _ in tqdm(range(n_epochs)):
		for data in train_dataloader:
			train_dataset.gen_params(img_size)
			optimizer.zero_grad()
			if use_cuda:
				X = Variable(data['X']).cuda()
				target = Variable(data['y']).cuda()
				model.cuda()
			else:
				X = Variable(data['X'])
				target = Variable(data['y'])
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