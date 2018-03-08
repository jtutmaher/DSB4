import numpy as np
import cv2
import os

def split_data(arr,dim=[8,8]):
	arr = np.array(np.split(arr,dim[0]))
	res = []
	for data in arr:
		res.extend(np.split(data,dim[1],axis=1))
	return np.array(res)

def read_images(im_dir):
	files = sorted(os.listdir(im_dir))
	first = True
	for file in files:
		if file == ".DS_Store":
			continue
		name = im_dir + file + "/images/"+file+".png"
		im = np.average(cv2.imread(name),axis=2)
		im = split_data(im[0:256,0:256]).reshape(64,32,32,1).astype('float32')
		if first == True:
			out = im
			first = False
		else:
			out = np.concatenate((out,im),axis=0)
	img_mean = np.mean(out,axis=0)
	img_std  = np.std(out)
	out -= img_mean
	out /= img_std
	return out

def read_labels(im_dir):
	files = sorted(os.listdir(im_dir))
	first = True
	for file in files:
		if file==".DS_Store":
			continue
		name = im_dir+file+"/masks/"
		mask_files = os.listdir(name)
		mask = np.array([np.average(cv2.imread(name+x),axis=2)[0:256,0:256] for x in mask_files])
		mask = split_data(np.sum(mask,axis=0)).reshape((64,32,32,1)).astype('float32')
		if first == True:
			out = mask
			first = False
		else:
			out = np.concatenate((out,mask),axis=0)
	return out

if __name__=="__main__":
	DIR = "./data/stage1_train/"
	print("Reading Images")
	xs = read_images(DIR)
	np.savez_compressed("./images.npz",xs)
	print(xs.shape)
	print("Reading Labels")
	ys = read_labels(DIR)
	np.savez_compressed("./labels.npz",ys)
	print(ys.shape)