import numpy as np
import cv2
import os
from skimage.color import rgb2gray
from skimage.filters import threshold_otsu
from scipy.misc import imfilter
from model import unet9
import pandas as pd
from skimage.morphology import label


def split_data(arr,dim=[8,8]):
    arr = np.array(np.split(arr,dim[0]))
    res = []
    for data in arr:
        res.extend(np.split(data,dim[1],axis=1))
    return np.array(res)

def reconstruct_data(arr,dim_pad=(256,256),dim_orig=(256,256),dim=(32,32)):
    x_stride = dim_pad[1]/dim[0]
    y_stride = dim_pad[0]/dim[1]
    out = np.zeros(dim_pad)
    count=0
    for y in range(y_stride):
        for x in range(x_stride):
            x_start = dim[0]*x
            x_end = dim[0]+x_start
            y_start = dim[1]*y
            y_end = dim[1]+y_start
            out[y_start:y_end,x_start:x_end] = arr[count,:,:,0]
            count+=1
    return out[:dim_orig[0],:dim_orig[1]]

def check_invert(im):
    hist = np.histogram(im)[0]
    score = np.sum(hist[-3:])
    total = np.sum(hist)
    if score/float(total)>0.15:
        return (1-im)
    else:
        return im

def load_test_data(test_path,dim=(32,32)):
    records = os.listdir(test_path)
    records.remove(".DS_Store")

    first = True
    testset = {}
    testset_im = {}
    prior_shape = {}
    post_shape = {}
    for rec in records:
        subdir = test_path+rec+"/images/"
        subfile = os.listdir(subdir)
        name = subfile[0][:-4]
        if len(subfile)>1:
            print("Error, more than one image found")
        arr = cv2.imread(subdir+subfile[0])
        arr = check_invert(rgb2gray(arr))
        arr2 = imfilter(arr,'edge_enhance')
        testset_im[name]=arr2
        prior_shape[name]=arr2.shape
    
        #Pad Array
        width_pad = dim[0]-arr2.shape[0]%dim[0]
        length_pad = dim[1]-arr2.shape[1]%dim[1]
        temp = np.zeros((arr2.shape[0]+width_pad,arr2.shape[1]+length_pad))
        temp[:arr2.shape[0],:arr2.shape[1]]=arr2
        post_shape[name]=temp.shape
    
        #Break into Grids
        N = ((arr2.shape[0]+width_pad)/dim[0])*((arr2.shape[1]+length_pad)/dim[1])
        im = split_data(temp,dim=[(arr2.shape[0]+width_pad)/dim[0],(arr2.shape[1]+length_pad)/dim[1]]).reshape(N,dim[0],dim[1],1).astype('float32')
        #im[im<65]=0
        testset[name]=im
    
        #Compute Mean and STD
        if first == True:
            out = im
            first = False
        else:
            out = np.concatenate((out,im),axis=0)
    
    print np.amax(out)
    img_mean = np.mean(out,axis=0)
    img_std  = np.std(out)

    #Normalize Test Set Data
    testset_norm = {}
    for key in testset:
        testset_norm[key] = (testset[key]-img_mean)/img_std
        
    return testset, testset_norm, prior_shape,post_shape

def predict(norms,prior,post,base=64):
	model = unet9(base=base)
	model.load_weights("./weights.h5")
	output = {}
	print "here"
	for key in norms:
		batch = norms[key]
		prior_new = prior[key]
		post_new = post[key]
		print("Inputs: "+key)
		print("Batch: "+str(batch.shape))
		print("Priors: "+str(prior_new))
		print("Posts: "+str(post_new))
		masks = model.predict(batch,verbose=1)
		masks_recon = reconstruct_data(masks,dim_pad=post_new,dim_orig=prior_new,dim=(base,base))
		output[key] = masks_recon
		print masks_recon.shape

	return output

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

def gen_csv(out,filename):
	new_test_ids = []
	rles = []

	for key in output:
		rle = list(prob_to_rles(output[key]))
		rles.extend(rle)
		new_test_ids.extend([key] * len(rle))

	sub = pd.DataFrame()
	sub['ImageId'] = new_test_ids
	sub['EncodedPixels'] = pd.Series(rles).apply(lambda x: ' '.join(str(y) for y in x))
	sub.to_csv(filename, index=False)

if __name__=="__main__":
	scale = 64
	tests, norms, prior,post = load_test_data("./data/stage1_test/",dim=[scale,scale])
	out = predict(norms,prior,post,base=scale)
	gen_csv(out,'sub-dsbowl2018-26.csv')



