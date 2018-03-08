import numpy as np
import os
from skimage.io import imsave
from model import unet9
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator

def load_train_data(train_file):
    arr = np.load(train_file)
    images,labels = arr['images'],arr['masks']

    mean = np.mean(images)  # mean for data centering
    std = np.std(images)  # std for data normalization

    images -= mean
    images /= std

    print np.amax(labels)
    labels /= 255.  # scale masks to [0, 1]
    print("Train Shape")
    print(images.shape,labels.shape)
    return images,labels

def augment_data(images,masks):
    # we create two instances with the same arguments
    data_gen_args = dict(featurewise_center=True,
                     featurewise_std_normalization=True,
                     rotation_range=90.,
                     width_shift_range=0.1,
                     height_shift_range=0.1,
                     zoom_range=0.2)
    image_datagen = ImageDataGenerator(**data_gen_args)
    mask_datagen = ImageDataGenerator(**data_gen_args)

    # Provide the same seed and keyword arguments to the fit and flow methods
    seed = 1
    image_datagen.fit(images, augment=True, seed=seed)
    mask_datagen.fit(masks, augment=True, seed=seed)

    image_generator = image_datagen.flow(images,batch_size=32)
    mask_generator = mask_datagen.flow(masks,batch_size=32)

    # combine generators into one which yields image and masks
    return image_generator,mask_generator
    #while True:
    #    yield zip(image_generator.next(), mask_generator.next())

def train_and_predict():
    print('-'*30)
    print('Loading and preprocessing train data...')
    images,labels = load_train_data("./data.npz")

    #print('-'*30)
    #print('Augmenting Data...')
    #image_generator,mask_generator = augment_data(images,labels)

    print('-'*30)
    print('Creating and compiling model...')
    print('-'*30)
    model = unet9(base=32)
    model_checkpoint = ModelCheckpoint('weights.h5', monitor='val_loss', save_best_only=True)

    print('-'*30)
    print('Fitting model...')
    print('-'*30)
    #model.fit_generator(zip(x,y),steps_per_epoch=int(images.shape[0]/float(32)),epochs=5,
    #        callbacks=[model_checkpoint])
    model.fit(images, labels, batch_size=32, epochs=8, verbose=1, shuffle=True,
              validation_split=0.2,
              callbacks=[model_checkpoint])


if __name__ == '__main__':
    train_and_predict()