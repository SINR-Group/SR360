#!/usr/bin/env python
# coding: utf-8

# In[1]:


from __future__ import print_function, division

from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, multiply, GlobalAveragePooling2D, MaxPooling2D
from keras.layers import BatchNormalization, Activation, Embedding, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam

import matplotlib.pyplot as plt

import numpy as np
from scipy.misc import imread
from scipy.misc import imresize
from scipy.misc import imsave
import glob
import random

from keras.preprocessing.image import ImageDataGenerator
from keras import applications
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical
#from keras.applications.resnet50 import preprocess_input

from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
#from keras.applications.vgg19 import preprocess_input as preprocess_vgg
from keras.applications.inception_v3 import preprocess_input as inception_preprocess

#from keras.applications.inception_resnet_v2 import preprocess_input

import numpy as np

import keras
import copy
import cv2
import os


# In[2]:


path = '../data/'
savepath = '../data/'
images = glob.glob(path+'new_data/*.npy')


# In[4]:


tile_sizes = np.loadtxt(path+'tile_sizes.txt', dtype='int')
images_sampled = {}
for tile in tile_sizes:
    if tile[2] > 200000:
        for i in range(30):
            images_sampled.setdefault(tile[0]*30+i, []).append(tile[1])


# In[11]:


def myGenerator(batch_size):
    while True:
        #index_list = random.sample(range(1, totalImages), batch_size)
        index_list = random.sample(images_sampled.keys(), batch_size)
        alldata_x = []
        alldata_y = []
        for i in index_list:
            print (i)
            frame = path+'sources/new_data/frame'+str(i)+'.npy'
            frame = np.load(frame)
            #tile_index = np.random.randint(0, 199)
            #print(i, tile_index, frame.shape, images_sampled[i])
            #alldata_x.append(tile_index*totalImages+i)
            alldata_y.append(frame[tile_index])
        alldata_x = np.array(alldata_x)
        alldata_y = np.array(alldata_y)
        #alldata_y = alldata_y/255.0
        alldata_y = (alldata_y.astype(np.float32) - 127.5) / 127.5
        yield alldata_y, alldata_y
x = myGenerator(10)
xtrain, ytrain = next(x)
print('xtrain shape:',xtrain.shape)
print('ytrain shape:',ytrain.shape)


# In[5]:


class CGAN():
    def __init__(self):
        # Input shape
        self.img_rows = 384
        self.img_cols = 384
        self.channels = 3
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.optimizer = Adam(0.0002, 0.5)
        self.latent_dim = 300
        
        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss=['mse'],
            optimizer=self.optimizer,
            metrics=['accuracy'])
        
        #print(self.discriminator.summary())
        
        # Build the generator
        self.generator = self.build_generator()
        
        print(self.generator.summary())
        
        noise = Input(shape=(self.latent_dim, ))
        img = self.generator(noise)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The discriminator takes generated image as input and determines validity
        # and the label of that image
        valid = self.discriminator(img)
        
        # The combined model  (stacked generator and discriminator)
        # Trains generator to fool discriminator
        self.combined = Model(noise, valid)
        self.combined.compile(loss=['mse'],
            optimizer=self.optimizer)

    def build_generator(self):
        model = Sequential()

        model.add(Dense(64 * 48 * 48, input_dim=300))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Reshape((48, 48, 64)))

        model.add(Conv2D(512, kernel_size=3, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))

        model.add(Conv2D(512, kernel_size=3, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))

        model.add(UpSampling2D())
        model.add(Conv2D(256, kernel_size=3, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        
        model.add(Conv2D(256, kernel_size=3, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        
        model.add(UpSampling2D())
        model.add(Conv2D(256, kernel_size=3, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))

        model.add(Conv2D(128, kernel_size=3, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))

        model.add(UpSampling2D())
        model.add(Conv2D(64, kernel_size=3, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))

        model.add(Conv2D(64, kernel_size=3, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))

        model.add(Conv2D(32, kernel_size=3, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))
    
        model.add(Conv2D(32, kernel_size=3, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))
        
        model.add(Conv2D(3, kernel_size=3, padding="same"))
        model.add(Activation("tanh"))

        noise = Input(shape=(self.latent_dim,))
        img   = model(noise)
        return model
    
    def build_autoencoder(self):
        self.generator.compile(loss=['mse'], optimizer=self.optimizer)
    
    def train_generator_autoencoder(self, epochs, batch_size=128, sample_interval=10):
        for epoch in range(epochs):
            X_train, y_train = next(myGenerator(batch_size))
            g_loss = self.generator.train_on_batch(X_train, X_train)
            print ("Epoch ", epoch, " G loss ", g_loss)
            if epoch % sample_interval == 0:
                self.sample_images(epoch)
                self.generator.save_weights(savepath+'weights/generator_weights_'+str(epoch)+'.h5')
            
    def build_discriminator(self):
        img   = Input(shape=(384, 384, 3))
        model = Sequential()
        model.add(Conv2D(64, (3, 3), input_shape=(384, 384, 3)))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(64,  (3, 3),  strides=(2, 2)))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.5))
        model.add(Conv2D(64, (3, 3)))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.5))
        model.add(Conv2D(64, (6, 6),  strides=(2, 2)))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.5))
        model.add(Conv2D(64, (3, 3),  strides=(2, 2)))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Conv2D(64, (3, 3),  strides=(2, 2)))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.5))
        model.add(Flatten())
        model.add(Dense(100))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(32))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(1))
        
        output    = model(img)
        model3    = Model(img, output)
        return model3

    def train(self, epochs, batch_size=128, sample_interval=50):
        random.seed(10)
        
        # Load the dataset
        for epoch in range(epochs):
            X_train, y_train = next(myGenerator(batch_size))
            
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            
            # Adversarial ground truths
            valid = np.ones((batch_size, 1))
            fake  = np.zeros((batch_size, 1))
            
            # ---------------------
            #  Train Discriminator
            # ---------------------
            
            # Generate a half batch of new images
            gen_imgs = self.generator.predict(noise)

            # Train the discriminator
            d_loss_real = self.discriminator.train_on_batch(X_train, valid)
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  Train Generator
            # ---------------------

            # Train the generator
            g_loss = self.combined.train_on_batch(noise, valid)

            # Plot the progress
            print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))

            # If at save interval => save generated image samples
            if epoch % sample_interval == 0:
                self.sample_images(epoch)
                self.generator.save_weights(savepath+'weights/generator_weights_'+str(epoch)+'.h5')
                self.discriminator.save_weights(savepath+'weights/discriminator_weights_'+str(epoch)+'.h5')

    def sample_images(self, epoch):
        r, c             = 1, 10
        noise            = np.random.normal(0, 1, (5, self.latent_dim))
        gen_imgs         = self.generator.predict(noise)
        
        # Rescale images 0 - 1
        temp     = (0.5 * gen_imgs + 0.5)*255
        gen_imgs = temp.astype(int)
        
        combined = np.array([gen_imgs[0], gen_imgs[1], gen_imgs[2], gen_imgs[3], gen_imgs[4]])
        combined = np.hstack(combined.reshape(5, 384,384, 3))
        imsave(savepath+"images/"+str(epoch)+".png", combined)
        
#         combined = np.array([Y_train[0], Y_train[1], Y_train[2], Y_train[3], Y_train[4]])
#         combined = np.hstack(combined.reshape(5, 384,384, 3))
#         imsave(savepath+"images/"+str(epoch)+"_real.png", combined)


# In[ ]:


#cgan = CGAN()
#cgan.build_discriminator()
#cgan.build_generator()
cgan.train(100000, 6, 100)
#cgan.build_autoencoder()
#cgan.train_generator_autoencoder(100000, 8, 100)


# In[ ]:




