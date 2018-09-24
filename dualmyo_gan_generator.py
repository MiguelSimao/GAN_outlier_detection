#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 13 17:19:33 2018

@author: simao
"""

import numpy as np
import time
import pickle
from dataset.dualmyo.utils import Loader, SynteticSequences
from dataset.dualmyo import dualmyofeatures
from tools import toolstimeseries as tts
from tools import toolsfeatures
from tools.postprocessing import PostProcessor

from sklearn import preprocessing
import keras
from keras import Model
from keras.optimizers import Adam
import keras.layers as kls

# ENSURE REPRODUCIBILITY ######################################################
import os
import random
os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(1337)
random.seed(12345)
###############################################################################


#%% LOAD DATA

DataLoader = Loader()

sample_data, sample_target = DataLoader.load()
sample_data = np.concatenate( [sample.reshape((1,) + sample.shape) for sample in sample_data], axis=0 )
sample_target = np.array(sample_target)

# Data split
ind_train, ind_val, ind_test = DataLoader.split(sample_target)

#%% FEATURE EXTRACTION

# Feature extraction
X_train = np.vstack([dualmyofeatures.extract_std(sample) for sample in sample_data[ind_train]])
X_val = np.vstack([dualmyofeatures.extract_std(sample) for sample in sample_data[ind_val]])
X_test = np.vstack([dualmyofeatures.extract_std(sample) for sample in sample_data[ind_test]])

# Feature scaling
feature_scaler = preprocessing.StandardScaler().fit(X_train)
X_train = feature_scaler.transform(X_train)
X_val = feature_scaler.transform(X_val)
X_test = feature_scaler.transform(X_test)

# Target processing
t_train = sample_target[ind_train]
t_val = sample_target[ind_val]
t_test = sample_target[ind_test]
t_train = toolsfeatures.onehotencoder(t_train, 9)
t_val = toolsfeatures.onehotencoder(t_val, 9)
t_test = toolsfeatures.onehotencoder(t_test, 9)

#%% GAN MODEL DEFINITION

class ACGAN():
    def __init__(self, batch_size=256, num_classes=24, gesture_size=23, latent_dim=10):
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.gesture_size = gesture_size
        
        adam = Adam(0.0002, .5)
        
        losses = ['binary_crossentropy', 'categorical_crossentropy']
        
        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.trainable = True
        self.discriminator.compile(loss=losses,
                                   optimizer=adam,
                                   metrics={'d_output_source': 'binary_crossentropy',
                                            'd_output_class': 'categorical_accuracy'}) #['categorical_accuracy']
        
        # Build the generator
        self.generator = self.build_generator()
        
        # The generator takes noise and the target label as input
        # and generates the corresponding digit of that label
        noise = kls.Input(shape=(self.latent_dim,))
        label = kls.Input(shape=(self.num_classes + 1,))
        sample = self.generator([noise, label])
        
        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The discriminator takes generated sample as input and determines validity
        # and the label of that sample
        valid, target_label = self.discriminator(sample)
        
        # The combined model  (stacked generator and discriminator)
        # Trains the generator to fool the discriminator
        self.combined = Model([noise, label], [valid, target_label])
        self.combined.compile(loss=losses,
                              optimizer=adam)
        
        
    
    def build_generator(self):
        # Generates samples from noisey class label conditions
        noise = kls.Input(shape=(self.latent_dim,))
        label = kls.Input(shape=(self.num_classes + 1,))
        
        model_input = kls.concatenate([noise,label])
        
        x = kls.Dense(200)(model_input)
        x = kls.GaussianNoise(.5)(x)
        x = kls.Activation('relu')(x)
        x = kls.BatchNormalization(momentum=0.8)(x)
        
        x = kls.Dense(200)(model_input)
        x = kls.GaussianNoise(.5)(x)
        x = kls.Activation('relu')(x)
        x = kls.BatchNormalization(momentum=0.8)(x)
        
        x = kls.Dense(200)(model_input)
        x = kls.Activation('relu')(x)
        x = kls.BatchNormalization(momentum=0.8)(x)
        
        x = kls.Dense(self.gesture_size)(x)
        x = kls.Activation('tanh')(x)
        
        return Model([noise,label], x)
    
    def build_discriminator(self):
        inputs = kls.Input(shape=(self.gesture_size,))
        
        x = kls.GaussianNoise(.4)(inputs)
        x = kls.Dense(200)(x)
        x = kls.Activation('relu')(x)
        
        x = kls.GaussianNoise(.4)(x)
        x = kls.Dense(200)(x)
        x = kls.Activation('relu')(x)
        
#        x = kls.BatchNormalization(momentum=0.8)(x)
        
        # Determine validity and label of the image
        validity = kls.Dense(1, activation="sigmoid", name='d_output_source')(x)
        label = kls.Dense(self.num_classes + 1, activation="softmax", name='d_output_class')(x)
        
        return Model(inputs, [validity, label])


#%% GAN MODEL INSTANCING

gan = ACGAN(batch_size=32, num_classes=8, gesture_size=16, latent_dim=6)

# Adversarial ground truths
valid = np.ones((gan.batch_size, 1))
fake = np.zeros((gan.batch_size, 1))


#%% ----- TRAIN ----- #

best_val_accuracy = 0

for epoch in range(12000):
        
    try:
        # ---------------------
        #  Train Discriminator
        # ---------------------

        # Select a random batch of samples
        idx = np.random.randint(0, X_train.shape[0], gan.batch_size)
        real_samples = X_train[idx]
        
        # Sample noise as generator input
        noise = np.random.normal(0, 1, (idx.shape[0], gan.latent_dim))

        # The labels of the digits that the generator tries to create an
        # sample representation of
        gened_labels = np.random.randint(0, gan.num_classes + 1, idx.shape[0])
        gened_labels = toolsfeatures.onehotencoder(gened_labels, gan.num_classes + 1)
        
        # Generate a half batch of new samples
        gened_samples = gan.generator.predict([noise, gened_labels])

        # Sample labels
        real_labels = t_train[idx]
        fake_labels = gened_labels

        # Train the discriminator
        d_loss_real = gan.discriminator.train_on_batch(real_samples, [valid, real_labels])
        d_loss_fake = gan.discriminator.train_on_batch(gened_samples, [fake, fake_labels])
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
        
        # ---------------------
        #  Train Generator
        # ---------------------

        # Train the generator
        g_loss = gan.combined.train_on_batch([noise, fake_labels], [valid, fake_labels])
        d_loss_val = gan.discriminator.evaluate(X_val, [np.ones((t_val.shape[0],1)), t_val], verbose=0)
        
        #
        if d_loss[4] > best_val_accuracy:
            best_val_accuracy = d_loss[4]
            best_weights_d = gan.discriminator.get_weights()
            best_weights_g = gan.generator.get_weights()

        # Epoch output:
        print ("%d [G loss: %f] [D loss: %f, s_acc.: %.2f%%, c_acc: %.2f%%, val_acc: %.2f%%] (Best: %.2f%%)" % \
              (epoch + 1, g_loss[0], d_loss[0], 100*d_loss[3], 100*d_loss[4], 100*d_loss_val[4], 100*best_val_accuracy))
        
    except KeyboardInterrupt:
        print('Training prematurely interrupted.')
        break


#%% DATASET GENERATION

n_outliers = 32
n_per_class = 32
n_per_batch = (n_outliers + n_per_class) * 8

# Noise
noise = np.random.normal(0, 1, ((n_per_batch, gan.latent_dim)))
# Generate outliers class
gened_labels = np.tile(np.arange(8), (n_per_class,))
# Add generated 
gened_labels = np.concatenate((gened_labels,np.tile(8, (n_outliers * 8))))
gened_labels = toolsfeatures.onehotencoder(gened_labels, 9)
gened_samples = gan.generator.predict([noise, gened_labels])


X_train_gen = gened_samples
t_train_gen = gened_labels
o_train_gen = np.zeros_like(gened_labels)
o_train = np.ones_like((t_train))

generated_dataset = {'X_train': X_train,
                     't_train': t_train,
                     'X_val': X_val,
                     't_val': t_val,
                     'X_test': X_test,
                     't_test': t_test,
                     'X_train_gen': X_train_gen,
                     't_train_gen': t_train_gen,
                     'o_train_gen': o_train_gen
                     }
#%% SAVE DATA

with open('dualmyo_dataset_generated.pkl','wb') as file:
    pickle.dump(generated_dataset, file)
    
gan.discriminator.save('./nets/trainedGan_dicriminator.h5')
gan.generator.save('./nets/trainedGan_generator.h5')
    