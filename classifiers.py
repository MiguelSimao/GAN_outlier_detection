#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 18 09:30:57 2018

@author: simao
"""

import numpy as np
import toolsfeatures
import keras.layers as kls
from keras import Model, regularizers
from keras.optimizers import Adam, SGD
import matplotlib.pyplot as plt
from sklearn import preprocessing


class ACGAN():
    def __init__(self, batch_size=256, num_classes=24, gesture_size=23,
                 latent_dim=10, label_noise=None, d_lr=.0002, g_lr=.001,
                 g_loss_weights=[1.3, 0.8]):
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.gesture_size = gesture_size
        self.label_noise = label_noise
        self.d_lr = d_lr
        self.g_lr = g_lr
        self.g_loss_w = g_loss_weights
        
        losses = ['binary_crossentropy', 'categorical_crossentropy']
        
        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.trainable = True
        self.discriminator.compile(loss={'d_output_source': 'binary_crossentropy',
                                         'd_output_class': 'categorical_crossentropy'},
                                   optimizer=Adam(d_lr, .5, decay=1e-7), #SGD(0.1, 0.1), #
                                   metrics=['accuracy'])
        #{'d_output_source': 'binary_crossentropy', 'd_output_class': 'categorical_accuracy'}
        
        # Build the generator
        self.generator = self.build_generator()
        
        # The generator takes noise and the target label as input
        # and generates the corresponding digit of that label
        noise = kls.Input(shape=(self.latent_dim,))
        label = kls.Input(shape=(self.num_classes + 1,))
        sample = self.generator([noise, label])
        
        # Generator
        self.generator.compile(loss='mean_squared_error',
                               optimizer=Adam())
        
        # For the combined model we will only train the generator
        self.discriminator.trainable = False
        
        # The discriminat.or takes generated sample as input and determines validity
        # and the label of that sample
        valid, target_label = self.discriminator(sample)
        
        # The combined model  (stacked generator and discriminator)
        # Trains the generator to fool the discriminator
        self.combined = Model([noise, label], [valid, target_label])
        self.combined.compile(loss=losses,
                              loss_weights=g_loss_weights,
                              optimizer=Adam(g_lr, 0.5, decay=1e-6) )
        
        
    
    def build_generator(self):
        # Generates samples from noisey class label conditions
        noise = kls.Input(shape=(self.latent_dim,))
        label = kls.Input(shape=(self.num_classes + 1,))
        
        model_input = kls.concatenate([noise,label])
        
        x = kls.Dense(256,
                      )(model_input) #activity_regularizer=regularizers.l1(0.0001)
        x = kls.GaussianNoise(.1)(x)
        x = kls.Activation('relu')(x)
        x = kls.BatchNormalization(momentum=0.1)(x)
        
        x = kls.Dense(256,
                      )(x)
        x = kls.GaussianNoise(.1)(x)
        x = kls.Activation('relu')(x)
        x = kls.BatchNormalization(momentum=0.1)(x)
#        x = kls.Dropout(0.3)(x)
#        x = kls.Dense(256)(x)
#        x = kls.GaussianNoise(.4)(x)
#        x = kls.Activation('relu')(x)
#        x = kls.BatchNormalization(momentum=0.5)(x)
        
        x = kls.Dense(self.gesture_size,
                      )(x)
        x = kls.Activation('linear')(x)
#        x = kls.ActivityRegularization(0.0,0.01)(x)
        return Model([noise,label], x)
    
    def build_discriminator(self):
        inputs = kls.Input(shape=(self.gesture_size,))
        
        x = kls.GaussianNoise(.4)(inputs)
        x = kls.Dense(300)(x)
        x = kls.Activation('relu')(x)
        
        x = kls.GaussianNoise(.4)(x)
        x = kls.Dense(300)(x)
        x = kls.Activation('relu')(x)
        x = kls.Dropout(0.3)(x)

        # Determine validity and label of the image
        validity = kls.Dense(1, activation="sigmoid", name='d_output_source')(x)
        label = kls.Dense(self.num_classes + 1, activation="softmax", name='d_output_class')(x)
        
        return Model(inputs, [validity, label])
    
    def fit(self, x=None, y=None, batch_size=None, epochs=1,
            validation_data=None, plot=None, runid=0):
        if not batch_size :
            batch_size=1
        if validation_data:
            X_val = validation_data[0]
            t_val = validation_data[1]
            
        # Split training data into batches
        X_train = []
        t_train = []
        for i in range(0, x.shape[0], batch_size):
            X_train.append(x[i:i+batch_size])
            t_train.append(y[i:i+batch_size])
        
        history = {'d_loss': [], 'g_loss': []}
        # Train for epochs
        for epoch in range(1, epochs + 1):
            try:
                gloss = np.empty((1,3)) #CHANGED (1,3) to (1,1)
                dloss = np.empty((1,5))
                
                # In each epoch, train all batches:
                for xp, tp in zip(X_train, t_train):
                    dloss0, gloss0 = self.train_on_batch(xp, tp)
                    gloss = np.append(gloss, gloss0.reshape((1,-1)), 0)
                    dloss = np.append(dloss, dloss0.reshape((1,-1)), 0)
            except KeyboardInterrupt:
                print('Training process interrupted prematurely.')
                break
            
            finally:
                # Validation at the end of the epoch
                if validation_data:
                    dloss_val = self.discriminator.evaluate(X_val,
                                                            [np.ones((t_val.shape[0],1)),
                                                                                   t_val],
                                                             verbose=0)
                else:
                    dloss_val = (0, 0, 0, 0, 0)
                dloss = np.mean(dloss, axis=0)
                gloss = np.mean(gloss, axis=0)
                
                history['d_loss'].append(dloss.reshape((1,-1)))
                history['g_loss'].append(gloss.reshape((1,-1)))
                
                # Epoch output:
                print('% 4d [G loss: %f]' % (epoch, gloss[0]), end='')
                print(' [D loss: %f, s_acc.: %.2f%%, c_acc: %.2f%%, val_acc: %.2f%%]' % \
                      (dloss[0], 100*dloss[3], 100*dloss[4], 100*dloss_val[4]))
                
                if plot and (epoch % plot) == 0:
                    print('Checkpoint!')
                    self.plot_fcn(True, epoch, x, y, runid=runid)
                
        return history
    
    def train_on_batch(self, x=None, t=None):
        # ---------------------
        #  Train Discriminator
        # ---------------------
        
        # Adversarial ground truths
        valid = np.ones((x.shape[0], 1))
        fake = np.zeros((x.shape[0], 1))
        
        # Select a batch of samples
        real_samples = x
        
        # Sample noise as generator input
        noise = np.random.normal(0, 1, (x.shape[0], self.latent_dim))
        
        # The labels of the digits that the generator tries to create an
        # sample representation of
        gened_labels = np.random.randint(0, self.num_classes, x.shape[0])
        gened_labels = toolsfeatures.onehotencoder(gened_labels, self.num_classes + 1)
        
        # Generate a half batch of new samples
        gened_samples = self.generator.predict([noise, gened_labels])

        # Sample labels
        real_labels = t
        fake_labels = gened_labels
        
        # Add noise to labels
        if self.label_noise:
            real_labels = toolsfeatures.label_noise(real_labels,
                                                    self.label_noise[0],
                                                    self.label_noise[1])
            fake_labels = toolsfeatures.label_noise(fake_labels,
                                                    self.label_noise[0],
                                                    self.label_noise[1])
        
        # Train the discriminator
#        d_loss_real = self.discriminator.train_on_batch(real_samples, [valid, real_labels])
#        d_loss_fake = self.discriminator.train_on_batch(gened_samples, [fake, fake_labels])
#        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
        d_loss = self.discriminator.train_on_batch(
                                    np.vstack((real_samples, gened_samples)),
                                    [np.vstack((valid,fake)),
                                     np.vstack((real_labels, fake_labels))])
        
        # ---------------------
        #  Train Generator
        # ---------------------

        # Add noise to some of the generated labels
#        j = int(gened_labels.shape[0] // 3)
#        gln = gened_labels[-j:]
#        probmax = np.random.uniform(0.8, 1.0)
#        gln[gln==1] = probmax
#        gln[gln!=probmax] = (1 - probmax) / (gln.shape[1] - 1)
#        fake_labels[-j:] = gln

        # Train the generator
        g_loss = self.combined.train_on_batch([noise, fake_labels],
                                              [valid, fake_labels])
#        d_loss_val = self.discriminator.evaluate(X_val,[np.ones((t_val.shape[0],1)), t_val], verbose=0)
        return np.array(d_loss), np.array(g_loss)
    
    def plot_fcn(self, save, epoch, X_train, t_train, runid):
        def myoind():
            myo0, myo1 = np.arange(8), np.arange(8,16)
            sensor_order = np.empty((myo0.size+myo1.size), dtype=np.int)
            sensor_order[0::2] = myo0
            sensor_order[1::2] = myo1
            return sensor_order
    
        num_examples = 32
        
        # Training samples targets:
        t_train_ind = np.argmax(t_train, axis=1)
        
        # Select real examples to show:
        vis_X = []
        for i in range(t_train_ind.max() + 1):
            I = np.argwhere(t_train_ind==i)[:num_examples].squeeze()
            vis_X.append(X_train[I])
        # "Other" class examples (all zeros)
        vis_X.append(np.zeros_like(vis_X[0]))
        
        # Generate samples:
        vis_X_gen = [] #gen'ed storage
        # for each class
        for i in range(t_train_ind.max() + 2):
            # for each sample (16)
            tmp_y = np.zeros_like(vis_X[0])
            for j in range(num_examples):
                noise = np.random.normal(0, 1, size=(1, self.latent_dim))
        
                t = np.zeros((1, 8))
                t[0,i] = 1
                t = toolsfeatures.onehotnoise(np.array([i]), 8, 0.4)
                tmp_y[j] = self.generator.predict([noise,t])
            vis_X_gen.append(tmp_y)
        
        # Scale data:
        scaler = preprocessing.MinMaxScaler().fit(np.concatenate(vis_X_gen+vis_X))
        for i in range(len(vis_X)):
            vis_X[i] = scaler.transform(vis_X[i])
            vis_X_gen[i] = scaler.transform(vis_X_gen[i])
        vis_X[-1] = np.zeros_like(vis_X[0])
        
        fig, axs = plt.subplots(nrows=2, ncols=8,
                                sharex=True, sharey=True, figsize=(14, 6))
        
        for row_ind in range(axs.shape[0]):
            for col_ind in range(axs.shape[1]):
                plt.axes(axs[row_ind,col_ind])
                if col_ind == 0:
                    if row_ind == 0:
                        plt.ylabel('Real')
                    else:
                        plt.ylabel('Generated')
                if row_ind == 1:
                    plt.xlabel('G%i' % col_ind)
                    axs[row_ind,col_ind].imshow(vis_X_gen[col_ind][:,myoind()])
                else:
                    axs[row_ind,col_ind].imshow(vis_X[col_ind][:,myoind()])
        
        fig.suptitle('[Net %i][Epoch:%i][d_lr=%g][g_lr=%g][w_loss=(%g,%g)]' % 
                          (runid, epoch, self.d_lr, self.g_lr,
                          self.g_loss_w[0], self.g_loss_w[1]))
        if save:
            fig.savefig('checkpoints/gan%08i.%04i.png' % (runid, epoch))
        else:
            fig.show()
        plt.close()
        
    
    def evaluate_discr(self, x=None, t=None):
        pass
