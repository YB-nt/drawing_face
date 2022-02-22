from tkinter import Scale
from turtle import shape
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Concatenate
from keras.layers import BatchNormalization, Activation, ZeroPadding2D, Add
from keras.layers.advanced_activations import LeakyReLU, ELU
from keras.layers.convolutional import UpSampling2D, Conv2D, Conv2DTranspose
from keras.layers.merge import add
from models.layers.layers import ReflectionPadding2D
from keras.models import Sequential, Model
from keras.initializers import RandomNormal
from keras.optimizers import Adam
from keras import backend as K

from keras.utils import plot_model

import datetime
import matplotlib.pyplot as plt
import sys

import numpy as np
import os
import pickle as pkl
import random

from collections import deque




class CycleGAN():
    def __init__(self,
                input_dim,
                learning_rate,
                lambda_validation,
                lambda_reconstr,
                lambda_id,
                generator_type,
                gen_n_filters,
                disc_n_filters,
                buffer_max_length = 50):

        self.input_dim = input_dim
        self.learning_rate =learning_rate 
        self.lambda_validation = lambda_validation
        self.lambda_reconstr = lambda_reconstr
        self.lambda_id = lambda_id
        self.generator_type = generator_type
        self.gen_n_filters = gen_n_filters
        self.disc_n_filters = disc_n_filters
        self.buffer_max_length = buffer_max_length

        self.img_row = input_dim[0]
        self.img_cols = input_dim[1]
        self.channels = input_dim[2]
        self.img_shape = input_dim

        self.d_losses = []
        self.g_losses =[]
        self.epoch =0
        self.buffer_A = deque(maxlen=self.buffer_max_length)
        self.buffer_B = deque(maxlen=self.buffer_max_length) 


        patch = int(self.img_row/2**4)

        self.disc_patch = (patch,patch,1)
        self.weight_init = RandomNormal(mean=0., stddev=0.02)
        self.compile_models()
    # cyclegan 모델  
    def bulid_model(self):
        # optimizer 정의 
        optimizer = Adam(0.0002,self.learning_rate)
        # 판별자  정의
        self.d_A = self.build_discriminator()
        self.d_B = self.build_discriminator()
        # 정의된 판별자 컴파일
        self.d_A.compile(
            loss='mse',
            optimizer=optimizer
            metrics=['accuracy']
        )
        self.d_B.complie(
            loss='mse',
            optimizer=optimizer
            metrics=['accuracy']
        )

        if(self.generator_type=='u'):
            self.g_AB = self.build_generator_u()
            self.g_BA = self.build_generator_u()
        elif(self.generator_type=='r'):
            self.g_AB = self.build_generator_r()
            self.g_BA = self.build_generator_r()
        else:
            raise 

        img_A = input(shape=self.img_shape)
        img_B = input(shape=self.img_shape)

        img_A_id = self.g_BA(img_A)
        img_B_id = self.g_AB(img_B)

        reconstr_A = self.g_BA(fake_B)
        reconstr_B = self.g_BA(fake_A)

        fake_B =self.g_AB(img_B)
        fake_A =self.g_BA(img_A)

        self.d_A.trainable = False
        self.d_B.trainable = False

        valid_A = self.d_A(fake_A)
        valid_B = self.d_B(fake_B)

        self.combined = Model(inputs=[img_A, img_B],
                              outputs=[ valid_A, valid_B,
                                        reconstr_A, reconstr_B,
                                        img_A_id, img_B_id ])
        self.combined.compile(loss=['mse', 'mse',
                                    'mae', 'mae',
                                    'mae', 'mae'],
                            loss_weights=[  1, 1,
                                            self.lambda_cycle, self.lambda_cycle,
                                            self.lambda_id, self.lambda_id ],
                            optimizer=optimizer)
                            
    def build_generator_u(self):
        # 생성 먼저야
        def downsample(layer_input,filters,f_size=4):
            d = Conv2D(filters,kernel_size=f_size,strides=2,padding='same')(layer_input)
            d = InstanceNormalization(axis=-1,center=False,scale=False)(d)
            d = Activation('relu')(d)

            return d

        def upsample(layer_input,skip_input,filters,f_size=4,dropout_rate=0):
            u = UpSampling2D(size=2)(layer_input)
            u = Conv2D(filter,kernel_size=f_size,strides=1,padding='same')(u)
            u = InstanceNormalization(axis=-1,center=False,scale=False)(u)
            u = Activation('relu')
            if(dropout_rate):
                u = Dropout(dropout_rate)(u)
            u = Concatenate()([u,skip_input])

            return u

     def build_generator_r(self):

        def conv7s1(layer_input, filters, final):
            y = ReflectionPadding2D(padding =(3,3))(layer_input)
            y = Conv2D(filters, kernel_size=(7,7), strides=1, padding='valid', kernel_initializer = self.weight_init)(y)
            if final:
                y = Activation('tanh')(y)
            else:
                y = InstanceNormalization(axis = -1, center = False, scale = False)(y)
                y = Activation('relu')(y)
            return y

        def downsample(layer_input,filters):
            y = Conv2D(filters, kernel_size=(3,3), strides=2, padding='same', kernel_initializer = self.weight_init)(layer_input)
            y = InstanceNormalization(axis = -1, center = False, scale = False)(y)
            y = Activation('relu')(y)
            return y

        def residual(layer_input, filters):
            shortcut = layer_input
            y = ReflectionPadding2D(padding =(1,1))(layer_input)
            y = Conv2D(filters, kernel_size=(3, 3), strides=1, padding='valid', kernel_initializer = self.weight_init)(y)
            y = InstanceNormalization(axis = -1, center = False, scale = False)(y)
            y = Activation('relu')(y)
            
            y = ReflectionPadding2D(padding =(1,1))(y)
            y = Conv2D(filters, kernel_size=(3, 3), strides=1, padding='valid', kernel_initializer = self.weight_init)(y)
            y = InstanceNormalization(axis = -1, center = False, scale = False)(y)

            return add([shortcut, y])

        def upsample(layer_input,filters):
            y = Conv2DTranspose(filters, kernel_size=(3, 3), strides=2, padding='same', kernel_initializer = self.weight_init)(layer_input)
            y = InstanceNormalization(axis = -1, center = False, scale = False)(y)
            y = Activation('relu')(y)
    
            return y


        # Image input
        img = Input(shape=self.img_shape)

        y = img

        y = conv7s1(y, self.gen_n_filters, False)
        y = downsample(y, self.gen_n_filters * 2)
        y = downsample(y, self.gen_n_filters * 4)
        y = residual(y, self.gen_n_filters * 4)
        y = residual(y, self.gen_n_filters * 4)
        y = residual(y, self.gen_n_filters * 4)
        y = residual(y, self.gen_n_filters * 4)
        y = residual(y, self.gen_n_filters * 4)
        y = residual(y, self.gen_n_filters * 4)
        y = residual(y, self.gen_n_filters * 4)
        y = residual(y, self.gen_n_filters * 4)
        y = residual(y, self.gen_n_filters * 4)
        y = upsample(y, self.gen_n_filters * 2)
        y = upsample(y, self.gen_n_filters)
        y = conv7s1(y, 3, True)
        output = y

   
        return Model(img, output)


    def build_discriminator(self):
        def d_layer(layer_input,filters,f_size=4,normalization=True):
            d = Conv2D(filters,kernel_size=f_size,strides=2,padding='same')(layer_input)
            if normalization:
                d = InstanceNormalization(axis=-1,center=False,scale=False)(y)
            d = LeakyReLU(0.2)(d)

            return d

        img = Input(shape=self.img_shape)

        d = d_layer(img,self.disc_n_filters,stride=2,normalization=False)
        d = d_layer(y,self.disc_n_filters*2, stride = 2) 
        d = d_layer(y,self.disc_n_filters*4, stride = 2)  
        d = d_layer(y,self.disc_n_filters*8, stride = 1)  

        output = Conv2D(1,kernel_size=4,strides=1,padding='same',kernel_initializer = self.weight_init)(d)

        return Model(img,output)

    def train_gen(self,img_A,img_B,valid):
        return self.combined.train_on_batch([img_A,img_B],
                                            [valid,valid,
                                            img_A,img_B,
                                            img_A,img_B])

    def train_dis(self,img_A,img_B,valid,fake_img):
        fake_B = self.g_AB.predict(img_A)
        fake_A = self.g_BA.predict(img_B)

        self.buffer_B.append(fake_B)
        self.buffer_A.append(fake_A)

        fake_A_rnd = random.sample(self.buffer_A,min(len(self.buffer_A),len(img_A)))
        fake_B_rnd = random.sample(self.buffer_B,min(len(self.buffer_B),len(img_B)))

               # Train the discriminators (original images = real / translated = Fake)
        dA_loss_real = self.d_A.train_on_batch(img_A, valid)
        dA_loss_fake = self.d_A.train_on_batch(fake_A_rnd, fake_img)
        dA_loss = 0.5 * np.add(dA_loss_real, dA_loss_fake)

        dB_loss_real = self.d_B.train_on_batch(img_B, valid)
        dB_loss_fake = self.d_B.train_on_batch(fake_B_rnd, fake_img)
        dB_loss = 0.5 * np.add(dB_loss_real, dB_loss_fake)

        # Total disciminator loss
        d_loss_total = 0.5 * np.add(dA_loss, dB_loss)

        return (
            d_loss_total[0]
            , dA_loss[0], dA_loss_real[0], dA_loss_fake[0]
            , dB_loss[0], dB_loss_real[0], dB_loss_fake[0]
            , d_loss_total[1]
            , dA_loss[1], dA_loss_real[1], dA_loss_fake[1]
            , dB_loss[1], dB_loss_real[1], dB_loss_fake[1]
        )
 

    def train(self, data_loader, run_folder, epochs, test_A_file, test_B_file, batch_size=1, sample_interval=50):
        start_time = datetime.datetime.now()

        # Adversarial loss ground truths
        valid = np.ones((batch_size,) + self.disc_patch)
        fake = np.zeros((batch_size,) + self.disc_patch)

        for epoch in range(self.epoch, epochs):
            for batch_i, (imgs_A, imgs_B) in enumerate(data_loader.load_batch()):

                d_loss = self.train_dis(imgs_A, imgs_B, valid, fake)
                g_loss = self.train_gen(imgs_A, imgs_B, valid)

                elapsed_time = datetime.datetime.now() - start_time

                # Plot the progress
                if batch_i % 100 == 0:
                    print ("[Epoch %d/%d] [Batch %d/%d] [D loss: %f, acc: %3d%%] [G loss: %05f, adv: %05f, recon: %05f, id: %05f] time: %s " \
                        % ( self.epoch, epochs,
                            batch_i, data_loader.n_batches,
                            d_loss[0], 100*d_loss[7],
                            g_loss[0],
                            np.sum(g_loss[1:3]),
                            np.sum(g_loss[3:5]),
                            np.sum(g_loss[5:7]),
                            elapsed_time))

                self.d_losses.append(d_loss)
                self.g_losses.append(g_loss)

                # If at save interval => save generated image samples
                if batch_i % sample_interval == 0:
                    self.sample_images(data_loader, batch_i, run_folder, test_A_file, test_B_file)
                    self.combined.save_weights(os.path.join(run_folder, 'weights/weights-%d.h5' % (self.epoch)))
                    self.combined.save_weights(os.path.join(run_folder, 'weights/weights.h5'))
                    self.save_model(run_folder)
                
            self.epoch += 1
    

    # test_set 으로 뽑아서 데이터 확인 
    def sample_images(self, data_loader, batch_i, run_folder, test_A_file, test_B_file):
        r, c = 2, 4

        for p in range(2):

            if p == 1:
                imgs_A = data_loader.load_data(domain="A", batch_size=1, is_testing=True)
                imgs_B = data_loader.load_data(domain="B", batch_size=1, is_testing=True)
            else:
                imgs_A = data_loader.load_img('data/%s/testA/%s' % (data_loader.dataset_name, test_A_file))
                imgs_B = data_loader.load_img('data/%s/testB/%s' % (data_loader.dataset_name, test_B_file))

            # Translate images to the other domain
            fake_B = self.g_AB.predict(imgs_A)
            fake_A = self.g_BA.predict(imgs_B)
            # Translate back to original domain
            reconstr_A = self.g_BA.predict(fake_B)
            reconstr_B = self.g_AB.predict(fake_A)

            # ID the images
            id_A = self.g_BA.predict(imgs_A)
            id_B = self.g_AB.predict(imgs_B)

            gen_imgs = np.concatenate([imgs_A, fake_B, reconstr_A, id_A, imgs_B, fake_A, reconstr_B, id_B])

            # Rescale images 0 - 1
            gen_imgs = 0.5 * gen_imgs + 0.5
            gen_imgs = np.clip(gen_imgs, 0, 1)

            titles = ['Original', 'Translated', 'Reconstructed', 'ID']
            fig, axs = plt.subplots(r, c, figsize=(25,12.5))
            cnt = 0
            for i in range(r):
                for j in range(c):
                    axs[i,j].imshow(gen_imgs[cnt])
                    axs[i, j].set_title(titles[j])
                    axs[i,j].axis('off')
                    cnt += 1
            fig.savefig(os.path.join(run_folder ,"images/%d_%d_%d.png" % (p, self.epoch, batch_i)))
            plt.close()

    def plot_model(self,run_folder):
        plot_model(self.combined, to_file=os.path.join(run_folder ,'viz/combined.png'), show_shapes = True, show_layer_names = True)
        plot_model(self.d_A, to_file=os.path.join(run_folder ,'viz/d_A.png'), show_shapes = True, show_layer_names = True)
        plot_model(self.d_B, to_file=os.path.join(run_folder ,'viz/d_B.png'), show_shapes = True, show_layer_names = True)
        plot_model(self.g_BA, to_file=os.path.join(run_folder ,'viz/g_BA.png'), show_shapes = True, show_layer_names = True)
        plot_model(self.g_AB, to_file=os.path.join(run_folder ,'viz/g_AB.png'), show_shapes = True, show_layer_names = True)

    def save_plot(self,folder):
        with open(os.path.join(folder, 'params.pkl'), 'wb') as f:
            pkl.dump([
                self.input_dim
                ,  self.learning_rate
                ,  self.buffer_max_length
                ,  self.lambda_validation
                ,  self.lambda_reconstr
                ,  self.lambda_id
                ,  self.generator_type
                ,  self.gen_n_filters
                ,  self.disc_n_filters
                ], f)

        self.plot_model(folder)

    def save_model(self,run_folder):
        self.combined.save(os.path.join(run_folder, 'model.h5')  )
        self.d_A.save(os.path.join(run_folder, 'd_A.h5') )
        self.d_B.save(os.path.join(run_folder, 'd_B.h5') )
        self.g_BA.save(os.path.join(run_folder, 'g_BA.h5')  )
        self.g_AB.save(os.path.join(run_folder, 'g_AB.h5') )

        pkl.dump(self, open( os.path.join(run_folder, "obj.pkl"), "wb" ))

    def load_weights(self,filepath):
        self.combined.load_weights(filepath)



"""
def file_random_select(file_path,batch_size):
            file_names = os.listdir(file_path)
            img_ran = random.choice(file_names,batch_size) 


Data loading 할떄, file_path에서 랜덤하게 batch_size-1 만큼 뽑아준다.
"""