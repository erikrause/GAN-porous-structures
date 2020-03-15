from tensorflow.keras.layers import (Activation, BatchNormalization, Concatenate, Dense,
                          Embedding, Flatten, Input, Reshape, Dropout,
                          Concatenate, Layer, LeakyReLU, ReLU,
                          Conv3D, Conv3DTranspose, MaxPooling3D, UpSampling3D, AveragePooling3D,
                          Conv2D, Conv2DTranspose, MaxPooling2D, UpSampling2D, AveragePooling2D)
#from keras.layers.advanced_activations import LeakyReLU, ReLU
#from keras.layers.convolutional import Conv2D, Conv2DTranspose, MaxPooling2D, UpSampling2D, AveragePooling2D
#from keras.layers.convolutional import Conv3D, Conv3DTranspose, MaxPooling3D, UpSampling3D, AveragePooling3D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras import backend
#import tensorflow as tf
from modules.models.pggan_layers import *
from modules.models.wgangp_layers import *
from tensorflow.keras.initializers import RandomNormal, he_normal
#from keras.constraints import MinMaxNorm
from tensorflow.keras import initializers
from functools import partial

import tensorflow as tf

global opt
global dis_opt
global weight_init
global lr
global dis_lr
global alpha
global batch_size
global conv_per_res     # number of conv layers per resolution
global n_filters
global filters_sizes

# Standart hyperparameters (can be changed)
lr = 0.0005
dis_lr = lr #*2
#opt = RMSprop(lr=lr)    #from vanilla WGAN paper
opt = Adam(lr=lr)        # from Progressive growing GAN paper
dis_opt = Adam(lr=dis_lr)
weight_init = he_normal()  #RandomNormal(stddev=0.02)
alpha = 0.2
batch_size = 16
conv_per_res = 1
n_filters = []
filters_sizes = []

class Generator(Model):
    def __init__(self, inputs, start_img_shape, outputs = None):

        self.start_img_shape = start_img_shape
        self.dims = len(self.start_img_shape) - 1

        if self.dims == 3:
            self.conv = Conv3DTranspose
            self.upsample = UpSampling3D
        elif self.dims == 2:
            self.conv = Conv2DTranspose
            self.upsample = UpSampling2D

        if outputs == None:
            model = self.__build(inputs)
            Model.__init__(self, model.inputs, model.outputs)
        elif outputs != None:
            Model.__init__(self, inputs, outputs)

    def __build(self, z_dim):
        
        if batch_size == None:
            input_Z = Input(shape=(z_dim,))
        else:
            input_Z = Input(batch_shape=(batch_size, z_dim))
        #input_C = Input(shape=(1,))

        #combined = Concatenate()([input_Z, input_C])

        units = n_filters[0]
        channels = self.start_img_shape[-1]   #?
        hidden_shape = tuple(x//(2) for x in (self.start_img_shape[:-1]))
        for i in range(self.dims):
            units = units * hidden_shape[i]
        unints = units * channels   # channles не используется!

        hidden_shape = list(hidden_shape)
        hidden_shape.append(n_filters[0])
        hidden_shape = tuple(hidden_shape)

        #g = PixelNormLayer()(input_Z)
        g = Dense(units)(input_Z)
        g = LeakyReLU(alpha)(g)
        #g = PixelNormLayer()(g)
        g = Reshape(hidden_shape)(g)
        #g = self.upsample()(g)

        g = self.upsample()(g)

        g = self.conv(n_filters[0], kernel_size=filters_sizes[0], strides=1, padding='same', kernel_initializer = weight_init)(g)
        g = BatchNormalization()(g)
        g = LeakyReLU(alpha)(g)
        #g = PixelNormLayer()(g)
        #g = self.upsample()(g)

        g = self.conv(n_filters[1], kernel_size=filters_sizes[0], strides=1, padding='same', kernel_initializer = weight_init)(g)
        g = BatchNormalization()(g)
        g = LeakyReLU(alpha)(g)
    
        g = self.conv(1, kernel_size=3, strides=1, padding='same', kernel_initializer = weight_init)(g)
        img = Activation('tanh')(g)

        return Model(inputs = input_Z, outputs = img)



class GAN(Model):
    def __init__(self, generator, discriminator):
        discriminator.trainable = False
        model = self.__build(generator, discriminator)
        Model.__init__(self, model.inputs, model.outputs)
        self.compile(loss='binary_crossentropy', 
                     optimizer=opt)

    def __build(self, generator, discriminator):

        #input_Z = Input(shape=(z_dim,)) 
        #input_C = Input(shape=(1,))

        img = generator(generator.inputs)#generator([input_Z, input_C])
    
        # Combined Generator -> Discriminator model
        classification = discriminator(img)
    
        model = Model(generator.inputs, classification)

        return model

class WGAN():
    def __init__(self, generator:Model, critic:Model):

        self.generator = generator
        self.critic = critic

        #########################
        # / START OF CRITIC BUILD
        self.generator.trainable = False

        z_disc = Input(batch_shape=(self.generator.input_shape))
        fake_img = self.generator(z_disc)
        real_img = Input(batch_shape=(self.critic.input_shape))

        fake = self.critic(fake_img)
        valid = self.critic(real_img)

        interpolated_img = RandomWeightedAverage()([real_img, fake_img])
        validity_interpolated = self.critic(interpolated_img)

        GRADIENT_PENALTY_WEIGHT = 10
        partial_gp_loss = partial(gradient_penalty_loss,
                          averaged_samples=interpolated_img,
                          gradient_penalty_weight = GRADIENT_PENALTY_WEIGHT)
        partial_gp_loss.__name__ = 'gradient_penalty' # Keras requires function names
        
        self.critic_model = Model(inputs=[real_img, z_disc],
                            outputs=[valid, fake, validity_interpolated])
        self.critic_model.compile(loss=[wasserstein_loss,
                                        wasserstein_loss,
                                        partial_gp_loss],
                                        optimizer=opt)
                                        #loss_weights=[1, 1, 10])
        # / END OF CRITIC BUILD
        #######################

        ############################
        # / START OF GENERATOR BUILD
        self.critic.trainable = False
        self.generator.trainable = True

        z_gen = Input(batch_shape=(self.generator.input_shape))
        img = self.generator(z_gen)
        valid = self.critic(img)
        self.generator_model = Model(z_gen, valid)
        self.generator_model.compile(loss=wasserstein_loss,
                                     optimizer=opt)
        # / END OF GENERATOR BUILD BUILD
        ################################

class Discriminator(Model):
    def __init__(self, img_shape=None, inputs = None, outputs = None):

        self.dims = len(img_shape) - 1
        if self.dims == 3:
            self.conv = Conv3D
            self.pool = AveragePooling3D
        elif self.dims == 2:
            self.conv = Conv2D
            self.pool = AveragePooling2D

        if outputs == None:
            model = self.__build(img_shape)
            Model.__init__(self, model.inputs, model.outputs)
        elif outputs != None:
            Model.__init__(self, inputs, outputs)
        
        # Vanilla GAN:
        #self.compile(loss='binary_crossentropy',
        #             metrics=['accuracy'],
        #             optimizer=dis_opt)
        # (+needs batchnorm layers)


    def __build(self, img_shape:tuple):

        if batch_size != None:
            batch_shape = [batch_size]
            for i in range (0,len(img_shape)):
                batch_shape.append(img_shape[i])

            input_img = Input(batch_shape=batch_shape)#(batch_size,8,8,8,1))#img_shape)
        else:
            input_img = Input(shape=img_shape)
        #input_C = Input(shape=(1,), name='Input_C')
        
        d = self.conv(n_filters[1], kernel_size=filters_sizes[0], strides = 1, padding='same', name='concat_layer', kernel_initializer = weight_init)(input_img)
        #d = BatchNormalization()(d)
        d = LeakyReLU(alpha)(d) 

        # 3 implementations of MinibatchStdDev layer:
        #d = MinibatchStdev()(d)    # machinelearningmastery
        d = MinibatchStatConcatLayer()(d)   # MSC-BUAA
        #d = minibatch_std_layer(d)     # Manning

        d = self.conv(n_filters[0], kernel_size=filters_sizes[0], strides = 1, padding='same', kernel_initializer = weight_init)(d)
        #d = BatchNormalization()(d)
        d = LeakyReLU(alpha)(d)
        d = self.pool()(d)
        
        #d = self.pool()(d)
    
        d = Flatten()(d)

        #combined = Concatenate(name='Concat_input_C')([d, input_C])    

        #d = Dense(128, kernel_initializer = weight_init, name='dense')(d)
        #d = BatchNormalization()(d)
        #d = LeakyReLU(alpha = self.alpha)(d)
    
        d = Dense(1, activation='linear')(d) 


        model = Model(inputs=input_img, outputs=d)
    
        return model

