from keras.layers import (Activation, BatchNormalization, Concatenate, Dense,
                          Embedding, Flatten, Input, Reshape, Dropout,
                          Concatenate, Layer, Multiply)
from keras.layers.advanced_activations import LeakyReLU, ReLU
from keras.layers.convolutional import Conv2D, Conv2DTranspose, MaxPooling2D, UpSampling2D, AveragePooling2D
from keras.layers.convolutional import Conv3D, Conv3DTranspose, MaxPooling3D, UpSampling3D, AveragePooling3D
from keras.models import Model
from keras.optimizers import Adam, RMSprop
from keras import backend
#import tensorflow as tf
from modules.models.pggan_layers import *
from modules.models.wgangp_layers import *
from keras.initializers import RandomNormal
#from keras.constraints import MinMaxNorm
from keras import initializers
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
global q_model
global EPSILON

q_model = None
EPSILON = 1e-8
# Standart hyperparameters (can be changed)
lr = 0.0005
dis_lr = lr #*2
#opt = RMSprop(lr=lr)    #from vanilla WGAN paper
opt = Adam(lr=lr)        # from Progressive growing GAN paper
dis_opt = Adam(lr=dis_lr)
weight_init = initializers.he_normal()  #RandomNormal(stddev=0.02)
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
            input_z = Input(shape=(z_dim,))
        else:
            input_z = Input(batch_shape=(batch_size, z_dim))
        input_c = Input(shape=(1,))

        #g = Dense(50, kernel_initializer = weight_init)(input_c)
        #g = LeakyReLU(alpha)(g)

        #combined = Concatenate()([input_z, g])

        g = Dense(z_dim)(input_c)
        g = LeakyReLU(alpha)(g)
        g = Multiply()([input_z, g])

        units = n_filters[0]
        channels = self.start_img_shape[-1]   #?
        hidden_shape = tuple(x//(2) for x in (self.start_img_shape[:-1]))
        for i in range(self.dims):
            units = units * hidden_shape[i]
        unints = units * channels   # channles не используется!

        hidden_shape = list(hidden_shape)
        hidden_shape.append(n_filters[0])
        hidden_shape = tuple(hidden_shape)

        #g = PixelNormLayer()(input_z)
        g = Dense(units, kernel_initializer = weight_init)(g)
        g = BatchNormalization()(g)
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
    
        g = self.conv(2, kernel_size=3, strides=1, padding='same', kernel_initializer = weight_init)(g)
        img = Activation('tanh')(g)

        return Model(inputs = [input_z, input_c], outputs = img)


class GAN(Model):
    def __init__(self, generator, discriminator):
        discriminator.trainable = False
        model = self.__build(generator, discriminator)
        Model.__init__(self, model.inputs, model.outputs)
        self.compile(loss='binary_crossentropy', 
                     optimizer=opt)

    def __build(self, generator, discriminator):

        #input_z = Input(shape=(z_dim,)) 
        #input_C = Input(shape=(1,))

        img = generator(generator.inputs)#generator([input_z, input_C])
    
        # Combined Generator -> Discriminator model
        classification = discriminator(img)
    
        model = Model(generator.inputs, classification)

        return model

class WGAN():
    def __init__(self, generator:Model, critic:Model):

        self.generator = generator
        self.critic = critic

        #########################
        # START OF CRITIC BUILD /
        self.generator.trainable = False

        z_disc = Input(batch_shape=(self.generator.input_shape[0]))
        c_disc = Input(batch_shape=(self.generator.input_shape[1]))     # TODO refactor this inputs to generator.input
        #combined_disc = Concatenate()([z_disc, c_disc])
        fake_img = self.generator([z_disc, c_disc])
        real_img = Input(batch_shape=(self.critic.input_shape[0]))

        #prob = self.critic(fake_img)
        fake = self.critic([fake_img, c_disc])
        valid = self.critic([real_img, c_disc])

        interpolated_img = RandomWeightedAverage()([real_img, fake_img])
        validity_interpolated = self.critic([interpolated_img, c_disc])

        GRADIENT_PENALTY_WEIGHT = 10
        partial_gp_loss = partial(gradient_penalty_loss,
                          averaged_samples=interpolated_img,
                          gradient_penalty_weight = GRADIENT_PENALTY_WEIGHT)
        partial_gp_loss.__name__ = 'gradient_penalty' # Keras requires function names
        
        self.critic_model = Model(inputs=[real_img, z_disc, c_disc],
                            outputs=[valid, fake, validity_interpolated])
        self.critic_model.compile(loss=[wasserstein_loss,
                                        wasserstein_loss,
                                        partial_gp_loss],
                                  optimizer=opt)
                                  #loss_weights=[1, 1, 10])
        # / END OF CRITIC BUILD
        #######################

        ############################
        # START OF GENERATOR BUILD /
        self.critic.trainable = False
        self.generator.trainable = True

        z_gen = Input(batch_shape=(self.generator.input_shape[0]))
        c_gen = Input(batch_shape=(self.generator.input_shape[1]))
        #combined_gen = Concatenate()([z_gen, c_gen])
        img = self.generator([z_gen, c_gen])
        valid = self.critic([img, c_gen])

        self.generator_model = Model(inputs=[z_gen, c_gen], outputs=valid)
        self.generator_model.compile(loss=wasserstein_loss,
                                     optimizer=opt)
        # / END OF GENERATOR BUILD BUILD
        ################################

def disc_mutual_info_loss(c_disc, aux_dist):
    """
    Mutual Information lower bound loss for discrete distribution.
    """
    reg_disc_dim = aux_dist.get_shape().as_list()[-1]
    cross_ent = - K.mean( K.sum( K.log(aux_dist + EPSILON) * c_disc, axis=1 ) )
    ent = - K.mean( K.sum( K.log(1./reg_disc_dim + EPSILON) * c_disc, axis=1 ) )
    return ent - cross_ent

def mutual_info_loss(c, c_given_x):
        """The mutual information metric we aim to minimize"""
        eps = 1e-8
        conditional_entropy = K.mean(- K.sum(K.log(c_given_x + eps) * c, axis=1))
        entropy = K.mean(- K.sum(K.log(c + eps) * c, axis=1))

        return conditional_entropy + entropy

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
        input_c = Input(shape=(1,), name='Input_C')

        c_channel_reshape = tuple(x for x in (img_shape[:]))
        c_dense_units = 1
        for i in range(0, 3):
            c_dense_units = c_dense_units * c_channel_reshape[i]

        d = Dense(c_dense_units)(input_c)
        d = LeakyReLU(alpha)(d)
        #d = LayerNormalization()(d)

        d = Reshape(c_channel_reshape)(d)
        d = Concatenate()([input_img, d])
    
        d = self.conv(n_filters[1], kernel_size=filters_sizes[0], strides = 1, padding='same', name='concat_layer', kernel_initializer = weight_init)(d)
        #d = BatchNormalization()(d)
        d = LeakyReLU(alpha)(d) 
        #d = LayerNormalization()(d)

        # 3 implementations of MinibatchStdDev layer:
        #d = MinibatchStdev()(d)    # machinelearningmastery
        d = MinibatchStatConcatLayer()(d)   # MSC-BUAA
        #d = minibatch_std_layer(d)     # Manning

        d = self.conv(n_filters[0], kernel_size=filters_sizes[0], strides = 1, padding='same', kernel_initializer = weight_init)(d)
        #d = BatchNormalization()(d)
        d = LeakyReLU(alpha)(d)
        #d = LayerNormalization()(d)
        d = self.pool()(d)
        
        #d = self.pool()(d)
    
        conv_end = Flatten()(d)

        #combined = Concatenate(name='Concat_input_C')([conv_end, input_c])    

        #d = Dense(512, kernel_initializer = weight_init, name='dense')(combined)
        #d = BatchNormalization()(d)
        #d = LeakyReLU(alpha)(d)

        #d = Dense(128, kernel_initializer = weight_init)(d)
        #d = BatchNormalization()(d)
        #d = LeakyReLU(alpha)(d)
    
        d = Dense(1, activation='linear')(conv_end) 

        return Model(inputs=[input_img, input_c], outputs=d)