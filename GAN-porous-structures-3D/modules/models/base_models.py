from keras.layers import (Activation, BatchNormalization, Concatenate, Dense,
                          Embedding, Flatten, Input, Multiply, Reshape, Dropout,
                          Concatenate, Layer, Add)
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Conv3D, Conv3DTranspose, MaxPooling3D, UpSampling3D, AveragePooling3D
from keras.models import Model
from keras.optimizers import Adam
from keras import backend
import tensorflow as tf

class Generator(Model):
    def __init__(self, z_dim):
        model = self.__build(z_dim)
        Model.__init__(self, model.inputs, model.outputs)

    def __build(self, z_dim):
  
        input_Z = Input(shape=(z_dim,))
        input_C = Input(shape=(1,))

        combined = Concatenate()([input_Z, input_C])
    
        g = Dense(64 * 4 * 4 * 4)(combined)
        g = Reshape((4, 4, 4, 64))(g)
  
        g = Conv3DTranspose(64, kernel_size=3, strides=1, padding='same')(g)
        g = BatchNormalization()(g)
        g = LeakyReLU(alpha=0.01)(g)
        g = UpSampling3D()(g)
    
        g = Conv3DTranspose(32, kernel_size=3, strides=1, padding='same')(g)
        g = BatchNormalization()(g)
        g = LeakyReLU(alpha=0.01)(g)
    
        g = Conv3DTranspose(1, kernel_size=3, strides=1, padding='same')(g)
        img = Activation('tanh')(g)

        return Model(inputs = [input_Z, input_C], outputs = img)

class Discriminator(Model):
    def __init__(self, img_shape:tuple):
        model = self.__build(img_shape)
        Model.__init__(self, model.inputs, model.outputs)
        self.compile(loss='binary_crossentropy',
                      optimizer=Adam(),
                      metrics=['accuracy'])

    def __build(self, img_shape:tuple):

        input_img = Input(shape = img_shape)
        input_C = Input(shape=(1,), name='Input_C')
    
        d = Conv3D(16, kernel_size=3, padding='same', name='concat_layer')(input_img)
        d = BatchNormalization()(d)
        d = LeakyReLU(alpha=0.01)(d) 
        d = Dropout(rate = 0.2)(d)
        d = AveragePooling3D()(d)
    
        d = Conv3D(16, kernel_size=3, padding='same')(d)
        #d = BatchNormalization()(d)
        #d = LeakyReLU(alpha=0.01)(d)
        #d = Dropout(rate = 0.2)(d)
        #d = AveragePooling3D()(d)
    
        d = Flatten()(d)

        combined = Concatenate(name='Concat_input_C')([d, input_C])    

        #d = Dense(128)(combined)
        d = BatchNormalization()(combined)
        d = LeakyReLU(alpha=0.01)(d)
        d = Dropout(rate = 0.2)(d)
    
        d = Dense(1, activation='sigmoid')(d)


        model = Model(inputs=[input_img, input_C], outputs=d)
    
        return model

class GAN(Model):
    def __init__(self, generator, discriminator):
        discriminator.trainable = False
        model = self.__build(generator, discriminator)
        Model.__init__(self, model.inputs, model.outputs)
        self.compile(loss='binary_crossentropy', 
                     optimizer=Adam())

    def __build(self, generator, discriminator):

        #input_Z = Input(shape=(z_dim,)) 
        #input_C = Input(shape=(1,))

        img = generator(generator.inputs)#generator([input_Z, input_C])
    
        # Combined Generator -> Discriminator model
        classification = discriminator([img, generator.inputs[1]])
    
        model = Model(generator.inputs, classification)

        return model