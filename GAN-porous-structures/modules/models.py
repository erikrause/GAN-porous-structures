from keras.layers import (Activation, BatchNormalization, Concatenate, Dense,
                          Embedding, Flatten, Input, Multiply, Reshape, Dropout,
                          Concatenate, Layer, Add)
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Conv2D, Conv2DTranspose, MaxPooling2D, UpSampling2D, AveragePooling2D
from keras.models import Model
from keras.optimizers import Adam
from keras import backend
import tensorflow as tf

import base_models



class WeightedSum(Add):
	# init with default value
	def __init__(self, alpha=0.0, **kwargs):
		super(WeightedSum, self).__init__(**kwargs)
		self.alpha = backend.variable(alpha, name='ws_alpha')

	# output a weighted sum of inputs
	def _merge_function(self, inputs):
		# only supports a weighted sum of two inputs
		assert (len(inputs) == 2)
		# ((1-a) * input1) + (a * input2)
		output = ((1.0 - self.alpha) * inputs[0]) + (self.alpha * inputs[1])
		return output

def add_discriminator_block(old_model, n_input_layers=6, n_filters=64, filter_size=(3,3)):
    old_input_shape = list(old_model.input_shape)
    input_shape = (old_input_shape[-2]*2, 
                   old_input_shape[-2]*2, 
                   old_input_shape[-1])
    input_img = Input(shape=input_shape)
    
    # New block/
    print(n_filters)
    
    d = Conv2D(n_filters, filter_size, padding='same')(input_img)
    d = BatchNormalization()(d)
    d = LeakyReLU(alpha=0.01)(d)
    d = Dropout(rate = droprate)(d)
    d = AveragePooling2D()(d)
  
    n_filters_last = old_model.layers[1].filters  #количество старых фильтров входа
    d = Conv2D(n_filters_last, filter_size, padding='same')(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(alpha=0.01)(d)
    d = Dropout(rate = droprate)(d)
    d = AveragePooling2D()(d)   
    
    block_new = d
    #/New block
    
    for i in range(n_input_layers, len(old_model.layers)):
        print(old_model.layers[i])
        d = old_model.layers[i](d)
        
    model1 = Model(input_img, d)
    
    downsample = AveragePooling2D()(input_img)
    
    block_old = old_model.layers[1](downsample)
    block_old = old_model.layers[2](block_old)
    block_old = old_model.layers[3](block_old)
    block_old = old_model.layers[4](block_old)
    block_old = old_model.layers[5](block_old)
    
    d = WeightedSum()([block_old, block_new])
    
    for i in range(n_input_layers, len(old_model.layers)):
        print(old_model.layers[i])
        d = old_model.layers[i](d)
        
    model2 = Model(input_img, d)
    return [model1, model2]

def add_generator_block(old_model, n_filters=64, filter_size=(3,3)):
    # get the end of the last block
    block_end = old_model.layers[-3].output
    # upsample, and define new block
    upsampling = UpSampling2D()(block_end)
    g = Conv2DTranspose(n_filters, filter_size, padding='same')(upsampling)
    g = BatchNormalization()(g)
    g = LeakyReLU(alpha=0.01)(g)
    
    g = Conv2DTranspose(n_filters, filter_size, padding='same')(g)
    g = BatchNormalization()(g)
    g = LeakyReLU(alpha=0.01)(g)
    # add new output layer
    g = Conv2DTranspose(1, (3,3), padding='same')(g)
    out_image = Activation('tanh')(g)
    # define model
    model1 = Model(old_model.inputs[0], out_image)
    
    # get the output layer from old model
    out_old = old_model.layers[-2]#[-1]
    # connect the upsampling to the old output layer
    out_old = out_old(upsampling)
    out_image2 = old_model.layers[-1](out_old)
    # define new output image as the weighted sum of the old and new models
    merged = WeightedSum()([out_image2, out_image])
    # define model
    model2 = Model(old_model.inputs[0], merged)
    return [model1, model2]

def build_gan(generator, discriminator):

    #model = Sequential()
    input_Z = Input(shape=(z_dim,)) 
    input_C = Input(shape=(1,))

    img = generator([input_Z, input_C])
    
    # Combined Generator -> Discriminator model
    classification = discriminator([img, input_C])
    
    model = Model([input_Z, input_C], classification)

    return model

def build_composite(discriminators, generators):
	model_list = list()
	# create composite models
	for i in range(len(discriminators)):
		g_models, d_models = generators[i], discriminators[i]
		# straight-through model
		d_models[0].trainable = False
		model1 = build_gan(g_models[0], d_models[0])

		model1.compile(loss='binary_crossentropy', optimizer=Adam())
		# fade-in model
		d_models[1].trainable = False
		model2 = build_gan(g_models[1], d_models[1])
		model2.compile(loss='binary_crossentropy', optimizer=Adam())
		# store
		model_list.append([model1, model2])
	return model_list


