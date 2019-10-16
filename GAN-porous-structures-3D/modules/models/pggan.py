from keras.layers import (Activation, BatchNormalization, Concatenate, Dense,
                          Embedding, Flatten, Input, Multiply, Reshape, Dropout,
                          Concatenate, Layer, Add)
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Conv3D, Conv3DTranspose, MaxPooling3D, UpSampling3D, AveragePooling3D
from keras.models import Model
from keras.optimizers import Adam
from keras import backend
import tensorflow as tf

from modules.models import base_models

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

def update_fadein(models, step, n_steps):
    # calculate current alpha (linear from 0 to 1)
    alpha = step / float(n_steps - 1)
    # update the alpha for each model
    for model in models:
        for layer in model.layers:
            if isinstance(layer, WeightedSum):
                backend.set_value(layer.alpha, alpha)

def add_discriminator_block(old_model: Model, n_input_layers=6, n_filters=64, filter_size=3):
    old_input_shape = list(old_model.input_shape)
    input_img_shape = (old_input_shape[0][-2]*2, 
                   old_input_shape[0][-2]*2,
                   old_input_shape[0][-2]*2,
                   old_input_shape[0][-1])
    input_img = Input(shape=input_img_shape)
    
    # New block/
    print(n_filters)
    
    d = Conv3D(n_filters, kernel_size=filter_size, padding='same')(input_img)
    d = BatchNormalization()(d)
    d = LeakyReLU(alpha=0.01)(d)
    d = Dropout(rate = 0.2)(d)
    d = AveragePooling3D()(d)
  
    n_filters_last = old_model.layers[1].filters  #количество старых фильтров входа
    d = Conv3D(n_filters_last, kernel_size=filter_size, padding='same')(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(alpha=0.01)(d)
    d = Dropout(rate = 0.2)(d)
    d = AveragePooling3D()(d)   
    
    block_new = d
    #/New block
    
    for i in range(n_input_layers, len(old_model.layers)):
        current_layer = old_model.layers[i]
        print(current_layer)
        if current_layer.name == 'Input_C':
            input_C = current_layer.input
            continue
        elif current_layer.name == 'Concat_input_C':
            d = current_layer([d, input_C])

        else:
            d = current_layer(d)
        
    straight_model = Model([input_img, input_C], d) #base_models.Discriminator
    straight_model.compile(loss='binary_crossentropy',
                      optimizer=Adam(),
                      metrics=['accuracy'])

    downsample = AveragePooling3D()(input_img)
    
    block_old = downsample
    for i in range(1, n_input_layers):
        block_old = old_model.layers[i](block_old)
    
    d = WeightedSum()([block_old, block_new])
    
    for i in range(n_input_layers, len(old_model.layers)):
        current_layer = old_model.layers[i]
        print(current_layer)
        if current_layer.name == 'Input_C':
            input_C = current_layer.input
            continue
        elif current_layer.name == 'Concat_input_C':
            d = current_layer([d, input_C])

        else:
            d = current_layer(d)
        
    fadein_model = Model([input_img, input_C], d)
    fadein_model.compile(loss='binary_crossentropy',
                      optimizer=Adam(),
                      metrics=['accuracy'])

    return [straight_model, fadein_model]

def add_generator_block(old_model, n_filters=64, filter_size=3):
    # get the end of the last block
    block_end = old_model.layers[-3].output
    # upsample, and define new block
    upsampling = UpSampling3D()(block_end)
    g = Conv3DTranspose(n_filters, kernel_size=filter_size, padding='same')(upsampling)
    g = BatchNormalization()(g)
    g = LeakyReLU(alpha=0.01)(g)
    
    g = Conv3DTranspose(n_filters, kernel_size=filter_size, padding='same')(g)
    g = BatchNormalization()(g)
    g = LeakyReLU(alpha=0.01)(g)
    # add new output layer
    g = Conv3DTranspose(1, kernel_size=3, padding='same')(g)
    out_image = Activation('tanh')(g)
    # define model
    straight_model = Model(old_model.inputs, out_image)
    
    # get the output layer from old model
    out_old = old_model.layers[-2]#[-1]
    # connect the upsampling to the old output layer
    out_old = out_old(upsampling)
    out_image2 = old_model.layers[-1](out_old)
    # define new output image as the weighted sum of the old and new models
    merged = WeightedSum()([out_image2, out_image])
    # define model
    fadein_model = Model(old_model.inputs, merged)
    return [straight_model, fadein_model]

def build_composite(discriminators, generators):
	model_list = list()
	# create composite models
	for i in range(len(discriminators)):
		g_models, d_models = generators[i], discriminators[i]
		# straight-through model
		straight_model = base_models.GAN(g_models[0], d_models[0])
		# fade-in model
		fadein_model = base_models.GAN(g_models[1], d_models[1])
		# store
		model_list.append([straight_model, fadein_model])
	return model_list


