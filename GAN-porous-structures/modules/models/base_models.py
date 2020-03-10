from keras.layers import (Activation, BatchNormalization, Concatenate, Dense,
                          Embedding, Flatten, Input, Reshape, Dropout,
                          Concatenate, Layer)
from keras.layers.advanced_activations import LeakyReLU, ReLU
from keras.layers.convolutional import Conv2D, Conv2DTranspose, MaxPooling2D, UpSampling2D, AveragePooling2D
from keras.layers.convolutional import Conv3D, Conv3DTranspose, MaxPooling3D, UpSampling3D, AveragePooling3D
from keras.models import Model
from keras.optimizers import Adam, RMSprop
from keras import backend
#import tensorflow as tf
from modules.models.layers import *
from keras.initializers import RandomNormal
from keras.constraints import Constraint
#from keras.constraints import MinMaxNorm
from keras import initializers
from functools import partial

from keras.layers.merge import _Merge

global constraint
global clip_value
global opt
global dis_opt
global weight_init
global lr
global dis_lr
global alpha
global batch_size

constraint = None
clip_value = 0.01
current_backend = backend.backend()

lr = 0.0005
dis_lr = lr*2
#opt = RMSprop(lr=lr)    #from vanilla WGAN paper
opt = Adam(lr=lr)        # from Progressive growing GAN paper
dis_opt = Adam(lr=dis_lr)
weight_init = initializers.he_normal()  #RandomNormal(stddev=0.02)
alpha = 0.2
batch_size = 16

# clip model weights to a given hypercube (vanilla WGAN)
class ClipConstraint(Constraint):
	# set clip value when initialized
	def __init__(self, clip_value):
		self.clip_value = clip_value

	# clip model weights to hypercube
	def __call__(self, weights):
		return backend.clip(weights, -self.clip_value, self.clip_value)

	# get the config
	def get_config(self):
		return {'clip_value': self.clip_value}

if current_backend == 'tensorflow':
    constraint = ClipConstraint(clip_value)      # WGAN constraint for critics kernel_constraint.
    print(current_backend)
elif current_backend == 'plaidml':
    print(current_backend)              # With plaidml constrainsts will do in train loop.
else:
    print(' WARNING: BACKEND ' + current_backend + ' NOT SUPPORTED')

# mini-batch standard deviation layer
class MinibatchStdev(Layer):
    # initialize the layer
    def __init__(self, **kwargs):
        super(MinibatchStdev, self).__init__(**kwargs)

    # perform the operation
    def call(self, inputs):
        # calculate the mean value for each pixel across channels
        mean = backend.mean(inputs, axis=0, keepdims=True)
        # calculate the squared differences between pixel values and mean
        squ_diffs = backend.square(inputs - mean)
        # calculate the average of the squared differences (variance)
        mean_sq_diff = backend.mean(squ_diffs, axis=0, keepdims=True)
        # add a small value to avoid a blow-up when we calculate stdev
        mean_sq_diff += 1e-8
        # square root of the variance (stdev)
        stdev = backend.sqrt(mean_sq_diff)
        # calculate the mean standard deviation across each pixel coord
        mean_pix = backend.mean(stdev, keepdims=True)
        # scale this up to be the size of one input feature map for each sample
        shape = backend.shape(inputs)
        output = backend.tile(mean_pix, (shape[0], shape[1], shape[2], shape[3], 1))
        # concatenate with the output
        combined = backend.concatenate([inputs, output], axis=-1)
        return combined   # was combined

	# define the output shape of the layer
    def compute_output_shape(self, input_shape):
        # create a copy of the input shape as a list
        input_shape = list(input_shape)
        # add one to the channel dimension (assume channels-last)
        input_shape[-1] += 1
        # convert list to a tuple
        return tuple(input_shape)

def minibatch_std_layer(layer, group_size=4):
    '''
    Will calculate minibatch standard deviation for a layer.
    Will do so under a pre-specified tf-scope with Keras.
    Assumes layer is a float32 data type. Else needs validation/casting.
    NOTE: there is a more efficient way to do this in Keras, but just for
    clarity and alignment with major implementations (for understanding) 
    this was done more explicitly. Try this as an exercise.
    '''
    # Hint!
    # If you are using pure Tensorflow rather than Keras, always remember scope
    # minibatch group must be divisible by (or <=) group_size
    group_size = K.minimum(group_size, K.shape(layer)[0])

    # just getting some shape information so that we can use
    # them as shorthand as well as to ensure defaults
    input = layer
    shape = list(K.int_shape(input))
    shape[0] = K.shape(input)[0]

    # Reshaping so that we operate on the level of the minibatch
    # in this code we assume the layer to be:
    # [Group (G), Minibatch (M), Width (W), Height (H) , Channel (C)]
    # but be careful different implementations use the Theano specific
    # order instead
    minibatch = K.reshape(layer, (group_size, -1, shape[1], shape[2], shape[3], shape[4]))

    # Center the mean over the group [M, W, H, C]
    minibatch -= K.mean(minibatch, axis=0, keepdims=True)
    # Calculate the variance of the group [M, W, H, C]
    minibatch = K.mean(K.square(minibatch), axis = 0)
    # Calculate the standard deviation over the group [M,W,H,C]
    minibatch = K.square(minibatch + 1e8)
    # Take average over feature maps and pixels [M,1,1,1]
    minibatch = K.mean(minibatch, axis=[1,2,3], keepdims=True)
    # Add as a layer for each group and pixels
    minibatch = K.tile(minibatch, [group_size, shape[1], shape[2], shape[3], shape[4]])
    # Append as a new feature map
    return K.concatenate([layer, minibatch], axis=1)

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
  
        input_Z = Input(batch_shape=(16,z_dim))
        #input_C = Input(shape=(1,))

        #combined = Concatenate()([input_Z, input_C])

        units = 64
        channels = self.start_img_shape[-1]   #?
        hidden_shape = tuple(x//(2) for x in (self.start_img_shape[:-1]))
        for i in range(self.dims):
            units = units * hidden_shape[i]
        unints = units * channels   # channles не используется!

        hidden_shape = list(hidden_shape)
        hidden_shape.append(64)
        hidden_shape = tuple(hidden_shape)

        #hidden_shape = (16,1,1,1,512)
        #units = 512

        #g = PixelNormLayer()(input_Z)
        g = Dense(units)(input_Z)
        g = LeakyReLU(alpha)(g)
        #g = PixelNormLayer()(g)
        g = Reshape(hidden_shape)(g)
        #g = self.upsample()(g)

        #g = self.conv(128, kernel_size=3, strides=1, padding='same', kernel_initializer = weight_init)(g)
        #g = BatchNormalization()(g)
        #g = LeakyReLU(alpha)(g)
        #g = PixelNormLayer()(g)

        g = self.upsample()(g)

        g = self.conv(64, kernel_size=3, strides=1, padding='same', kernel_initializer = weight_init)(g)
        g = BatchNormalization()(g)
        g = LeakyReLU(alpha)(g)
        #g = PixelNormLayer()(g)
        #g = self.upsample()(g)

        g = self.conv(32, kernel_size=3, strides=1, padding='same', kernel_initializer = weight_init)(g)
        g = BatchNormalization()(g)
        g = LeakyReLU(alpha)(g)
  
        #g = self.conv(32, kernel_size=3, strides=1, padding='same', kernel_initializer = weight_init)(g)
        #g = BatchNormalization()(g)
        #g = LeakyReLU(alpha)(g)
        #g = PixelNormLayer()(g)
        #g = self.upsample()(g)
    
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
        #discriminator.trainable = False
        #model = self.__build(generator, discriminator)

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
        #######################

class Critic(Model):
    def __init__(self, img_shape=None, inputs = None, outputs = None, alpha = 0.2, droprate = 0.2):

        self.dims = len(img_shape) - 1
        if self.dims == 3:
            self.conv = Conv3D
            self.pool = AveragePooling3D
        elif self.dims == 2:
            self.conv = Conv2D
            self.pool = AveragePooling2D

        if outputs == None:
            self.alpha = 0.2
            self.droprate = droprate
            model = self.__build(img_shape)
            Model.__init__(self, model.inputs, model.outputs)
        elif outputs != None:
            Model.__init__(self, inputs, outputs)

        self.compile(loss=wasserstein_loss,
                      optimizer=dis_opt)

    def __build(self, img_shape:tuple):

        input_img = Input(shape = img_shape)
        #input_C = Input(shape=(1,), name='Input_C')
    
        #d = conv(32, kernel_size=1, strides = 1, padding='same', name='concat_layer')(input_img)
        #d = LeakyReLU(alpha = self.alpha)(d) 
        #d = AveragePooling2D()(d)
    
        d = self.conv(32, kernel_size=3, strides = 1, padding='same', name='concat', kernel_initializer = weight_init, kernel_constraint=constraint)(input_img)
        d = BatchNormalization()(d)
        d = LeakyReLU(alpha = self.alpha)(d)
        #d = Dropout(rate = self.droprate)(d)
        d = self.pool()(d)

        d = self.conv(64, kernel_size=3, strides = 1, padding='same', name = 'conv', kernel_initializer = weight_init, kernel_constraint=constraint)(d)
        d = BatchNormalization()(d)
        d = LeakyReLU(alpha = self.alpha)(d)
        #d = Dropout(rate = self.droprate)(d)
        d = self.pool()(d)
    
        d = Flatten()(d)

        #combined = Concatenate(name='Concat_input_C')([d, input_C])    

        d = Dense(128, kernel_initializer = weight_init, name='dense', kernel_constraint=constraint)(d)
        d = BatchNormalization()(d)
        d = ReLU()(d)
        #d = Dropout(rate = self.droprate)(d)
    
        d = Dense(1, activation='linear', kernel_constraint=constraint)(d)      #нужен ли constraint для Dense?


        model = Model(inputs=input_img, outputs=d)
    
        return model

class Discriminator(Model):
    def __init__(self, img_shape=None, inputs = None, outputs = None, alpha = 0.2, droprate = 0.2):

        self.dims = len(img_shape) - 1
        if self.dims == 3:
            self.conv = Conv3D
            self.pool = AveragePooling3D
        elif self.dims == 2:
            self.conv = Conv2D
            self.pool = AveragePooling2D

        if outputs == None:
            self.alpha=0.2
            self.droprate = droprate
            model = self.__build(img_shape)
            Model.__init__(self, model.inputs, model.outputs)
        elif outputs != None:
            Model.__init__(self, inputs, outputs)
        
        #Original GAN (needs batchnorm layers):
        #self.compile(loss='binary_crossentropy',
        #             metrics=['accuracy'],
        #             optimizer=dis_opt)


    def __build(self, img_shape:tuple):

        input_img = Input(batch_shape=(16,8,8,8,1))#img_shape)
        #input_C = Input(shape=(1,), name='Input_C')
    
        d = self.conv(32, kernel_size=3, strides = 1, padding='same', name='concat_layer', kernel_initializer = weight_init)(input_img)
        d = BatchNormalization()(d)
        d = LeakyReLU(alpha = self.alpha)(d) 

        #d = MinibatchStdev()(d)
        d = MinibatchStatConcatLayer()(d)   # MSC-BUAA
        #d = minibatch_std_layer(d)     # Manning

        #d = self.conv(32, kernel_size=3, strides = 1, padding='same', name='concat', kernel_initializer = weight_init)(d)
        #d = BatchNormalization()(d)
        #d = LeakyReLU(alpha = self.alpha)(d)
        #d = self.pool()(d)

        d = self.conv(64, kernel_size=3, strides = 1, padding='same', kernel_initializer = weight_init)(d)
        d = BatchNormalization()(d)
        d = LeakyReLU(alpha = self.alpha)(d)
        d = self.pool()(d)

        #d = self.conv(128, kernel_size=3, strides = 1, padding='same', kernel_initializer = weight_init)(d)
        #d = BatchNormalization()(d)
        #d = LeakyReLU(alpha = self.alpha)(d)
        
        #d = self.pool()(d)
    
        d = Flatten()(d)

        #combined = Concatenate(name='Concat_input_C')([d, input_C])    

        d = Dense(128, kernel_initializer = weight_init, name='dense')(d)
        d = BatchNormalization()(d)
        d = LeakyReLU(alpha = self.alpha)(d)
    
        d = Dense(1, activation='sigmoid')(d) 


        model = Model(inputs=input_img, outputs=d)
    
        return model

class RandomWeightedAverage(_Merge):
    """Takes a randomly-weighted average of two tensors. In geometric terms, this
    outputs a random point on the line between each pair of input points.
    Inheriting from _Merge is a little messy but it was the quickest solution I could
    think of. Improvements appreciated."""

    def _merge_function(self, inputs):
        weights = K.random_uniform((batch_size, 1, 1, 1, 1))
        return (weights * inputs[0]) + ((1 - weights) * inputs[1])

def gradient_penalty_loss(y_true, y_pred, averaged_samples,
                          gradient_penalty_weight):
    """Calculates the gradient penalty loss for a batch of "averaged" samples.
    In Improved WGANs, the 1-Lipschitz constraint is enforced by adding a term to the
    loss function that penalizes the network if the gradient norm moves away from 1.
    However, it is impossible to evaluate this function at all points in the input
    space. The compromise used in the paper is to choose random points on the lines
    between real and generated samples, and check the gradients at these points. Note
    that it is the gradient w.r.t. the input averaged samples, not the weights of the
    discriminator, that we're penalizing!
    In order to evaluate the gradients, we must first run samples through the generator
    and evaluate the loss. Then we get the gradients of the discriminator w.r.t. the
    input averaged samples. The l2 norm and penalty can then be calculated for this
    gradient.
    Note that this loss function requires the original averaged samples as input, but
    Keras only supports passing y_true and y_pred to loss functions. To get around this,
    we make a partial() of the function with the averaged_samples argument, and use that
    for model training."""
    # first get the gradients:
    #   assuming: - that y_pred has dimensions (batch_size, 1)
    #             - averaged_samples has dimensions (batch_size, nbr_features)
    # gradients afterwards has dimension (batch_size, nbr_features), basically
    # a list of nbr_features-dimensional gradient vectors
    gradients = K.gradients(y_pred, averaged_samples)[0]
    # compute the euclidean norm by squaring ...
    gradients_sqr = K.square(gradients)
    #   ... summing over the rows ...
    gradients_sqr_sum = K.sum(gradients_sqr,
                              axis=np.arange(1, len(gradients_sqr.shape)))
    #   ... and sqrt
    gradient_l2_norm = K.sqrt(gradients_sqr_sum)
    # compute lambda * (1 - ||grad||)^2 still for each single sample
    gradient_penalty = gradient_penalty_weight * K.square(1 - gradient_l2_norm)
    # return the mean as loss over all the batch samples
    return K.mean(gradient_penalty)

def wasserstein_loss(y_true, y_pred):
    """Calculates the Wasserstein loss for a sample batch.
    The Wasserstein loss function is very simple to calculate. In a standard GAN, the
    discriminator has a sigmoid output, representing the probability that samples are
    real or generated. In Wasserstein GANs, however, the output is linear with no
    activation function! Instead of being constrained to [0, 1], the discriminator wants
    to make the distance between its output for real and generated samples as
    large as possible.
    The most natural way to achieve this is to label generated samples -1 and real
    samples 1, instead of the 0 and 1 used in normal GANs, so that multiplying the
    outputs by the labels will give you the loss immediately.
    Note that the nature of this loss means that it can be (and frequently will be)
    less than 0."""
    return K.mean(y_true * y_pred)