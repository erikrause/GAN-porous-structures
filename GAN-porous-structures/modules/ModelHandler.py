from modules.models import pggan, base_models
from modules.preprocess import DataLoader
import pickle
import tensorflow as tf
import numpy as np
from keras import backend
import time
from PIL import Image
import matplotlib.pyplot as plt
from keras.utils import plot_model

import os.path

class ModelHandler():
    def __init__(self, directory:str, start_shape:tuple, z_dim:int, n_blocks:int, n_filters, filter_sizes, data_loader:DataLoader, weights_dir=''):     #, discriminators, generators, gans):
        directory = directory + 'output/'
        self.directory = directory + 'History'
        self.samples_dir = directory + 'samples'
        # Models initialize:
        self.models = dict()
        self.discriminators = []
        self.generators = [] 
        self.gans = []
        self.n_blocks = n_blocks
        self.start_shape = start_shape
        upscale = 2**(n_blocks-1)
        self.end_shape = (start_shape[0]*upscale,
                          start_shape[1]*upscale,
                          start_shape[2])
        self.z_dim = z_dim
        self.build_models(start_shape, z_dim, n_filters, filter_sizes)
        #
        # Logs:
        self.d_losses = []
        self.g_losses = []
        self.d_acces = []
        #
        # For current metrics:
        self.d_loss = []
        self.g_loss = []
        self.d_acc = []
        self.iteration = 0
        self.model_iteration = 0
        self.is_fadein = False
        self.is_logs_loaded = False
        self.parameters = dict()
        #
        # Train params:
        #self.batch_size = 64
        #self.sample_interval = 100
        self.data_loader = data_loader
        self.z_global = np.random.normal(0, 1, (1, self.z_dim))
        #

        if weights_dir != '':
            self.load_models_weights_from_dir(weights_dir)
            self.iteration = input('print start iteration')
            self.model_iteration = input('print start model_iteration')

        elif self.__check_log_files():
            self.d_losses = self.load_from_file('/d_losses.log')
            self.d_loss = self.d_losses[-1][0]
            self.g_losses = self.load_from_file('/g_losses.log')
            self.g_loss = self.g_losses[-1][0]
            self.d_acces = self.load_from_file('/d_acces.log')
            self.d_acc = self.d_acces[-1][0]
            self.iteration = self.d_losses[-1][-3]
            self.model_iteration = self.d_losses[-1][-2]
            self.is_fadein = bool(self.d_losses[-1][-1])
            self.is_logs_loaded = True  # Не используется
            self.parameters = self.load_from_file('/parameters')
            # Initialise alpha from last checkpoint:
            pggan.update_fadein(self.discriminators[model_iteration][is_fadein])
            #################
            #self.iteration = 0  #debug
            #self.model_iteration = 2    #debug
            print('All logs loaded.')
            self.load_models_weights()
            print('All weights loaded.')
        else:
            tf.gfile.MkDir(directory)
            tf.gfile.MkDir(self.directory)
            tf.gfile.MkDir(self.samples_dir)
            tf.gfile.MkDir('{self.samples_dir}/next/'.format(self=self))
            tf.gfile.MkDir('{self.directory}/models_diagrams/'.format(self=self))
            print('Starting new logs.')

        
        print('Last checkpoint:')
        print('iteration: ', self.iteration)
        print('model_iteration: ', self.model_iteration)
        print('is_fadein', int(self.is_fadein))
        print('d_loss: ', self.d_loss)
        print('g_loss: ', self.g_loss)
        print('d_acc: ', self.d_acc)

    def __check_file(self, filename):
        return os.path.exists('{self.directory}/{filename}'
                              .format(self=self,filename=filename))
    def __check_log_files(self):
        return (self.__check_file('/d_losses.log') and
                self.__check_file('/g_losses.log') and
                self.__check_file('/d_acces.log'))

    # не используется
    def check_model_files(self):
        #for i in range(0, n_blocks):
            #discriminators[i][0].load_weights('E:/Практика/beadpack/History-x64-fadein-8k/discriminators/normal_discriminator-{}.h5'.format(i))
            #discriminators[i][1].load_weights('E:/Практика/beadpack/History-x64-fadein-8k/discriminators/fadein_discriminator-{}.h5'.format(i))         
            #generators[i][0].load_weights('E:/Практика/beadpack/History-x64-fadein-8k/generators/normal_generator-{}.h5'.format(i))
            #generators[i][1].load_weights('E:/Практика/beadpack/History-x64-fadein-8k/generators/fadein_generator-{}.h5'.format(i))
            models_count = len(discriminators)
            return __check_file('/discriminators/normal_discriminator-{}.h5'.format(models_count))

    def load_models_weights(self):
        #models_dir = '{self.directory}/models-{self.model_iteration}/'.format(self=self)
        models_dir = '{self.directory}/models-weight/'.format(self=self)
        for i in range(0, self.n_blocks):
            self.discriminators[i][0].load_weights('{}/discriminators/normal_discriminator-{}.h5'.format(models_dir, i))
            self.discriminators[i][1].load_weights('{}/discriminators/fadein_discriminator-{}.h5'.format(models_dir, i))
            self.generators[i][0].load_weights('{}/generators/normal_generator-{}.h5'.format(models_dir, i))
            self.generators[i][1].load_weights('{}/generators/fadein_generator-{}.h5'.format(models_dir, i))
            self.gans[i][0].load_weights('{}/gans/normal_gan-{}.h5'.format(models_dir, i))
            self.gans[i][1].load_weights('{}/gans/fadein_gan-{}.h5'.format(models_dir, i))

    def load_models(self):
        # NEED TO TEST:
        models_dir = '{self.directory}/models/'.format(self=self)
        for i in range(0, self.n_blocks):
            self.discriminators.extend(load('{}/discriminators/normal_discriminator-{}.h5'.format(models_dir, i)),
                                       load('{}/discriminators/fadein_discriminator-{}.h5'.format(models_dir, i)))
            self.generators.extend(load('{}/generators/normal_generator-{}.h5'.format(models_dir, i)),
                                   load('{}/generators/fadein_generator-{}.h5'.format(models_dir, i)))
            self.gans.extend(load('{}/gans/normal_gan-{}.h5'.format(models_dir, i)),
                            load('{}/gans/fadein_gan-{}.h5'.format(models_dir, i)))
        ###########

    def load_models_weights_from_dir(self, weights_dir):
        for i in range(0, self.n_blocks):
            self.discriminators[i][0].load_weights('{}/discriminators/normal_discriminator-{}.h5'.format(weights_dir, i))
            self.discriminators[i][1].load_weights('{}/discriminators/fadein_discriminator-{}.h5'.format(weights_dir, i))
            self.generators[i][0].load_weights('{}/generators/normal_generator-{}.h5'.format(weights_dir, i))
            self.generators[i][1].load_weights('{}/generators/fadein_generator-{}.h5'.format(weights_dir, i))
            self.gans[i][0].load_weights('{}/gans/normal_gan-{}.h5'.format(weights_dir, i))
            self.gans[i][1].load_weights('{}/gans/fadein_gan-{}.h5'.format(weights_dir, i))

    def build_models(self, start_shape:tuple, z_dim:int, n_filters, filter_sizes):
        for model in (base_models.Discriminator, base_models.Generator):

        base_discriminator = base_models.Discriminator(start_shape)
        #self.discriminators.append(base_discriminator)   
        self.models.update({base_models.Discriminator : {start_shape : base_discriminator}})

        base_generator = base_models.Generator(z_dim)
        #self.generators.append(base_generator)
        self.models.update({base_models.Generator : {start_shape : base_generator}})

        base_gan = base_models.GAN(base_generator, base_discriminator)

        models = [base_models.Discriminator, base_models.Generator, base_models.GAN]

        dis_last = base_discriminator
        gen_last = base_generator
        gan_last = base_gan
        last_shape = start_shape

        for i in range(1, self.n_blocks):
	        # Add upsample block/
            new_discriminators = pggan.add_discriminator_block(dis_last,
                                                               n_filters = n_filters[i]//4,#//8,
                                                               filter_size = filter_sizes[i])
            self.models.update({base_models.Discriminator : {last_shape : new_discriminators}})

            new_generators = pggan.add_generator_block(old_generator,
                                                       n_filters = n_filters[i],
                                                       filter_size = filter_sizes[i])
            self.models.update({base_models.Generator : {last_shape : new_generators}})
            #/ Add upsample block

            #dis_last = self.discriminators[-1]
            #gen_last = self.generators[-1]
            dis_last = self.models[base_models.Discriminator][last_shape]
            gen_last = self.models[base_models.Generator][last_shape]
            last_shape = upscale(last_shape)

        self.gans = pggan.build_composite(self.discriminators, self.generators) #?

    def upscale(self, shape:tuple, dims=2):
        new_shape = list(shape)
        for i in range(dims):
            new_shape[i] = new_shape[i]*2

        return tuple(new_shape)

    #def get_generator(self):
        #return self.generators[current]
        
    def save_metrics(self):#, d_loss, g_loss, d_acc):
        # NEED TO CLEAR CODE:
        self.__update_metric(self.d_loss, self.d_losses)
        self.__update_metric(self.g_loss, self.g_losses)
        self.__update_metric(self.d_acc, self.d_acces)

        self.__to_file(self.d_losses, '/d_losses.log')
        self.__to_file(self.g_losses, '/g_losses.log')
        self.__to_file(self.d_acces, '/d_acces.log')
        self.__to_file(self.parameters, '/parameters')
        ######################

    def load_from_file(self, filename):
        logs = []
        with open('{self.directory}/{filename}'.format(self=self, filename=filename), 'rb') as file:
            logs = pickle.load(file)
            
        return logs

    def __update_metric(self, metric, logs):
        logs.append([metric, self.iteration, self.model_iteration, int(self.is_fadein)])

    def __to_file(self, logs, filename):
        with open('{self.directory}/{filename}'.format(self=self, filename=filename), 'wb') as file:
            pickle.dump(logs, file)
    
    def save_models_weights(self):
        #tf.gfile.MkDir('{self.directory}/models-{self.model_iteration}'.format(self=self))
        #models_dir = '{self.directory}/models-{self.model_iteration}/'.format(self=self)
        tf.gfile.MkDir('{self.directory}/models-weight/'.format(self=self))
        models_dir = '{self.directory}/models-weight/'.format(self=self)
        i = 0
        for i in range(0, len(self.generators)):
            tf.gfile.MkDir('{models_dir}/generators'.format(models_dir=models_dir))
            self.generators[i][0].save_weights('{models_dir}/generators/normal_generator-{i}.h5'.format(models_dir=models_dir, i=i))
            self.generators[i][1].save_weights('{models_dir}/generators/fadein_generator-{i}.h5'.format(models_dir=models_dir, i=i))
      
        for i in range(0, len(self.discriminators)):
            tf.gfile.MkDir('{models_dir}/discriminators'.format(models_dir=models_dir))
            self.discriminators[i][0].save_weights('{models_dir}/discriminators/normal_discriminator-{i}.h5'.format(models_dir=models_dir, i=i))
            self.discriminators[i][1].save_weights('{models_dir}/discriminators/fadein_discriminator-{i}.h5'.format(models_dir=models_dir, i=i))
      
        for i in range(0, len(self.gans)):
            tf.gfile.MkDir('{models_dir}/gans'.format(models_dir=models_dir))
            self.gans[i][0].save_weights('{models_dir}/gans/normal_gan-{i}.h5'.format(models_dir=models_dir, i=i))
            self.gans[i][1].save_weights('{models_dir}/gans/fadein_gan-{i}.h5'.format(models_dir=models_dir, i=i))

    def save_models(self):
        tf.gfile.MkDir('{self.directory}/models/'.format(self=self))
        models_dir = '{self.directory}/models/'.format(self=self)
        i = 0
        for i in range(0, len(self.generators)):
            tf.gfile.MkDir('{models_dir}/generators'.format(models_dir=models_dir))
            self.generators[i][0].save('{models_dir}/generators/normal_generator-{i}.h5'.format(models_dir=models_dir, i=i))
            self.generators[i][1].save('{models_dir}/generators/fadein_generator-{i}.h5'.format(models_dir=models_dir, i=i))
      
        for i in range(0, len(self.discriminators)):
            tf.gfile.MkDir('{models_dir}/discriminators'.format(models_dir=models_dir))
            self.discriminators[i][0].save('{models_dir}/discriminators/normal_discriminator-{i}.h5'.format(models_dir=models_dir, i=i))
            self.discriminators[i][1].save('{models_dir}/discriminators/fadein_discriminator-{i}.h5'.format(models_dir=models_dir, i=i))
      
        for i in range(0, len(self.gans)):
            tf.gfile.MkDir('{models_dir}/gans'.format(models_dir=models_dir))
            self.gans[i][0].save('{models_dir}/gans/normal_gan-{i}.h5'.format(models_dir=models_dir, i=i))
            self.gans[i][1].save('{models_dir}/gans/fadein_gan-{i}.h5'.format(models_dir=models_dir, i=i))
      
    def generate_imgs(self, resolution, iteration, generator, n=4, fadein=False):
        z = np.random.normal(0, 1, (n, self.z_dim))

        imgs_mean = np.random.random((n, 1))*2 - 1
        gen_imgs = generator.predict([z, imgs_mean])

        gen_imgs = (gen_imgs+1)*127.5
        gen_imgs = gen_imgs.astype('uint8')
        
        fn = 'norm'
        if fadein:
            fn = 'fade'
            #prob = gen_imgs[0,:,:,0]
        for i in range(0, n):
            img = Image.fromarray(gen_imgs[i,:,:,0])
            file_name = '{self.samples_dir}/x{resolution}-{fn}-i{iteration}-n{i}'.format(self=self,
                                                                                               resolution=resolution,
                                                                                               fn=fn,
                                                                                               iteration=iteration,
                                                                                               i=i)
            e = 0
            while os.path.exists('{file_name}-{e}.png'.format(file_name=file_name, e=e)):
                e += 1
            
            img.save('{file_name}-{e}.png'.format(file_name=file_name, e=e))
    
    # не используется
    def sample_next(self, resolution, iteration, description=''):   
        tf.gfile.MkDir('{self.samples_dir}/next/x32-norm'.format(self=self))
        self.gen_two(self.generators[1][0], '/next/x32-norm-i{}-m{}-{}'.format(iteration, self.model_iteration, description))
        tf.gfile.MkDir('{self.samples_dir}/next/x32-fade'.format(self=self))
        self.gen_two(self.generators[1][1], '/next/x32-fade-i{}-m{}-{}'.format(iteration, self.model_iteration, description))
        tf.gfile.MkDir('{self.samples_dir}/next/x64-norm'.format(self=self))
        self.gen_two(self.generators[2][0], '/next/x64-norm-i{}-m{}-{}'.format(iteration, self.model_iteration, description))
        tf.gfile.MkDir('{self.samples_dir}/next/x32-fade'.format(self=self))
        self.gen_two(self.generators[2][1], '/next/x64-fade-i{}-m{}-{}'.format(iteration, self.model_iteration, description))
        #self.gen_two(self.generators[3][0], '/next/x128-norm{}'.format(iteration))
        #self.gen_two(self.generators[3][1], '/next/x128-fade{}'.format(iteration))
    # не используется
    def gen_two(self, generator, filename):
        imgs_mean = np.array([[0.15]])
        gen_img = generator.predict([self.z_global, imgs_mean])
        fig=plt.figure()
        plt.imshow(gen_img[0,:,:,0], cmap='gray')
        fig.savefig(self.samples_dir + filename)
        plt.close(fig)

        imgs_mean = np.array([[0.65]])
        gen_img = generator.predict([self.z_global, imgs_mean])
        fig=plt.figure()
        plt.imshow(gen_img[0,:,:,0], cmap='gray')
        fig.savefig(self.samples_dir + filename+' 2')
        plt.close(fig)

    def train(self, n_straight, n_fadein, batch_size:int, sample_interval:int, last_model=99999999):
  
      while (self.model_iteration < len(self.discriminators)*2-1) or (self.model_iteration <= last_model):        # check end of loop
          i = self.model_iteration
          if (i % 2 == 0):    # if model is straight
              self.is_fadein = False
              iterations = n_straight[i//2]
              n_resolution = i//2
          else:
              self.is_fadein = True
              iterations = n_fadein[(i+1)//2]
              n_resolution = (i+1)//2

          self.train_block(iterations, batch_size, sample_interval, n_resolution)
          self.model_iteration += 1
          self.iteration = 0 
          
    def train_block(self, iterations:int, batch_size:int, sample_interval:int, n_resolution:int):
        # Get models for current resolution layer:
        int_fadein = int(self.is_fadein)
        is_straight = not self.is_fadein
        int_straight = int(is_straight)

        d_model = self.discriminators[n_resolution][int_fadein]
        g_model = self.generators[n_resolution][int_fadein]
        gan_model = self.gans[n_resolution][int_fadein]   

        d_model.summary()
        plot_model(d_model, to_file='{self.directory}/models_diagrams/discriminator-{self.model_iteration}.png'.format(self=self))
        g_model.summary()
        plot_model(g_model, to_file='{self.directory}/models_diagrams/generator-{self.model_iteration}.png'.format(self=self))
        #self.iteration = 0     
        # Labels for real/fake imgs
        real = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        print('Training-{}-{}-model/'.format(self.model_iteration, int_fadein))

        resolution = self.start_shape[0]*2**(n_resolution)
        print(resolution, resolution)

        self.sample_next(resolution, self.iteration, 'start')  

        while self.iteration < iterations:

            start_time = time.time()
            if self.is_fadein:
                alpha = pggan.update_fadein([g_model, d_model, gan_model], self.iteration, iterations)
                self.parameters.update({'alpha':alpha})
                #pggan.update_fadein([g_model, d_model, gan_model], 1, 2)    
            # -------------------------
            #  Train the Discriminator
            # -------------------------
        
            #resolution = self.d_model.inputs[0].shape[2].value
            # ДЛЯ СТАРЫХ ВЕРСИЙ ЮЗАТЬ ЭТО:
            #resolution = d_model.inputs[0].shape[1][1]
            
            
            downscale = 128 // resolution
            # Get a random batch of real images
            imgs = self.data_loader.get_batch(batch_size, self.end_shape[:2], downscale)
            imgs_mean = np.mean(imgs, axis=(1,2))
        
            # Generate a batch of fake images
            z = np.random.normal(0, 1, (batch_size, self.z_dim))
            gen_imgs = g_model.predict([z, imgs_mean])

            # Train Discriminator
            d_loss_real = d_model.train_on_batch([imgs, imgs_mean], real)
            d_loss_fake = d_model.train_on_batch([gen_imgs, imgs_mean], fake)
            self.d_loss, self.d_acc = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  Train the Generator
            # ---------------------

            # Generate a batch of fake images
            z = np.random.normal(0, 1, (batch_size, self.z_dim))
            #gen_imgs = generator.predict([z,imgs_mean])

            # Train Generator
            self.g_loss = gan_model.train_on_batch([z, imgs_mean], real)
            
            end_time = time.time()
            iteration_time = end_time - start_time
        
            self.iteration += 1

            if (self.iteration) % sample_interval == 0:
                # Save losses and accuracies so they can be plotted after training
                self.save_metrics()
                self.save_models()
                self.generate_imgs(resolution, self.iteration, g_model, 1, self.is_fadein)
                self.sample_next(resolution, self.iteration)       # В ОТДЕЛЬНЫЙ ПОТОК

                # Output training progress
                print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f] [Time: %f.4]" %
                      (self.iteration, self.d_loss, 100.0 * self.d_acc, self.g_loss, iteration_time))

                # Output a sample of generated image
                #sample_images(generator)
                # Get alpha for debug:
                self.__get_alpha(d_model)


        print('/End of training-{}-{}-model'.format(self.model_iteration, int_fadein))

    #for debug:
    def __get_alpha(self, model):
        for layer in model.layers:
            if isinstance(layer, pggan.WeightedSum):
                print(backend.get_value(layer.alpha))