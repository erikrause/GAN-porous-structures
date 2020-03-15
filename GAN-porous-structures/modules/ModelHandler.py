from modules.models import base_models, pggan
from tensorflow.keras.models import Model
#from modules.models import pggan3D as pggan
#from modules.models import base_models3D as base_models
from modules.preprocess import DataLoader
import pickle
import numpy as np
from tensorflow.keras import backend
import time
from PIL import Image
import matplotlib.pyplot as plt
from tensorflow.keras.utils import plot_model
#from typing import Dict, Tuple  # попробовать позже (статическая типизация)
import os.path
import _thread as thread
import os

class ModelHandler():
    def __init__(self, directory:str, start_shape:tuple, z_dim:int, n_blocks:int, n_filters, filter_sizes, data_loader:DataLoader, weights_dir=''):     #, discriminators, generators, gans):
        directory = directory + 'output/'
        self.directory = directory + 'History'
        self.samples_dir = directory + 'samples'
        self.dims = len(start_shape) - 1
        # Models initialize:
        self.models = dict()
        self.n_blocks = n_blocks
        self.start_shape = start_shape
        self.current_shape = start_shape
        upscale = 2**(n_blocks-1)
        #self.end_shape = (start_shape[0]*upscale,
        #                  start_shape[1]*upscale,
        #                  start_shape[2])
        self.end_shape = start_shape[:-1]
        self.end_shape = tuple(x*upscale for x in self.end_shape)
        self.end_shape = list(self.end_shape)
        self.end_shape.append(1)
        self.end_shape = tuple(self.end_shape)

        self.z_dim = z_dim
        self.build_models(start_shape, z_dim, n_filters, filter_sizes)
        #
        # Logs:
        self.d_losses_real = []
        self.d_losses_fake = []
        self.g_losses = []
        self.d_losses = []  #new
        #
        # For current metrics:
        self.d_loss_real = 0
        self.d_loss_fake = 0
        self.d_loss = 0     #new
        self.g_loss = []
        self.iteration = 0
        self.model_iteration = 0
        self.resolution_iteration = 0
        self.is_fadein = False
        self.is_logs_loaded = False
        self.parameters = dict()
        #
        # Train params:
        #self.batch_size = 64
        #self.sample_interval = 100
        self.data_loader = data_loader
        self.z_global = np.random.normal(0, 1, (self.get_batch_size_for_sample(), self.z_dim))
        
        #

        if weights_dir != '':
            self.load_models_weights_from_dir(weights_dir)
            self.iteration = input('print start iteration')
            self.model_iteration = input('print start model_iteration')

        elif self.__check_log_files():
            self.d_losses_real = self.load_from_file('/d_losses_real.log')
            self.d_losses_fake = self.load_from_file('/d_losses_fake.log')
            self.d_loss_real = self.d_losses_real[-1][0]
            self.d_loss_fake = self.d_losses_fake[-1][0]
            self.g_losses = self.load_from_file('/g_losses.log')
            self.g_loss = self.g_losses[-1][0]
            self.iteration = self.d_losses_real[-1][-3]
            self.model_iteration = self.d_losses_real[-1][-2]
            self.is_fadein = bool(self.d_losses_real[-1][-1])
            self.parameters = self.load_from_file('/parameters')
            self.z_global = self.load_from_file("/z_global.log")

            self.resolution_iteration = (self.model_iteration + 1*int(self.is_fadein))//2
            self.current_shape = self.upscale(self.start_shape, k = self.resolution_iteration)

            ################
            # Initialise alpha from last checkpoint:
            if self.is_fadein:
                fadein_models = []
                for model in [base_models.Discriminator, base_models.Generator]:
                    fadein_models.append(self.models[model][self.current_shape][self.is_fadein])
                    pggan.update_fadein(fadein_models, 0, 0, alpha = self.parameters['alpha'])
            #################

            print('All logs loaded.')
            self.load_models_weights()
            print('All weights loaded.')
        else:
            os.makedirs(directory, exist_ok=True)
            os.makedirs(self.directory, exist_ok=True)
            os.makedirs(self.samples_dir, exist_ok=True)
            os.makedirs('{self.samples_dir}/next/'.format(self=self), exist_ok=True)
            os.makedirs('{self.directory}/models_diagrams/'.format(self=self), exist_ok=True)
            print('Starting new logs.')

            self.__to_file(self.z_global, "/z_global.log")

        
        print('Last checkpoint:')
        print('iteration: ', self.iteration)
        print('model_iteration: ', self.model_iteration)
        print('is_fadein', int(self.is_fadein))
        print('d_loss_real: ', self.d_loss_real)
        print('d_loss_fake: ', self.d_loss_fake)
        print('g_loss: ', self.g_loss)

    def __check_file(self, filename):
        return os.path.exists('{self.directory}/{filename}'
                              .format(self=self,filename=filename))
    def __check_log_files(self):
        return (self.__check_file('/d_losses_real.log') and
                self.__check_file('/g_losses.log') and
                self.__check_file('/d_losses_fake.log'))    #need to refactoring

    # не используется
    def check_model_files(self):
            models_count = len(discriminators)
            return __check_file('/discriminators/normal_discriminator-{}.h5'.format(models_count))

    def fadein_label(self, is_fadein):
        if int(is_fadein) == 1:
            return 'fadein'
        else:
            return 'straight'

    # Закружает весовые коэф. из файлов в собранные программой модели (модели хранятся в self.models:dict)
    def load_models_weights(self):

        models_dir = '{self.directory}/models-weights/'.format(self=self)
        for model in [base_models.Discriminator, base_models.Generator]:    #, base_models.GAN]:
            current_model:Model = self.models[model][self.current_shape][self.is_fadein]
            current_model.load_weights("{models_dir}/{name}s/{name}-x{res}-{is_fadein}.h5".format(models_dir=models_dir,
                                                                                                  name = model.__name__,
                                                                                                  is_fadein=self.fadein_label(self.is_fadein),
                                                                                                  res=self.current_shape[0]))
    # Загружает модели из файлов (не используается).
    def load_models(self):
        # NEED TO TEST:
        models_dir = '{self.directory}/models/'.format(self=self)
        for i in range(0, self.n_blocks):
            shape = self.upscale(self.start_shape, k = i)
            for model in [base_models.Discriminator, base_models.Generator]:    #, base_models.GAN]:
                resolution_model = self.models[model][shape]
                for n in range(0, len(resolution_model)):
                    resolution_model[n].load('{models_dir}/{name}s/{n}_{name}-x{res}.h5'.format(models_dir=models_dir,
                                                                                                       name = model.__name__,
                                                                                                       n=self.fadein_label(n), 
                                                                                                       res=shape[0]))
    # Собирает и компилирует модели в self.models:dict, сортируя их по resolution и is_fadein.
    def build_models(self, start_shape:tuple, z_dim:int, n_filters:np.array, filter_sizes:np.array):
        # Build base models/
        models = [base_models.Discriminator, base_models.Generator, base_models.WGAN]
        for model in models:
            if model == base_models.WGAN:
                dis_last = self.models[base_models.Discriminator][start_shape][0]
                gen_last = self.models[base_models.Generator][start_shape][0]
                base_model = model(gen_last, dis_last)
            elif model == base_models.Generator:
                base_model = model(z_dim, start_img_shape=start_shape)
            elif model == base_models.Discriminator:
                base_model = model(img_shape = start_shape)

            self.models.update({model : {start_shape : [base_model]}})         
        # /
        last_shape = start_shape

        for i in range(1, self.n_blocks):
	        # Add upsample block/
            new_shape = self.upscale(last_shape)
            for model in models:
                last_model = self.models[model][last_shape][0]
                
                filters = n_filters[i]
                if model  == base_models.WGAN:
                    last_discriminators = self.models[base_models.Discriminator][new_shape]
                    last_generators = self.models[base_models.Generator][new_shape]
                    last_model = [last_discriminators, last_generators]
                #else:
                    #if model == base_models.Discriminator:
                        #filters = n_filters[i] // 4

                new_models = pggan.add_block(last_model, n_filters = filters, filter_size = filter_sizes[i])
                self.models[model].update({new_shape : new_models})
            
            last_shape = new_shape
            
    def plot_models(self, shape):
        d_model.summary()
        plot_model(d_model, 
                   to_file='{self.directory}/models_diagrams/discriminator-{self.model_iteration}.png'.format(self=self), 
                   show_shapes=True)
        g_model.summary()
        plot_model(g_model, 
                   to_file='{self.directory}/models_diagrams/generator-{self.model_iteration}.png'.format(self=self), 
                   show_shapes=True)
        plot_model(wgan.critic_model,
                   to_file='{self.directory}/models_diagrams/critic_model-{self.model_iteration}.png'.format(self=self), 
                   show_shapes=True)
        plot_model(wgan.generator_model, 
                   to_file='{self.directory}/models_diagrams/generator_model-{self.model_iteration}.png'.format(self=self), 
                   show_shapes=True)

    def upscale(self, shape:tuple, k = 1):

        new_shape = list(shape)
        dims = len(shape) - 1
        for n in range(0, k):
            for i in range(dims):
                new_shape[i] = new_shape[i]*2

        return tuple(new_shape)
        
    def save_metrics(self):#, d_loss, g_loss, d_acc):

        # NEED TO REFACTORING:
        self.__update_metric(self.d_loss_real, self.d_losses_real)
        self.__update_metric(self.d_loss_fake, self.d_losses_fake)
        self.__update_metric(self.g_loss, self.g_losses)
        self.__update_metric(self.d_loss, self.d_losses)

        self.__to_file(self.d_losses_real, '/d_losses_real.log')
        self.__to_file(self.d_losses_fake, '/d_losses_fake.log')
        self.__to_file(self.g_losses, '/g_losses.log')
        self.__to_file(self.d_losses, '/d_losses.log')
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

        models_dir = '{self.directory}/models-weights/'.format(self=self)
        os.makedirs(models_dir, exist_ok=True)
        
        for model in [base_models.Discriminator, base_models.Generator]:    #, base_models.GAN]:
            current_model:Model = self.models[model][self.current_shape][self.is_fadein]
            os.makedirs('{models_dir}/{name}s'.format(models_dir = models_dir, name = model.__name__), exist_ok = True)
            current_model.save_weights("{models_dir}/{name}s/{name}-x{res}-{is_fadein}.h5".format(models_dir = models_dir,
                                                                                                  name = model.__name__,
                                                                                                  is_fadein = self.fadein_label(self.is_fadein),
                                                                                                  res = self.current_shape[0]))
    # Не используется
    def save_models(self):  # NEED TO TEST

        os.makedirs('{self.directory}/models/'.format(self=self), exist_ok=True)
        models_dir = '{self.directory}/models/'.format(self=self)
        for i in range(0, self.n_blocks):
            shape = self.upscale(self.start_shape, k = i)
            for model in [base_models.Discriminator, base_models.Generator, base_models.GAN]:
                os.makedirs('{models_dir}/{name}s'.format(models_dir=models_dir, name = model.__name__), exist_ok=True)
                resolution_model = self.models[model][shape]
                for n in range(0, len(resolution_model)):
                    resolution_model[n].save_weights('{models_dir}/{name}s/{n}_{name}-x{res}.h5'.format(models_dir=models_dir,
                                                                                                       name = model.__name__,
                                                                                                       n=self.fadein_label(n), 
                                                                                                       res=shape[0]))
      
    def get_batch_size_for_sample(self):

        if base_models.batch_size == None:
            return 1
        else:
            return base_models.batch_size

    # batch_size for static batch_size (PlaidML костыль)
    def generate_imgs(self, resolution, iteration, generator, axis, n_imgs=4, step=4, fadein=False):
        
        step = resolution//step

        batch_size = self.get_batch_size_for_sample()

        if axis == 3:
            z = np.random.normal(0, 1, (batch_size, self.z_dim))
            imgs_mean = np.random.random((batch_size, 1))*2 - 1
            ax=(1,2,3)
        else:
            z = np.random.normal(0, 1, (batch_size, self.z_dim))
            imgs_mean = np.random.random((batch_size, 1))*2 - 1
            ax=(1,2)
        gen_imgs = generator.predict(z)
        rm = np.mean(gen_imgs, axis=ax)
        gen_imgs = (gen_imgs+1)*127.5
        gen_imgs = gen_imgs.astype('uint8')
        
        fn = 'norm'
        if fadein:
            fn = 'fade'

        for i in range(0, n_imgs):
            if axis == 3:
                img = Image.fromarray(gen_imgs[0,i*step,:,:,0])
            elif axis == 2:
                img = Image.fromarray(gen_imgs[i,:,:,0])
            file_name = '{self.samples_dir}/x{resolution}-{fn}-i{iteration}-n{i}-(zm{imgs_mean:2f}-rm{rm:2f})'.format(self=self,
                                                                                               resolution=resolution,
                                                                                               fn=fn,
                                                                                               iteration=iteration,
                                                                                               i=i,
                                                                                               imgs_mean=imgs_mean[0][0],   # ONLY FOR 3D!
                                                                                               rm=rm[0][0])

            

            e = 0
            while os.path.exists('{file_name}-{e}.png'.format(file_name=file_name, e=e)):
                e += 1
            
            img.save('{file_name}-{e}.png'.format(file_name=file_name, e=e))
    
    # не используется (for debug)
    def sample_next(self, resolution, iteration, extension='.png'):
        for i in range(1, self.n_blocks-1):
            resolution = self.start_shape[1] *2**i
            os.makedirs('{self.samples_dir}/next/x{resolution}-norm'.format(self=self, resolution=resolution), exist_ok=True)
            os.makedirs('{self.samples_dir}/next/x{resolution}-fade'.format(self=self, resolution=resolution), exist_ok=True)
            shape = self.upscale(self.start_shape, k = i)
            for model in [base_models.Generator]:
                resolution_model = self.models[model][shape]
                for n in range(0, len(resolution_model)):
                    self.gen_two(resolution_model[1], '/next/x{}-fade/r{}-i{}-m{}{}'.format(resolution, self.resolution_iteration, iteration, self.model_iteration, extension))
                    self.gen_two(resolution_model[0], '/next/x{}-norm/r{}-i{}-m{}{}'.format(resolution, self.resolution_iteration, iteration, self.model_iteration, extension))
        #self.gen_two(self.generators[3][0], '/next/x128-norm{}'.format(iteration))
        #self.gen_two(self.generators[3][1], '/next/x128-fade{}'.format(iteration))
    # не используется (for debug)
    def gen_two(self, generator:Model, filename:str):
        #imgs_mean = np.array([[0.15]])

        gen_imgs = generator.predict(self.z_global)
        gen_imgs = (gen_imgs+1)*127.5
        gen_imgs = gen_imgs.astype('uint8')

        img = Image.fromarray(gen_imgs[0,4,:,:,0])
        img.save(self.samples_dir + filename)

        '''
        fig=plt.figure()
        plt.imshow(gen_img[0,:,:,0], cmap='gray')
        fig.savefig(self.samples_dir + filename)
        plt.close(fig)

        imgs_mean = np.array([[0.65]])
        gen_img = generator.predict(self.z_global)
        fig=plt.figure()
        plt.imshow(gen_img[0,:,:,0], cmap='gray')
        fig.savefig(self.samples_dir + filename+' 2')
        plt.close(fig)'''

    def train(self, n_straight, n_fadein, batch_size:int, sample_interval:int, last_model=99999999):
    
        #self. model_iteration = 0 #debug
        while (self.model_iteration < self.n_blocks*2-1) or (self.model_iteration <= last_model):        # need to check end of loop
            i = self.model_iteration
            if (i % 2 == 0):    # if model is straight
                self.is_fadein = False
                iterations = n_straight[i//2]
                n_resolution = i//2
            else:
              self.is_fadein = True
              iterations = n_fadein[(i+1)//2]
              n_resolution = (i+1)//2
          
            #self.current_shape = self.upscale(start_shape, k=n_resolution)
            self.resolution_iteration = (self.model_iteration + 1*int(self.is_fadein))//2
            self.current_shape = self.upscale(self.start_shape, k = self.resolution_iteration)
          
            self.train_block(iterations, batch_size, sample_interval, n_resolution)

            #########
            # КОСТЫЛЬ для PLAIDML! TODO: причина бага?
            current_backend = backend.backend()
            if current_backend == 'tensorflow':
                print(current_backend)
            elif current_backend == 'plaidml':
                print(current_backend)              # With plaidml need to reload program
                print("Backend == PlaidML. Закройте окно и запустите программу снова для тренировки следующего слоя!")
                input()
            else:
                print(' WARNING: BACKEND ' + current_backend + ' NOT SUPPORTED')
                
            ##########
            
            ##############################
            # TODO: код не рабочий, удалить (нужен был для jupyter)
            if not self.is_interrupt:
                self.model_iteration += 1   
                self.iteration = 0
            else:
                input_string = self.input_string
                print('Change total iterations for {} model, or print Enter to exit: '.format(self.model_iteration))
                input_string = input()
                if input_string.startswith('-') and input_string[1:].isdigit():
                    di = int(input_string)
                    self.iterations = input_string + di
                else:
                    break
            ##############################
                
    # Thread for interrupt train loop with keyboard: 
    def input_thread(self, is_interrupt):
        input()
        is_interrupt.append(True)
    ##############


    def train_block(self, iterations:int, batch_size:int, sample_interval:int, n_resolution:int):
        # Get models for current resolution layer:
        int_fadein = int(self.is_fadein)
        is_straight = not self.is_fadein
        int_straight = int(is_straight) # реверс

        # Загрузка моделей для текущего resolution из словаря
        models = []
        for model in [base_models.Discriminator, base_models.Generator,base_models.WGAN]:
            models.append(self.models[model][self.current_shape][self.is_fadein])
        d_model:base_models.Discriminator = models[0]
        g_model:base_models.Generator = models[1]
        wgan:base_models.WGAN = models[2]

        d_model.summary()
        plot_model(d_model, 
                   to_file='{self.directory}/models_diagrams/discriminator-{self.model_iteration}.png'.format(self=self), 
                   show_shapes=True)
        g_model.summary()
        plot_model(g_model, 
                   to_file='{self.directory}/models_diagrams/generator-{self.model_iteration}.png'.format(self=self), 
                   show_shapes=True)
        plot_model(wgan.critic_model,
                   to_file='{self.directory}/models_diagrams/critic_model-{self.model_iteration}.png'.format(self=self), 
                   show_shapes=True)
        plot_model(wgan.generator_model, 
                   to_file='{self.directory}/models_diagrams/generator_model-{self.model_iteration}.png'.format(self=self), 
                   show_shapes=True)
        alpha = -1
        # Labels for real/fake imgs
        real = -np.ones((batch_size, 1))
        fake = np.ones((batch_size, 1))
        dummy = np.zeros((batch_size, 1))

        print('Training-{}-{}-model/'.format(self.model_iteration, int_fadein))
        print('total iterations: ', iterations)
        print('current iteration: ', self.iteration)

        #resolution = self.start_shape[0]*2**(n_resolution)
        print(self.current_shape)
        resolution = self.current_shape[0]
        axis = len(self.current_shape) - 1
        self.generate_imgs(resolution, self.iteration, g_model, axis, n_imgs = 1, fadein=self.is_fadein)
        #self.sample_next(self.current_shape[0], self.iteration, 'start')  

        # TODO: Delete interrupt
        self.is_interrupt = []
        self.input_string = ''
        #thread.start_new_thread(self.input_thread, (self.is_interrupt,))
        
        downscale = self.end_shape[0] // resolution
        data_size = 128 * (downscale)//2
        #self.data_loader.get_batch(data_size, self.end_shape[:-1], downscale)

        #start_lr =  base_models.lr / ((self.model_iteration + 1)/2)
        start_lr = base_models.lr
        dis_start_lr = base_models.dis_lr
        #backend.set_value(d_model.optimizer.lr, dis_start_lr)
        #backend.set_value(gan_model.optimizer.lr, start_lr)

        #####
        # Old train history plot (from manning book):
        critic_losses = []
        gen_losses = []
        accuracies = []
        iteration_checkpoints = []
        #####

        # TRAIN  LOOP:
        while self.iteration < iterations and not self.is_interrupt:
            
            #current_lr = backend.get_value(d_model.optimizer.lr)
            #remaining_lr = 0.0000005 / (self.model_iteration + 1) - current_lr
            #remaining_steps = iterations - self.iteration

            #if remaining_steps == 0:
            #    lr = 1.0
            #else:
                #alpha = step / float(n_steps - 1)
            #    dlr = remaining_lr / remaining_steps
            #    lr = current_lr + dlr
            # Another:
            #lr = start_lr / (1 + 0.000005 * self.iteration)
            
            start_time = time.time()
            if self.is_fadein:
                alpha = pggan.update_fadein([g_model, d_model], self.iteration, iterations)
                #alpha = pggan.update_lod(g_model, d_model, self.iteration, iterations)

                # debug
                z = np.random.normal(0, 1, (batch_size, self.z_dim))
                gen_imgs = g_model.predict(z)
                prob = d_model.predict(gen_imgs)
                #pggan.update_fadein([g_model, d_model, gan_model], 1, 2) 
            #print('fadein time: ', prob2)
            # -------------------------
            #  Train the Discriminator
            # -------------------------
        
            #resolution = self.d_model.inputs[0].shape[2].value
            # ДЛЯ СТАРЫХ ВЕРСИЙ ЮЗАТЬ ЭТО:
            #resolution = d_model.inputs[0].shape[1][1]

            ########################
            # Critic train loop
            for i in range(0, 5):
                # Get a random batch of real images
                imgs = self.data_loader.get_batch(batch_size, self.end_shape[:-1], downscale)
                imgs_mean = np.mean(imgs, axis=self.__get_axis(self.current_shape))
            
                # Generate a batch of noizes
                z = np.random.normal(0, 1, (batch_size, self.z_dim))
                #gen_imgs = g_model.predict(z)

                # Train the Discriminator
                #self.d_loss_real = d_model.train_on_batch(imgs, real[:batch_size])
                #self.d_loss_fake = d_model.train_on_batch(gen_imgs, fake[:batch_size])

                # Train the critic
                self.d_loss = wgan.critic_model.train_on_batch([imgs, z],
                                                               [real, fake, dummy])
            # / Critic train loop
            #######################

            #self.d_loss, self.d_acc = 0.5 * np.add(self.d_loss_real, self.d_loss_fake)
            #self.d_loss_real = np.mean(self.d_loss_real)
            #self.d_loss_fake = np.mean(self.d_loss_fake)    
            
            # ---------------------
            #  Train the Generator
            # ---------------------

            z = np.random.normal(0, 1, (batch_size, self.z_dim))

            # Train Generator
            #self.g_loss = gan_model.train_on_batch(z, real)
            self.g_loss = wgan.generator_model.train_on_batch(z, real)
            
            # Learning rate decay calculating
            #if self.is_fadein:
            #    lr = start_lr/1.1
            #    dis_lr = dis_start_lr/1.1
            #else:
            #    decay = (1 - (self.iteration / iterations)) ** 2    
            #    lr = start_lr * decay +  + 0.00025
            #    dis_lr = dis_start_lr * decay +  + 0.00025

            #backend.set_value(d_model.optimizer.lr, dis_lr)
            #backend.set_value(gan_model.optimizer.lr, lr)

            
            end_time = time.time()
            iteration_time = end_time - start_time
        
            self.iteration += 1


            if (self.iteration) % sample_interval == 0:
                #####
                # Old train history plot (from manning book):
                # Save losses and accuracies so they can be plotted after training
                critic_losses.append(self.d_loss)
                gen_losses.append(self.g_loss)
                #accuracies.append(100.0 * self.d_acc)
                iteration_checkpoints.append(self.iteration)
                self.plot_losses(critic_losses, gen_losses, iteration_checkpoints, resolution, self.is_fadein)
                #####

                # Save losses and accuracies so they can be plotted after training
                self.save_metrics()
                self.save_models_weights()
                self.parameters.update({'alpha':alpha, 'is_fadein': self.is_fadein})
                self.generate_imgs(resolution, self.iteration, g_model, axis, 4, fadein=self.is_fadein)
                self.sample_next(resolution, self.iteration)       # В ОТДЕЛЬНЫЙ ПОТОК

                # Output training progress
                print("%d [D loss: 1: %f 2: %f 3: %f 4: %f] [G loss: %f] [Time: %f.4]" %
                      (self.iteration, self.d_loss[0], self.d_loss[1], self.d_loss[2], self.d_loss[3], self.g_loss, iteration_time))
                #print('Discriminator learning rate: ', backend.get_value(d_model.optimizer.lr))
                #print('Generator learning rate: ', backend.get_value(gan_model.optimizer.lr))
                # Output a sample of generated image
                #sample_images(generator)

                # Get alpha for debug:
                #self.__get_alpha(d_model)
                self.__get_alpha(d_model)

                #print('update lr time: ', lr_time)

        print('/End of training-{}-{}-model'.format(self.model_iteration, int_fadein))



    def plot_losses(self, critic_losses, gen_losses, iteration_checkpoints, resolution, is_fadein):

        fn = 'norm'
        if is_fadein:
            fn = 'fade'

        training_plots_dir = self.directory + "/training plots"

        #if not os.path.exists(training_plots_dir):
        os.makedirs(training_plots_dir, exist_ok=True)

        critic_losses = np.array(critic_losses)
        gen_losses = np.array(gen_losses)

        # Plot training losses for Discriminator and Generator
        plt.figure(figsize=(15, 10))
        plt.plot(iteration_checkpoints, critic_losses.T[0], label="Critic real loss")
        plt.plot(iteration_checkpoints, critic_losses.T[1], label="Critic fake loss")
        plt.plot(iteration_checkpoints, gen_losses, label="Generator loss")

        plt.xticks(iteration_checkpoints, rotation=90)

        plt.title("Training Loss")
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid()
        plt.savefig("{}/losses-{}-{}.png".format(training_plots_dir, resolution, fn))
        plt.close()

    def __get_axis(self, shape):
        if len(shape) - 1 == 3:
            axis = (1,2,3)
        elif len(shape) - 1 == 2:
            axis = (1,2)
        return axis

    #for debug:
    def __get_alpha(self, model):
        for layer in model.layers:
            if isinstance(layer, pggan.WeightedSum):
                print("Fadein alpha: ", backend.get_value(layer.alpha))
        #for debug:
    def __get_lod(self, model):
        if hasattr(model, 'cur_lod'): 
            print("Fadein alpha: ", backend.get_value(model.cur_lod))