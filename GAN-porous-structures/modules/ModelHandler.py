from modules.models import pggan, base_models
from modules.preprocess import DataLoader
import pickle
import tensorflow as tf
import numpy as np
from keras import backend

import os.path

class ModelHandler():
    def __init__(self, directory:str, img_shape:tuple, z_dim:int, n_blocks:int, n_filters, filter_sizes, data_loader:DataLoader):     #, discriminators, generators, gans):
        self.directory = directory + '/History'
        # Models initialize:
        self.discriminators = discriminators
        self.generators = generators 
        self.gans = gans
        self.n_blocks = n_blocks
        self.build_models()
        # Logs:
        self.d_losses = []
        self.g_losses = []
        self.d_acces = []
        # For current metrics:
        self.d_loss = []
        self.g_loss = []
        self.d_acc = []
        self.iteration = 0
        self.model_iteration = 0
        self.is_fadein = False
        self.is_logs_loaded = False
        
        if self.__check_log_files():
            self.d_losses = self.load_logs('/d_losses.log')
            self.d_loss = self.d_losses[-1][0]
            self.g_losses = self.load_logs('/g_losses.log')
            self.g_loss = self.g_losses[-1][0]
            self.d_acces = self.load_logs('/d_acces.log')
            self.d_acc = self.d_acces[-1][0]
            self.iteration = self.d_losses[-1][-2]
            self.model_iteration = self.d_losses[-1][-1]
            self.is_logs_loaded = True
            print('All logs loaded.')
            self.load_weights()
            print('All weights loaded.')
        else:
            tf.gfile.MkDir(self.directory)
            print('Starting new logs.')

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

    def load_weights(self):
        for i in range(0, self.n_blocks):
            self.discriminators[i][0].load_weights('{}/discriminators/normal_discriminator-{}.h5'.format(directory, i))
            self.discriminators[i][1].load_weights('{}/discriminators/fadein_discriminator-{}.h5'.format(directory, i))
            self.generators[i][0].load_weights('{}/generators/normal_generator-{}.h5'.format(directory, i))
            self.generators[i][1].load_weights('{}/generators/fadein_generator-{}.h5'.format(directory, i))

    def build_models(self):
        base_discriminator = base_models.Discriminator(start_shape)
        self.discriminators.append([base_discriminator, base_discriminator])   

        base_generator = base_models.Generator(z_dim)
        self.generators.append([base_generator, base_generator])

        for i in range(1, n_blocks):
            #Block for discriminator/
            old_discriminator = self.discriminators[i - 1][0]
	        # create new model for next resolution
            new_discriminators = pggan.add_discriminator_block(old_discriminator,
                                                               n_filters = n_filters[i]//8,
                                                               filter_size = filter_sizes[i])
            self.discriminators.append(new_discriminators)
            #/Block for discriminator

            #Block for generator/
            old_generator = self.generators[i - 1][0]
            new_generators = pggan.add_generator_block(old_generator,
                                                       n_filters = n_filters[i],
                                                       filter_size = filter_sizes[i])
            self.generators.append(new_generators)
            #/Block for generator

        self.gans = pggan.build_composite(self.discriminators, self.generators)

    #def get_generator(self):
        #return self.generators[current]
        
    def save_metrics(self):#, d_loss, g_loss, d_acc):

        self.__update_metric(self.d_loss, self.d_losses)
        self.__update_metric(self.g_loss, self.g_losses)
        self.__update_metric(self.d_acc, self.d_acces)

        self.__save_logs(self.d_losses, '/d_losses.log')
        self.__save_logs(self.g_losses, '/g_losses.log')
        self.__save_logs(self.d_acces, '/d_acces.log')

    def load_logs(self, filename):
        logs = []
        with open('{self.directory}/{filename}'.format(self=self, filename=filename), 'rb') as file:
            logs = pickle.load(file)
            
        return logs

    def __update_metric(self, metric, logs):
        logs.append([metric, self.iteration, self.model_iteration])

    def __save_logs(self, logs, filename):
        with open('{self.directory}/{filename}'.format(self=self, filename=filename), 'wb') as file:
            pickle.dump(logs, file)
    
    def save_models(self):
        i = 0
        for i in range(0, len(self.generators)):
            tf.gfile.MkDir('{self.directory}/generators'.format(self=self))
            self.generators[i][0].save_weights('{self.directory}/generators/normal_generator-{i}.h5'.format(self=self, i=i))
            self.generators[i][1].save_weights('{self.directory}/generators/fadein_generator-{i}.h5'.format(self=self, i=i))
      
        for i in range(0, len(self.discriminators)):
            tf.gfile.MkDir('{self.directory}/discriminators'.format(self=self))
            self.discriminators[i][0].save_weights('{self.directory}/discriminators/normal_discriminator-{i}.h5'.format(self=self, i=i))
            self.discriminators[i][1].save_weights('{self.directory}/discriminators/fadein_discriminator-{i}.h5'.format(self=self, i=i))
      
        for i in range(0, len(self.gans)):
            tf.gfile.MkDir('{self.directory}/gans'.format(self=self))
            self.gans[i][0].save_weights('{self.directory}/gans/normal_gan-{i}.h5'.format(self=self, i=i))
            self.gans[i][1].save_weights('{self.directory}/gans/fadein_gan-{i}.h5'.format(self=self, i=i))
      
    def generate_imgs(self, resolution, iteration, generator, n=4, fadein=False):
        z = np.random.normal(0, 1, (n, z_dim))

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
            file_name = '{self.directory}/x{resolution}-{fn}-i{iteration}-n{i}'.format(self=self,
                                                                                               resolution=resolution,
                                                                                               fn=fn,
                                                                                               iteration=iteration,
                                                                                               i=i)
            e = 0
            while os.path.exists('{file_name}-{e}.png'.format(file_name=file_name, e=e)):
                e += 1
            
            img.save('{file_name}-{e}.png'.format(file_name=file_name, e=e))
    
    # не используется
    def sample_next(self, resolution, iteration, generators):   
        self.gen_two(generators[2][0], '/x64-norm{}'.format(iteration))
        self.gen_two(generators[3][0], '/x128-norm{}'.format(iteration))
        self.gen_two(generators[3][1], '/x128-fade{}'.format(iteration))
    # не используется
    def gen_two(self, generator, filename):
        imgs_mean = np.array([[0]])
        gen_img = generator.predict([z_global, imgs_mean])
        fig=plt.figure()
        plt.imshow(gen_img[0,:,:,0], cmap='gray')
        fig.savefig(self.directory + filename)
        plt.close(fig)

        imgs_mean = np.array([[0.65]])
        gen_img = generator.predict([z_global, imgs_mean])
        fig=plt.figure()
        plt.imshow(gen_img[0,:,:,0], cmap='gray')
        fig.savefig(self.directory + filename+'2')
        plt.close(fig)

    def train(self, n_straight:int, n_fadein:int, batch_size:int):
  
      if self.model_iteration == 0:
          self.is_fadein = Fasle
          self.train_block(n_straight[0], batch_size)
          self.save_models()
          self.model_iteration += 1
      #sample_images(g_straight)
  
      for i in range(self.model_iteration, len(self.discriminators)):
        self.is_fadein = True
        self.train_block(n_fadein[i], batch_size)  
        #print('/G_fadein' + str(i))
        #sample_images(g_fadein)
        self.save_models()

        self.is_fadein = Fasle
        self.train_block(n_straight[i], batch_size)
        #print('/G_straight' + str(i))
        #sample_images(g_straight)
        self.save_models()

    def train_block(self,
                iterations:int,
                batch_size:int):
        # Get models for current resolution layer:
        d_model = self.generators[self.model_iteration][int(self.is_fadein)]
        g_model = self.generators[self.model_iteration][int(self.is_fadein)]
        gan_model = self.generators[self.model_iteration][int(self.is_fadein)]
        #self.iteration = 0     
        # Labels for real/fake imgs
        real = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        print('Training-{}-{}-model/'.format(self.model_iteration, int(self.is_fadein)))

        for iteration in range(iterations):

            start_time = time.time()
            if self.is_fadein:
                update_fadein([g_model, d_model, gan_model], iteration, iterations)    
            # -------------------------
            #  Train the Discriminator
            # -------------------------
        
            #resolution = self.d_model.inputs[0].shape[2].value
            # ДЛЯ СТАРЫХ ВЕРСИЙ ЮЗАТЬ ЭТО:
            resolution = d_model.inputs[0].shape[1][1]
        
            downscale = 128 // resolution
            # Get a random batch of real images
            imgs = data_loader.get_batch(batch_size, 128, downscale)
            imgs_mean = np.mean(imgs, axis=(1,2))
        
            # Generate a batch of fake images
            z = np.random.normal(0, 1, (batch_size, z_dim))
            gen_imgs = self.generator.predict([z, imgs_mean])

            # Train Discriminator
            d_loss_real = d_model.train_on_batch([imgs, imgs_mean], real)
            d_loss_fake = d_model.train_on_batch([gen_imgs, imgs_mean], fake)
            self.d_loss, self.d_acc = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  Train the Generator
            # ---------------------

            # Generate a batch of fake images
            z = np.random.normal(0, 1, (batch_size, z_dim))
            #gen_imgs = generator.predict([z,imgs_mean])

            # Train Generator
            self.g_loss = gan_model.train_on_batch([z, imgs_mean], real)
        
            end_time = time.time()
            iteration_time = end_time - start_time
        
            if (iteration + 1) % sample_interval == 0:
                # Save losses and accuracies so they can be plotted after training
                self.iterations[-1] = iteration + 1
                self.save_metrics()

                # Output training progress
                print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f] [Time: %f.4]" %
                      (self.iteration + 1, self.d_loss, 100.0 * self.accuracy, self.g_loss, iteration_time))

                # Output a sample of generated image
                #if (iteration / sample_interval) % 10 == 0:
                #sample_images(generator)
                # Get alpha for debug:
                self.__get_alpha(d_model)
                #get_alpha(generator)

                self.generate_imgs(resolution, self.iteration + 1, g_model, 4, self.is_fadein)
                self.sample_next(resolution, self.iteration + 1)

        print('/End of training-{}-{}-model'.format(self.model_iteration, int(self.is_fadein)))

    #for debug:
    def __get_alpha(self, model):
        for model in models:
            for layer in model.layers:
                if isinstance(base_layer, pggan.WeightedSum):
                    print(backend.get_value(base_layer.alpha))