from keras import Model, backend
from modules.models import pggan, base_models
from modules.models.base_models import Discriminator, Generator, GAN
from modules.preprocess import DataLoader
from modules.ModelHandler import ModelHandler
import time



def train(mh: ModelHandler, n_straight:int, n_fadein:int, batch_size:int, start=0):
  
  if start == 0:
      mh.model_iteration = 0
      mh.is_fadein = Fasle
      #g_straight, d_straight, gan_straight = mh.generators[0][0], mh.discriminators[0][0], mh.gans[0][0]
      print('Training first')
      train_block(mh, n_straight[0], batch_size)
      mh.save_models()
      print('/g_straight 0')
      start += 1
  
  #sample_images(g_straight)
  
  for i in range(start, len(generators)):
    #[g_straight, g_fadein] = mh.generators[i]
    #[d_straight, d_fadein] = mh.discriminators[i]
    #[gan_straight, gan_fadein] = gans[i]
    
    print('Training fadein %d' % (i))
    train_block(g_fadein, d_fadein, gan_fadein, n_fadein[i], batch_size, mh, True)
    
    print('/G_fadein' + str(i))
    #sample_images(g_fadein)
    mh.save_models()

    print('Training straight-through %d' % (i))
    train_block(g_straight, d_straight, gan_straight, n_straight[i], batch_size, mh)
    
    print('/G_straight' + str(i))
    #sample_images(g_straight)
    
    mh.save_models()

    #return mh

losses = []
accuracies = []
iteration_checkpoints = []

def train_block(mh:ModelHandler,
                iterations:int,
                batch_size:int,
                fadein=False):
    #times = []
    #mh.iteration = 0
    
    # Labels for real images: all ones
    real = np.ones((batch_size, 1))

    # Labels for fake images: all zeros
    fake = np.zeros((batch_size, 1))

    for iteration in range(iterations):

        if fadein:
            update_fadein([mh.generator, mh.d_model, gan_model], iteration, iterations)
            mh.gen
        start_time = time.time()
        # -------------------------
        #  Train the Discriminator
        # -------------------------
        
        #resolution = mh.d_model.inputs[0].shape[2].value
        # ДЛЯ СТАРЫХ ВЕРСИЙ ЮЗАТЬ ЭТО:
        resolution = mh.d_model.inputs[0].shape[1][1]
        
        downscale = 128 // resolution
        # Get a random batch of real images
        imgs = data_loader.get_batch(batch_size, 128, downscale)
        imgs_mean = np.mean(imgs, axis=(1,2))
        
        # Generate a batch of fake images
        z = np.random.normal(0, 1, (batch_size, z_dim))
        gen_imgs = mh.generator.predict([z, imgs_mean])

        # Train Discriminator
        d_loss_real = mh.d_model.train_on_batch([imgs, imgs_mean], real)
        d_loss_fake = mh.d_model.train_on_batch([gen_imgs, imgs_mean], fake)
        d_loss, accuracy = 0.5 * np.add(d_loss_real, d_loss_fake)

        # ---------------------
        #  Train the Generator
        # ---------------------

        # Generate a batch of fake images
        z = np.random.normal(0, 1, (batch_size, z_dim))
        #gen_imgs = generator.predict([z,imgs_mean])

        # Train Generator
        g_loss = gan_model.train_on_batch([z, imgs_mean], real)
        
        end_time = time.time()
        iteration_time = end_time - start_time
        
        if (iteration + 1) % sample_interval == 0:
            # Save losses and accuracies so they can be plotted after training
            mh.iterations[-1] = iteration + 1
            mh.update_metrics(d_loss, g_loss, accuracy)
            
            losses.append((d_loss, g_loss))
            accuracies.append(100.0 * accuracy)
            iteration_checkpoints.append(iteration + 1)

            # Output training progress
            print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f] [Time: %f.4]" %
                  (iteration + 1, d_loss, 100.0 * accuracy, g_loss, iteration_time))

            # Output a sample of generated image
            #if (iteration / sample_interval) % 10 == 0:
            #sample_images(generator)
            # Get alpha for debug:
            #get_alpha(mh.d_model)
            #get_alpha(mh.generator)

            mh.generate_imgs(resolution, iteration + 1, mh.generator, 4, fadein)
            mh.sample_next(resolution, iteration + 1)