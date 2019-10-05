from keras import Model, backend
from modules.models import pggan

def update_fadein(models, step, n_steps):
    # calculate current alpha (linear from 0 to 1)
    alpha = step / float(n_steps - 1)
    # update the alpha for each model
    for model in models:
        for layer in model.layers:
            if isinstance(base_layer, pggan.WeightedSum):
                backend.set_value(base_layer.alpha, alpha)

def train(g_models, d_models, gan_models, n_norm, n_fadein, batch_size, history, start=0):
  
  if start == 0:
      g_normal, d_normal, gan_normal = g_models[0][0], d_models[0][0], gan_models[0][0]
      print('Training first')
      train_block(g_normal, d_normal, gan_normal, n_norm[0], batch_size, history)
      history.save_models()
      print('/G_normal 0')
      start += 1
  
  #sample_images(g_normal)
  
  for i in range(start, len(g_models)):
    [g_normal, g_fadein] = g_models[i]
    [d_normal, d_fadein] = d_models[i]
    [gan_normal, gan_fadein] = gan_models[i]
    
    print('Training fadein %d' % (i))
    train_block(g_fadein, d_fadein, gan_fadein, n_fadein[i], batch_size, history, True)
    
    print('/G_fadein' + str(i))
    #sample_images(g_fadein)
    history.save_models()

    print('Training straight-through %d' % (i))
    train_block(g_normal, d_normal, gan_normal, n_norm[i], batch_size, history)
    
    print('/G_straight' + str(i))
    #sample_images(g_normal)
    
    history.save_models()

    #return history