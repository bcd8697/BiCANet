import tensorflow as tf
from tensorflow.keras import Model, layers

class mod_seq(Model):
  '''
  Sequence for the complex loss to calculate the result in the end of every CCPB Block
  ----------
  layer_bb - layer from the backbone architecture before CCPB block
  layer_ccpb - CCPB block which goes after the corresponding backbone layer
  ----------
  '''
  
  def __init__(self, layer_bb, layer_ccpb):
    super(mod_seq, self).__init__()
    
    self.layer_bb = layer_bb
    self.layer_ccpb = layer_ccpb

  def call(self, x):
    return self.layer_ccpb(self.layer_bb(x))

def sigmoid_cross_entropy_loss(logits, labels, images, model, img_size, l = 0.1):
  '''
  Function to compute loss according to the formula L = L_f + lambda * sum(L_i)
  For reference: https://arxiv.org/pdf/1708.04943.pdf
  ----------
  logits - logits (images before sigmoid/softmax convertation)
  labels - mask elements
  model - object of NN class
  img_size - tuple or list with (height, with) of images for training
  l - empirical coefficient, non-negative parameter that leverages the trade-off between two losses (L_f and L_i).
  ----------
  '''
  # compute loss L_f
  loss = tf.nn.sigmoid_cross_entropy_with_logits(labels = labels, logits = logits)
  
  L_i = 0 # to sum up losses L_i for all CCPB blocks
  
  # compute losses L_i
  for i in range(4):
    m = mod_seq(layer_bb = tf.keras.models.Model(inputs = model.layers[0].inputs,
                                                   outputs = model.layers[0].get_layer('block' + str(i+1) + '_pool').output),
                  layer_ccpb = model.layers[i+1])(images)
    
    L_i += tf.nn.sigmoid_cross_entropy_with_logits(labels = labels,
                                                   logits = tf.image.resize(m, 
                                                                            size = (img_size[0], img_size[1]), 
                                                                            method = 'bilinear'))
    
  loss += l * L_i 
                                                                
  # averaging across the batch
  return tf.reduce_mean(loss)
