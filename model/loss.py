import tensorflow as tf

def sigmoid_cross_entropy_loss(logits, labels, lambda = 0.1, model, img_size):
  '''
  Function to compute loss according to the formula L = L_f + lambda * sum(L_i)
  For reference: https://arxiv.org/pdf/1708.04943.pdf
  ----------
  logits - logits (images before sigmoid/softmax convertation)
  labels - mask elements
  lambda - empirical coefficient, non-negative parameter that leverages the trade-off between two losses (L_f and L_i).
  model - object of NN class
  img_size - tuple or list with (height, with) of images for training
  ----------
  '''
  # compute loss L_f
  loss = tf.nn.sigmoid_cross_entropy_with_logits(labels = labels, logits = logits)
  
  L_i = 0 # to sum up losses L_i for all CCPB blocks
  
  # compute losses L_i
  for i in range(4):
    mod_seq = tf.keras.models.Sequential([tf.keras.models.Model(inputs = model.layers[0].inputs, 
                                                                outputs = model.layers[0].get_layer('block' + str(i+1) + _pool').output),
                                          model.layers[i+1] ])
    
    L_i += tf.nn.sigmoid_cross_entropy_with_logits(labels = labels,
                                                   logits = tf.image.resize(mod_seq.output, 
                                                                            size = (img_size[0], img_size[1]), 
                                                                            method = 'bilinear'))
    
  loss += lambda * L_i 
                                                                
  # averaging across the batch
  return tf.reduce_mean(loss)
