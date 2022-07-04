import tensorflow as tf

def sigmoid_with_logits(logits, labels):
  '''
  Function to compute cross entropy loss (sigmoid with logits)
  ----------
  logits - logits (images before sigmoid/softmax convertation)
  labels - mask elements
  ----------
  Function returns mean value of the loss across the batch
  '''
  # compute loss 
  loss = tf.nn.sigmoid_cross_entropy_with_logits(labels = labels, logits = logits)                                                              
  # averaging across the batch
  return tf.reduce_mean(loss)
