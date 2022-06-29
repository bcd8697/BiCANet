import tensorflow as tf

def sigmoid_cross_entropy_loss(x, y):
  # compute loss
  loss = tf.nn.sigmoid_cross_entropy_with_logits(labels = y, logits = x)
  # averaging across the batch
  return tf.reduce_mean(loss)
