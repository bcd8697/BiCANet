# -*- coding: utf-8 -*-
"""loss.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1wbz7AyfCmg2JQw6AXlNBggs5zdaZrutS
"""

import tensorflow as tf

def sigmoid_cross_entropy_loss(x, y):
  # compute loss
  loss = tf.nn.sigmoid_cross_entropy_with_logits(labels = y, logits = x)
  # averaging across the batch
  return tf.reduce_mean(loss)