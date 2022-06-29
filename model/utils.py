from glob import glob

import random

import tensorflow as tf
import tensorflow.io as tfio
from tensorflow.keras import Model, layers

import matplotlib.pyplot as plt
import numpy as np
import skimage
from skimage import io

import numpy as np

import os

def split_data(img_path, mask_path, save_path = '/content/', train_size = 0.75, val_size = 0.15, test_size = 0.1):
  '''
  Function splits images into 3 folders with train-val-test data
  ----------
  img_path - path to images files
  mask_path - path to masks files
  save_path - path where to save the resulting split
  train_size - size of train data in percentage (range 0.0 - 1.0)
  val_size - size of val data in percentage (range 0.0 - 1.0)
  test_size - size of test data in percentage (range 0.0 - 1.0)
  ----------
  Function does not return any value
  '''
  assert 0.0 < train_size < 1.0 , 'Train data percentage should be given in range 0.0 - 1.0!'
  assert 0.0 < val_size < 1.0 , 'Val data percentage should be given in range 0.0 - 1.0!'
  assert 0.0 < test_size < 1.0 , 'Test data percentage should be given in range 0.0 - 1.0!'
  assert train_size + val_size + test_size == 1.0, 'train_size + val_size + test_size should be equal to 1.0!'
  

  # before splitting data into train-val-test
  # creating folders to fill both for images and masks
  for p in ['images', 'masks']:
    os.makedirs(save_path + 'train/' + p + '/')
    os.makedirs(save_path + 'val/' + p + '/')
    os.makedirs(save_path + 'test/' + p + '/')

  # reading names of images
  names = os.listdir(img_path)
  # shuffle images (just in case)
  random.shuffle(names)

  for idx_name, name in enumerate(names):
    if idx_name < int(train_size * len(names)): # filling up the train
      os.replace(img_path + name, save_path + '/train/images/' + name)
      os.replace(mask_path + name[:-3] + 'png', save_path + '/train/masks/' + name[:-3] + 'png')
      
    elif idx_name < int((train_size + val_size) * len(names)): # filling up the validation
      os.replace(img_path + name, save_path + '/val/images/' + name)
      os.replace(mask_path + name[:-3] + 'png', save_path + '/val/masks/' + name[:-3] + 'png')
      
    else: # filling up the test
      os.replace(img_path + name, save_path + '/test/images/' + name)
      os.replace(mask_path + name[:-3] + 'png', save_path + '/test/masks/' + name[:-3] + 'png')
    
  os.rmdir(IMAGES_PATH)
  os.rmdir(MASKS_PATH)

def parse_image(img_path):
  '''
  Function to parse images and to make some convertations and preparations for furter use
  Might be used through mapping it on the whole dataset
  ----------
  img_path - path to the 
  ----------
  Function returns dict with 2 elements: image and segmentation mask
  '''

  image = tf.io.read_file(img_path)
  image = tf.image.decode_jpeg(image, channels=3)
  image = tf.image.convert_image_dtype(image, tf.uint8)

  mask_path = tf.strings.regex_replace(img_path, "images", "masks")
  mask_path = tf.strings.regex_replace(mask_path, "jpg", "png")
  mask = tf.io.read_file(mask_path)
  mask = tf.image.decode_png(mask, channels=1)
  mask = tf.where(mask == 255, np.dtype('uint8').type(1), mask)

  return {'image': image, 'segmentation_mask': mask}

def display_sample(display_list):
    '''
    Function shows side-by-side an input image and the mask
    '''
    plt.figure(figsize=(10, 8))

    title = ['Input Image', 'True Mask']

    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i+1)
        plt.title(title[i])
        plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
        plt.axis('off')
    plt.show()

