import tensorflow as tf
import tensorflow.io as tfio
from tensorflow.keras import Model, layers

import numpy as np

class CCPB(tf.keras.layers.Layer):
  '''
  CCPB: Context Condense Projection Block for BiCANet
  For reference: https://arxiv.org/pdf/2003.09669.pdf
  '''
  def __init__(self):
    super(CCPB, self).__init__(img_size)
    self.conv_in = layers.Conv2D(filters = 12, kernel_size = 1, padding = 'same', activation = 'relu', input_shape = (img_size[0], img_size[1], -1))
    
    self.conv1_branch1 = layers.Conv2D(filters = 4, kernel_size = 1, padding = 'same', activation = 'relu')
    self.conv1_branch2 = layers.Conv2D(filters = 4, kernel_size = 1, padding = 'same', activation = 'relu')
    self.conv1_branch3 = layers.Conv2D(filters = 4, kernel_size = 1, padding = 'same', activation = 'relu')

    self.conv3_branch21 = layers.Conv2D(filters = 4, kernel_size = 3, padding = 'same', activation = 'relu')
    self.conv3_branch31 = layers.Conv2D(filters = 4, kernel_size = 3, padding = 'same', activation = 'relu')
    self.conv3_branch32 = layers.Conv2D(filters = 4, kernel_size = 3, padding = 'same', activation = 'relu')

    self.conv_out = layers.Conv2D(filters = 1, kernel_size = 1, padding = 'same', activation = 'relu')

  
  def call(self, x):
    
    x = self.conv_in(x)

    # D - branch 1
    d1 = self.conv1_branch1(x)
    # D - branch 2
    d2 = self.conv3_branch21(self.conv1_branch2(x))
    # D - branch 3
    d3 = self.conv3_branch32(self.conv3_branch31(self.conv1_branch3(x)))
    # concat D - branches together
    d = tf.concat([d1, d2, d3], axis = 3)

    # residual connection
    x += d

    return self.conv_out(x)

class BCIB(tf.keras.layers.Layer):
  '''
  BCIB : Bi-directional Context Interaction Block for BiCANet
  For reference: https://arxiv.org/pdf/2003.09669.pdf
  '''
  def __init__(self):
    super(BCIB, self).__init__()
    self.conv_stride2 = tf.keras.layers.Conv2D(filters = 1, kernel_size = 1, strides = 2, padding = 'same')
    self.conv_stride4 = tf.keras.layers.Conv2D(filters = 1, kernel_size = 1, strides = 4, padding = 'same')
    self.conv_stride8 = tf.keras.layers.Conv2D(filters = 1, kernel_size = 1, strides = 8, padding = 'same')
    self.up2 = tf.keras.layers.UpSampling2D(size = 2, interpolation = 'bilinear')
    self.up4 = tf.keras.layers.UpSampling2D(size = 4, interpolation = 'bilinear')
    self.up8 = tf.keras.layers.UpSampling2D(size = 8, interpolation = 'bilinear')

  def call(self, path1, path2, path3, path4, is_training = False):
    # branch 1
    b11, b12, b13, b14 = path1, self.up2(path2), self.up4(path3), self.up8(path4)
    # size checking
    if b11.shape != b12.shape or b11.shape != b13.shape or b11.shape != b14.shape:
      b12 = tf.image.resize(b12, size =(b11.shape[1], b11.shape[2]), method = 'bilinear')
      b13 = tf.image.resize(b13, size =(b11.shape[1], b11.shape[2]), method = 'bilinear')
      b14 = tf.image.resize(b14, size =(b11.shape[1], b11.shape[2]), method = 'bilinear')
    b1 = b11 + b12 + b13 + b14
    
    # branch 2
    b21, b22, b23, b24 = self.conv_stride2(path1), path2, self.up2(path3), self.up4(path4)
    # size checking
    if b22.shape != b21.shape or b22.shape != b23.shape or b22.shape != b24.shape:
      b21 = tf.image.resize(b21, size =(b22.shape[1], b22.shape[2]), method = 'bilinear')
      b23 = tf.image.resize(b23, size =(b22.shape[1], b22.shape[2]), method = 'bilinear')
      b24 = tf.image.resize(b24, size =(b22.shape[1], b22.shape[2]), method = 'bilinear')
    b2 = b21 + b22 + b23 + b24

    # branch 3
    b31, b32, b33, b34 = self.conv_stride4(path1), self.conv_stride2(path2), path3, self.up2(path4)
    # size checking
    if b33.shape != b31.shape or b33.shape != b32.shape or b33.shape != b34.shape:
      b31 = tf.image.resize(b31, size =(b33.shape[1], b33.shape[2]), method = 'bilinear')
      b32 = tf.image.resize(b32, size =(b33.shape[1], b33.shape[2]), method = 'bilinear')
      b34 = tf.image.resize(b34, size =(b33.shape[1], b33.shape[2]), method = 'bilinear')
    b3 = b31 + b32 + b33 + b34
    
    # branch 4
    b41, b42, b43, b44 = self.conv_stride8(path1), self.conv_stride4(path2), self.conv_stride2(path3), path4
    # size checking
    if b44.shape != b41.shape or b44.shape != b42.shape or b44.shape != b43.shape:
      b41 = tf.image.resize(b41, size =(b44.shape[1], b44.shape[2]), method = 'bilinear')
      b42 = tf.image.resize(b42, size =(b44.shape[1], b44.shape[2]), method = 'bilinear')
      b43 = tf.image.resize(b43, size =(b44.shape[1], b44.shape[2]), method = 'bilinear')
    b4 = b41 + b42 + b43 + b44

    return b1, b2, b3, b4

class MRF(tf.keras.layers.Layer):
  '''
  MRF: Multi-resolution Fusion Block for BiCANet
  For reference: https://arxiv.org/pdf/2003.09669.pdf
  '''
  def __init__(self):
    super(MRF, self).__init__()
  
  def call(self, path1, path2, path3, path4, is_training = False):
    path1 = path1
    path2 = tf.image.resize(path2, size =(path1.shape[1], path1.shape[2]), method = 'bilinear')
    path3 = tf.image.resize(path3, size =(path1.shape[1], path1.shape[2]), method = 'bilinear')
    path4 = tf.image.resize(path4, size =(path1.shape[1], path1.shape[2]), method = 'bilinear')
    return tf.concat([path1, path2, path3, path4], axis = 3)

class ChannelWiseAttention(tf.keras.layers.Layer):
    '''
    Channel-wise Attention Mechanism implemented as in https://arxiv.org/pdf/1709.01507.pdf
    J. Hu, L. Shen, and G. Sun, вЂњSqueeze-and-excitation networks,вЂќ in CVPR, 2018, pp. 675вЂ“678.
    '''
    def __init__(self, ALPHA = 1, C = 10, D = 4):
        super(ChannelWiseAttention, self).__init__()

        # squeeze
        self.gap = tf.keras.layers.GlobalAveragePooling2D()
        
        # excitation
        self.fc0 = tf.keras.layers.Dense(int(ALPHA * D), use_bias=False, activation=tf.nn.relu)
        
        self.fc1 = tf.keras.layers.Dense(D, use_bias=False, activation=tf.nn.sigmoid)

        # reshape so we can do channel-wise multiplication
        self.rs = tf.keras.layers.Reshape((1, 1, D))

    def call(self, inputs):
        # calculate channel-wise attention vector
        z = self.gap(inputs)
        u = self.fc0(z)
        u = self.fc1(u)
        u = self.rs(u)
        return u * inputs

class MCFB(tf.keras.layers.Layer):
  '''
  MCFB: Multi-scale Context Fusion Block for BiCANet
  For reference: https://arxiv.org/pdf/2003.09669.pdf
  '''
  def __init__(self, img_width, img_height):
    super(MCFB, self).__init__()

    self.conv_3x3 = layers.Conv2D(filters = 1, kernel_size = 3, padding = 'same', activation = 'relu') 
    self.conv_1xK = layers.Conv2D(filters = 1, kernel_size = (1,5), padding = 'same', activation = 'relu') 
    self.conv_Kx1 = layers.Conv2D(filters = 1, kernel_size = (5,1), padding = 'same', activation = 'relu') 
    self.maxpooling = layers.MaxPool2D(pool_size = 1, padding = 'same')
    
    self.M = max(img_width, img_height) # taking max of image's width and height
    self.conv_1xM = layers.Conv2D(filters = 1, kernel_size = (1, self.M), padding = 'same', activation = 'relu') 
    self.conv_Mx1 = layers.Conv2D(filters = 1, kernel_size = (self.M, 1), padding = 'same', activation = 'relu') 

  def call(self, x):
    # local
    local_interactions = self.conv_3x3(x)
    # long-ranged
    local_interactions += self.conv_Kx1(self.conv_1xK(x))
    # global interactions
    global_interactions = tf.nn.sigmoid(self.conv_Mx1(self.conv_1xM(self.maxpooling(x))))
    res = tf.math.multiply(local_interactions, global_interactions)
    res += local_interactions

    return res

class BiCANet_with_backbone(Model):
  '''
  BiCANet model implementation
  For reference: https://arxiv.org/pdf/2003.09669.pdf
  '''
  def __init__(self, img_width, img_height):
    super(BiCANet_with_backbone, self).__init__()

    self.init_backbone(img_width, img_height)
    self.ccpb1 = CCPB((img_height, img_width))
    self.ccpb2 = CCPB((img_height, img_width))
    self.ccpb3 = CCPB((img_height, img_width))
    self.ccpb4 = CCPB((img_height, img_width))

    self.bcib = BCIB()

    self.mrf = MRF()
    self.ch_attention = ChannelWiseAttention()
    self.up2 = tf.keras.layers.UpSampling2D (size = 2, interpolation = 'bilinear')
    self.mcfb = MCFB(img_width = img_width, img_height = img_height)

  def init_backbone(self, img_width, img_height):
    self.backbone = tf.keras.applications.vgg16.VGG16(weights='imagenet', 
                                                      include_top = False,
                                                      input_shape = (img_height, img_width, 3))
    self.backbone.trainable = False

  def _call(self, x):
    final_output = self.backbone(x)
    backbone_out_1 = tf.keras.models.Model(inputs = self.backbone.inputs, 
                                          outputs = self.backbone.get_layer('block1_pool').output)
    backbone_out_2 = tf.keras.models.Model(inputs = self.backbone.inputs, 
                                          outputs = self.backbone.get_layer('block2_pool').output)
    backbone_out_3 = tf.keras.models.Model(inputs = self.backbone.inputs, 
                                          outputs = self.backbone.get_layer('block3_pool').output)
    backbone_out_4 = tf.keras.models.Model(inputs = self.backbone.inputs, 
                                          outputs = self.backbone.get_layer('block4_pool').output)
    
    return backbone_out_1(x), backbone_out_2(x), backbone_out_3(x), backbone_out_4(x)

  def call(self, x, is_training = False):
    x1, x2, x3, x4 = self._call(x)
    
    # adaptive convolution
    x1, x2, x3, x4 = self.ccpb1(x1), self.ccpb2(x2), self.ccpb3(x3), self.ccpb4(x4)

    # BCIB
    x1, x2, x3, x4 = self.bcib(x1, x2, x3, x4)

    # multi-resolution fusion
    x = self.mrf(x1, x2, x3, x4)

    # channel-wise attention block and 2x Upsampling
    x = self.up2(self.ch_attention(x))

    # MCFB
    x = self.mcfb(x)

    # since we have only 2 classes, we may want to use sigmoid function instead of softmax
    # NB: we need to return logits for our loss function while training
    if is_training:
      return x
    
    return tf.math.sigmoid(x)
