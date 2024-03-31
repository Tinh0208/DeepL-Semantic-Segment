# -*- coding: utf-8 -*-
"""U-Net.model

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1U1Kg42Lo4cD4dlyTMQnBjh3mBRP5oaS7
"""

from tensorflow import keras
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPool2D, UpSampling2D, concatenate, Conv2DTranspose, BatchNormalization, Dropout
from keras.layers import Activation, Concatenate

def conv_block(input, num_filters, block_name, batch_norm:bool=True, dropout_rate:float=None):
  x = Conv2D(num_filters, 3, padding='same',name=block_name+'_conv1')(input)
  if batch_norm:
    x = BatchNormalization(name=block_name+'_norm1')(x)
  x = Activation('relu', name=block_name+'_act1')(x)
  if dropout_rate:
    x = Dropout(dropout_rate, name=block_name+'_drop1')(x)

  x = Conv2D(num_filters, 3,padding='same',name=block_name+'_conv2')(x)
  if batch_norm:
    x = BatchNormalization(name=block_name+'_norm2')(x)
  x = Activation('relu', name=block_name+'_act2')(x)
  if dropout_rate:
    x = Dropout(dropout_rate, name=block_name+'_drop2')(x)

  return x


def encoder_block(input, num_filters, block_name:str, batch_norm=True, dropout_rate:float = None):
  s = conv_block(input, num_filters, block_name, dropout_rate)
  p = MaxPool2D((2,2),name=block_name+'_pool')(s)
  # if dropout_rate:
  #   p = Dropout(dropout_rate, name=block_name+'_dropout')(p)

  return s, p


def decoder_block(input, skip_features: list, num_filters, block_name:str, batch_norm=True, dropout_rate:float = None):
  d = Conv2DTranspose(num_filters, (3,3), strides=2, padding='same', name=block_name+'_upconv')(input)
  d = Concatenate(name=block_name+'_cat')([d, *skip_features])
  # if dropout_rate:
  #   d = Dropout(dropout_rate, name=block_name+'_dropout')(d)
  d = conv_block(d, num_filters, block_name, dropout_rate)
  
  return d

def Unet(input_shape, output_classes = 1,base_filter=64, batch_norm = True, dropout_rate = None):
  bf = base_filter
  filters = [bf, bf*2, bf*4, bf*8, bf*16]

  inputs= Input(shape = input_shape, name='Main Input')

  # encoder
  s1, p1 = encoder_block(inputs, filters[0], 'En_1', batch_norm, dropout_rate)
  s2, p2 = encoder_block(p1, filters[1], 'En_2', batch_norm, dropout_rate)
  s3, p3 = encoder_block(p2, filters[2], 'En_3', batch_norm, dropout_rate)
  s4, p4 = encoder_block(p3, filters[3], 'En_4', batch_norm, dropout_rate)

  # bridge
  b1 = conv_block(p4, filters[4], 'Bottleneck', dropout_rate)

  # decoder
  d1 = decoder_block(b1, [s4], filters[3], 'De_1', batch_norm, dropout_rate)
  d2 = decoder_block(d1, [s3], filters[2], 'De_2', batch_norm, dropout_rate)
  d3 = decoder_block(d2, [s2], filters[1], 'De_3', batch_norm, dropout_rate)
  d4 = decoder_block(d3, [s1], filters[0], 'De_4', batch_norm, dropout_rate)

  outputs = Conv2D(output_classes, 1, padding='same', activation='softmax',name='Output')(d4)

  model = Model(inputs, outputs, name='U-Net')
  return model

