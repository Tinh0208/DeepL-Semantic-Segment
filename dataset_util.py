# -*- coding: utf-8 -*-
"""dataset_util

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1MgQfgVgXPAe7n28ByyVviFnKKSqrcc9u
"""

import os
from glob import glob
import numpy as np
import skimage.transform as st
import random
from PIL import Image
from tensorflow.keras.utils import load_img, img_to_array, to_categorical

class DataLoader():
  def __init__(self, dataset_path:str):
    self.base_path = dataset_path
    self.data_path = {
        'train': os.path.join(self.base_path,'train'),
        'valid': os.path.join(self.base_path,'valid'),
        'test': os.path.join(self.base_path,'test')
    }
    label_base = os.path.join(self.base_path,'labels')
    self.label_path = {
        'train':os.path.join(label_base,'train'),
        'valid':os.path.join(label_base,'valid'),
        'test':os.path.join(label_base,'test')
    }

  def load(self, purpose:str, image_shape:tuple, shuffle = False, seed = None, anti_aliasing = False):
    image_list = []
    mask_list = []

    img_path = sorted(glob(self.data_path[purpose]+'/*'))
    mask_path = sorted(glob(self.label_path[purpose]+'/*'))

    if shuffle:
      temp = list(zip(img_path,mask_path))
      if seed:
        random.seed(seed)
      random.shuffle(temp)
      img_path, mask_path = zip(*temp)

    for path in img_path:
      img = load_img(path,color_mode='rgb')
      img = img_to_array(img)
      img = st.resize(img, image_shape,order=0, preserve_range=True, anti_aliasing=anti_aliasing)
      image_list.append(img)

    for path in mask_path:
      mask = np.load(path)
      mask = st.resize(mask, image_shape,order=0, preserve_range=True, anti_aliasing=anti_aliasing)
      mask_list.append(mask)

    # print(img_path)
    # print(mask_path)

    return np.array(image_list), np.array(mask_list)

def preprocessing(images, labels, n_classes):
  # Thêm chiều cho nhãn: (224,224,3) => (224,224,3,1)
  labels = np.expand_dims(labels, axis=-1)
  labels = np.asarray(labels, dtype=np.float32)
  # print(labels.shape)

  # Chuẩn hóa ảnh về miền [0,1]
  images = np.asarray(images, dtype=np.float32)
  images /= 255

  # Chuyển nhãn thành onehot
  temp = to_categorical(labels, num_classes = n_classes)
  labels = temp.reshape((labels.shape[0], labels.shape[1],
                                  labels.shape[2], n_classes))
  print('Label data shape is: {}'.format(labels.shape))

  return images, labels

