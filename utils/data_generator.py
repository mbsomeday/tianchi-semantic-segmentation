from tensorflow.python.keras.preprocessing.image import Iterator
import numpy as np
import pandas as pd
import os
from keras_applications import imagenet_utils

from utils.uitls import *


class DataIterator(Iterator):
    def __init__(self,
                image_data_generator,
                csv_path,
                trainingData_path,
                num_classes,
                batch_size,
                target_size,
                shuffle=True,
                seed=None,
                data_aug_rate=0.
                ):
        self.image_data_generator = image_data_generator
        self.csv_path = csv_path
        self.trainingData_path = trainingData_path
        self.num_classes = num_classes
        self.target_size = target_size
        self.data_aug_rate = data_aug_rate
        self.train_mask, num_images = self._getlen_()
        
        super(DataIterator, self).__init__(num_images, batch_size, shuffle, seed)
        
    def _getlen_(self):
        train_mask = pd.read_csv(self.csv_path, sep='\t', names=['name', 'mask'])
        leng = len(train_mask)
        return train_mask, leng
        
    def _get_batches_of_transformed_samples(self, index_array):
        X = np.zeros(shape=(len(index_array), ) + self.target_size + (3, ))
        Y = np.zeros(shape=(len(index_array), ) + self.target_size + (self.num_classes, ))

        for i, imgIdx in enumerate(index_array):
            image, label = load_imageAndMask(self.train_mask, self.trainingData_path, imgIdx)
            
            image, label = resize_image(image, label, self.target_size)
            
            #  todo: data augmentation
            
            image = imagenet_utils.preprocess_input(image.astype("float32"), data_format="channels_last",
                                                    mode="torch")
            label = one_hot(label, self.num_classes)
            
            X[i], Y[i] = image, label
            
        return X, Y
            
        
class ImageDataenerator():
    def __init__(self,
                 csv_path,
                 trainingData_path,
                 random_crop=False,
                 rotation_rang=0,
                 brightnese_range=None,
                 zoom_range=0.0,
                 channel_shift_range=0.0,
                 horizontal_flip=False,
                 vertical_flip=False):
        self.csv_path = csv_path
        self.trainingData_path = trainingData_path
        self.random_crop = random_crop
        self.rotation_rane = rotation_rang
        self.brightness_range = brightnese_range
        self.zoom_range = zoom_range
        self.channel_shift_range = channel_shift_range
        self.horizontal_flip = horizontal_flip
        self.vertical_flip = vertical_flip
        
    def flow(self,
             num_classes,
             batch_size,
             target_size,
             shuffle=True,
             seed=None,
             data_aug_rate=0.
             ):
        return DataIterator(image_data_generator=self,
                            csv_path=self.csv_path,
                            trainingData_path=self.trainingData_path,
                            num_classes=num_classes,
                            batch_size=batch_size,
                            target_size=target_size,
                            shuffle=shuffle,
                            seed=seed,
                            data_aug_rate=data_aug_rate
                            )


if __name__ == '__main__':
    a = [4,5,6,7]
    for i, idx in enumerate(a):
        print('i:', i, ', idx:', idx)































