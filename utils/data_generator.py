from tensorflow.python.keras.preprocessing.image import Iterator
import pandas as pd
import numpy as np
from keras_applications import imagenet_utils


from utils.uitls import *


class DataIterator(Iterator):
    def __init__(self,
                 generator=None,
                 csv_path=None,
                 data_path=None,
                 num_classes=None,
                 batch_size=None,
                 target_size=None,
                 shuffle=True,
                 seed=None
                 ):
        self.generator = generator
        self.csv_path = csv_path
        self.data_path = data_path
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.target_size = target_size
        self.shuffle = shuffle
        self.seed = seed
        self.train_mask, self.num_samples = self._getinfo()
        super(DataIterator, self).__init__(self.num_samples, self.batch_size, self.shuffle, self.seed)
    
    def _getinfo(self):
        train_mask = pd.read_csv(self.csv_path, sep='\t', names=['name', 'mask'])
        num_samples = len(train_mask)
        return train_mask, num_samples
    
    def _get_batches_of_transformed_samples(self, index_array):
        X = np.zeros(shape=(len(index_array), ) + self.target_size + (3, ))
        Y = np.zeros(shape=(len(index_array),) + self.target_size + (self.num_classes, ))
        
        for i, imgIdx in enumerate(index_array):
            image, label = load_image_label(self.train_mask, self.data_path, imgIdx)
            if self.generator.random_crop:
                image, label = crop_image(image=image, label=label, cropped_size=self.target_size)
            else:
                image, label = resize_image(image=image, label=label, target_size=self.target_size)
                
            #     TODO 这里可以做一些flip、brightness等操作
            
            label = one_hot(label=label, num_classes=self.num_classes)
            image = imagenet_utils.preprocess_input(image.astype("float32"), data_format="channels_last",
                                                    mode="torch")
            X[i] = image
            Y[i] = label
            
        return X, Y
    
    
class ImageDataGenerator():
    def __init__(self,
                 random_crop=False,
                 rotation_rang=0,
                 brightness_range=None,
                 zoom_range=0.0,
                 channel_shift_range=0.0,
                 horizontal_flip=False,
                 vertical_flip=False
                 ):
        self.random_crop = random_crop
        self.rotation_rang = rotation_rang
        self.brightness_range = brightness_range
        self.zoom_range = zoom_range
        self.channel_shift_range = channel_shift_range
        self.horizontal_flip = horizontal_flip
        self.vertical_flip = vertical_flip
    
    def flow(self,
             csv_path=None,
             data_path=None,
             num_classes=None,
             batch_size=None,
             target_size=None,
             shuffle=True,
             seed=None
             ):
        '''
        
        :param csv_path:
        :param data_path:
        :param num_classes:
        :param batch_size:
        :param target_size: (h, w)
        :param shuffle:
        :param seed:
        :return:
        '''
        return DataIterator(generator=self,
                            csv_path=csv_path,
                            data_path=data_path,
                            num_classes=num_classes,
                            batch_size=batch_size,
                            target_size=target_size,
                            shuffle=shuffle,
                            seed=seed
                            )























