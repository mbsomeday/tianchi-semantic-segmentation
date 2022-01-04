from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam
import pandas as pd
import os
import cv2
import argparse

from utils.uitls import *
from builders import *
from models import *
from base_models import *


parser = argparse.ArgumentParser()

parser.add_argument('--semantic_train', type=str, default=r'C:\Users\wang\Desktop\bbb\code\dataset\semantic\train')
parser.add_argument('--semantic_csv', type=str, default=r'C:\Users\wang\Desktop\bbb\code\dataset\semantic\train_mask.csv')

args = parser.parse_args()

# model = FCN2(input_size=224)



def train(model, dataset,
          epochs=10, batch_size = 1000,
          learning_rate=1e-4, beta_1=0.9, beta_2=0.999):
    adam_optimizer = Adam(lr=learning_rate, beta_1=beta_1, beta_2=beta_2)
    model.model.compile(loss='binary_crossentropy', optimizer=adam_optimizer)
    
    
    
    
    


if __name__ == '__main__':
    # train_mask = pd.read_csv(args.semantic_csv, sep='\t', names=['name', 'mask'])
    # img, mask = load_imageAndMask(train_mask, args.semantic_train, 0)
    # img_path = os.path.join(args.semantic_train, train_mask['name'][0])
    # mask = rle_decode(train_mask['mask'][0]).astype('float32')
    # img = cv2.imread(img_path)
    # cv2.imshow('img', img)
    # cv2.imshow('mask', mask)
    # cv2.waitKey(0)

    net = builder(num_classes=10, input_size=(400, 400), model='FCN-32s', base_model='VGG16')
    net.summary()
    






























