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
from utils.losses import *
# from utils.data_generator import *
from utils.data_generator import *
from utils.learning_rate import *
from utils.callbacks import *
from utils.helpers import *
from utils.metrics import *


parser = argparse.ArgumentParser()

parser.add_argument('--train_csv', type=str, default=r'C:\Users\wang\Desktop\bbb\code\dataset\semantic\clean\train_mask.csv')
parser.add_argument('--val_csv', type=str, default=r'C:\Users\wang\Desktop\bbb\code\dataset\semantic\clean\val_mask.csv')
parser.add_argument('--data_path', type=str, default=r'C:\Users\wang\Desktop\bbb\code\dataset\semantic\train')

parser.add_argument('--validation_freq', help='How often to perform validation.', type=int, default=1)
parser.add_argument('--valid_batch_size', help='The validation batch size.', type=int, default=10)
parser.add_argument('--checkpoint_freq', help='How often to save a checkpoint.', type=int, default=1)
parser.add_argument('--num_epochs', help='The number of epochs to train for.', type=int, default=10)
parser.add_argument('--lr_scheduler', type=str, default='cosine_decay')
parser.add_argument('--learning_rate', help='The initial learning rate.', type=float, default=3e-4)
parser.add_argument('--lr_warmup', help='Whether to use lr warm up.', type=bool, default=True)
parser.add_argument('--num_classes', type=int, default=2)
parser.add_argument('--batch_size', type=int, default=16)

args = parser.parse_args()

paths = check_related_path(os.getcwd())

train_mask = pd.read_csv(args.train_csv, sep='\t', names=['name', 'mask'])
val_mask = pd.read_csv(args.val_csv, sep='\t', names=['name', 'mask'])
train_samples = len(train_mask)
val_samples = len(val_mask)
steps_per_epoch = train_samples // args.batch_size
validation_steps = val_samples // args.valid_batch_size


net = builder(num_classes=args.num_classes, input_size=(224, 224), model='FCN-32s', base_model='VGG16')

losses = { 'ce': categorical_crossentropy_with_logits}
loss = losses['ce']

optimizers = { 'adam': tf.keras.optimizers.Adam(learning_rate=args.learning_rate)}

net.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate),
            loss=loss,
            metrics=[MeanIoU(args.num_classes)])

# callbacks
lr_decays = {'step_decay': step_decay(args.learning_rate, args.num_epochs, warmup=args.lr_warmup),
             'poly_decay': poly_decay(args.learning_rate, args.num_epochs, warmup=args.lr_warmup),
             'cosine_decay': cosine_decay(args.num_epochs, args.learning_rate, warmup=args.lr_warmup)}
lr_decay = lr_decays[args.lr_scheduler]

learning_rate_scheduler = LearningRateScheduler(schedule=lr_decay,
                                                learning_rate=args.learning_rate,
                                                warmup=args.lr_warmup,
                                                steps_per_epoch=steps_per_epoch,
                                                verbose=1)
base_model = 'VGG16'
# model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
#     filepath=os.path.join(paths['checkpoints_path'],
#                           '{model}_based_on_{base}_'.format(model=args.model, base=base_model)),
#     save_best_only=True, period=args.checkpoint_freq,monitor='val_mean_io_u', mode='max'
# )

net_callbacks = [learning_rate_scheduler]

train_gen = ImageDataGenerator(random_crop=False)
val_gen = ImageDataGenerator(random_crop=False)


train_generator = train_gen.flow(csv_path=args.train_csv,
                                 data_path=args.data_path,
                                 num_classes=args.num_classes,
                                 batch_size=10,
                                 target_size=(224, 224))

val_generator = val_gen.flow(csv_path=args.val_csv,
                                 data_path=args.data_path,
                                 num_classes=args.num_classes,
                                 batch_size=10,
                                 target_size=(224, 224))

net.fit_generator(train_generator,
                  steps_per_epoch=10,
                  epochs=5,
                  callbacks=net_callbacks,
                  validation_data=val_generator,
                  validation_steps=validation_steps,
                  validation_freq=args.validation_freq,
                  verbose=1
                  )
    
    
    































