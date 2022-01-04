import tensorflow as tf
from models import *

layers = tf.keras.layers

def builder(num_classes, input_size=(256, 256), model='FCN-8', base_model='VGG'):
    models = {
        'FCN-8S': FCN,
        'FCN-16s': FCN,
        'FCN-32s': FCN,
    }
    net = models[model](num_classes, model, base_model)
    inputs = layers.Input(shape=input_size+(3,))
    
    return net(inputs)




























