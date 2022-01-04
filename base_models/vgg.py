import tensorflow as tf
import numpy as np

layers = tf.keras.layers
backend = tf.keras.backend
models = tf.keras.models


class VGG():
    def __init__(self, version='VGG16', dilation=None, **kwargs):
        super(VGG, self).__init__(**kwargs)
        
        params = {'VGG16': [2, 2, 3, 3, 3],
                  'vgg19': [2, 2, 4, 4, 4]}
        self.version = version
        assert version in params
        self.params = params[version]
        
        if dilation is None:
            self.dilation = [1, 1]
        
        assert len(self.dilation) == 2
        
    def __call__(self, inputs, output_stages='c5', *args, **kwargs):
        # inputs = layers.Input(shape=(244, 244, 3))

        dilation = self.dilation
        x = inputs
        
        for i in range(self.params[0]):
            x = layers.Conv2D(64, (3, 3), activation='relu', padding='same',
                              name='block1_conv'+str(i+1))(x)
        x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)
        c1 = x
        
        for i in range(self.params[1]):
            x = layers.Conv2D(128, (3, 3), activation='relu',
                              padding='same',
                              name='block2_conv'+str(i+1))(x)
        x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)
        c2 = x
        
        for i in range(self.params[2]):
            x = layers.Conv2D(256, (3, 3), activation='relu', padding='same',
                              name='block3_conv'+str(i+1))(x)
        x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)
        c3 = x
        
        for i in range(self.params[3]):
            x = layers.Conv2D(512, (3, 3), activation='relu', padding='same',
                              name='block4_conv'+str(i+1),
                              dilation_rate=dilation[0]
                              )(x)
        if dilation[0] == 1:
            x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)
        c4 = x

        for i in range(self.params[4]):
            x = layers.Conv2D(512, (3, 3), activation='relu', padding='same',
                              name='block5_conv' + str(i + 1),
                              dilation_rate=dilation[1]
                              )(x)
        if dilation[1] == 1:
            x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)
        c5 = x

        self.outputs = {'c1': c1,
                        'c2': c2,
                        'c3': c3,
                        'c4': c4,
                        'c5': c5 }
        if type(output_stages) is not list:
            return self.outputs[output_stages]
        else:
            return [self.outputs[ci] for ci in output_stages]
        
        
        
        

if __name__ == '__main__':
    pass
    
























