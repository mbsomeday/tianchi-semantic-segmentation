from models import Network
import tensorflow as tf

layers = tf.keras.layers
models = tf.keras.models
backend = tf.keras.backend

class FCN(Network):
    def __init__(self, num_classes, version='FCN-32s', base_model='VGG16', **kwargs):
        fcn = {
            'FCN-32s': self._fcn_32s,
        }
        base_model = base_model
        self.fcn = fcn[version]
        super(FCN, self).__init__(num_classes, version, base_model, **kwargs)
    
    def __call__(self, inputs, *args, **kwargs):
        return self.fcn(inputs)
    
    def _conv_relu(self, x, filters, kernel_size):
        x = layers.Conv2D(filters, kernel_size, padding='same', kernel_initializer='he_normal')(x)
        x = layers.ReLU()(x)
        return x
        
    
    def _fcn_32s(self, inputs):
        num_classes = self.num_classes
        
        x = self.encoder(inputs)
        x = self._conv_relu(x, 4096, 7)
        x = layers.Dropout(rate=0.5)(x)
        x = self._conv_relu(x, 4096, 1)
        x = layers.Dropout(rate=0.5)(x)
        
        x = layers.Conv2D(num_classes, 1, kernel_initializer='he_normal')(x)
        x = layers.Conv2DTranspose(num_classes, 64, strides=32, padding='same', kernel_initializer='he_normal')(x)
        
        outputs = x
        return models.Model(inputs, outputs, name=self.version)





























