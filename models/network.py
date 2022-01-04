from base_models import *


class Network():
    def __init__(self, num_classes, version='PAN', base_model='ResNet50', dilation=None, **kwargs):
        super(Network, self).__init__(**kwargs)
        if base_model in ['VGG16', 'VGG19']:
            self.encoder = VGG(base_model, dilation=dilation)
            
        self.num_classes = num_classes
        self.version = version
        self.base_model = base_model
    
    def __call__(self, inputs, *args, **kwargs):
        return inputs
    
    def get_version(self):
        return self.version
    
    def get_base_model(self):
        return self.base_model
    



























