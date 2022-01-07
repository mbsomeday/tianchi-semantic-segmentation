import tensorflow as tf
import numpy as np

callbacks = tf.keras.callbacks
backend = tf.keras.backend


class LearningRateScheduler(callbacks.Callback):
    def __init__(self,
                 schedule,
                 learning_rate=None,
                 warmup=False,
                 steps_per_epoch=None,
                 verbose=1
                 ):
        super(LearningRateScheduler, self).__init__()
        self.schedule = schedule
        self.learning_rate = learning_rate
        self.warmup = warmup
        self.steps_per_epoch = steps_per_epoch
        self.verbose = verbose
        self.warmup_epochs = 2 if warmup else 0
        self.warmup_steps = int(steps_per_epoch) * self.warmup_epochs if warmup else 0
        self.global_batch = 0
        
    def on_train_batch_begin(self, batch, logs=None):
        self.global_batch += 1
        if self.global_batch < self.warmup_epochs:
            lr = self.learning_rate * self.global_batch / self.warmup_epochs
            backend.set_value(self.model.optimizer.lr, lr)
            if self.verbose > 0:
                print('\nBatch %05d: LearningRateScheduler warming up learning rate to %s'
                      % (self.global_batch, lr))
    
    def on_epoch_begin(self, epoch, logs=None):
        lr = float(backend.get_value(self.model.optimizer.lr))
        if epoch >= self.warmup_epochs:
            lr = self.schedule(epoch - self.warmup_epochs)
            
        backend.set_value(self.model.optimizer.lr, lr)
        
        if self.verbose > 0:
            print('\nEpoch %05d: LearningRateScheduler reducing learning rate to %s.'
                  % (epoch + 1, lr))
            
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logs['lr'] = backend.get_value(self.model.optimizer.lr)
        
        
        
        

        
























