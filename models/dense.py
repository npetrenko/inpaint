import tensorflow as tf
from tensorflow import layers as tfl
from .model import Model, AbstractDecoderModel, AbstractDiscriminatorModel

class DecoderModel(AbstractDecoderModel):
    def __call__(self, inp):
        with tf.variable_scope("decoder", reuse=tf.AUTO_REUSE) as model_scope:
            x = tfl.Dense(64)(inp)
            x = self.batchnorm()(x)
            x = tf.nn.leaky_relu(x)
            
            x = tfl.Dense(256)(x)
            x = self.batchnorm()(x)
            x = tf.nn.leaky_relu(x)
            
            x = tfl.Dense(256)(x)
            x = self.batchnorm()(x)
            x = tf.nn.leaky_relu(x)
            
            x = tfl.Dense(28*28, activation=tf.nn.tanh)(x)
            x = tf.reshape(x, [-1, 28, 28, 1])
        
        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=model_scope.name)
        self.post_call(model_scope)
        
        return x

class DiscriminatorModel(AbstractDiscriminatorModel):
    def __call__(self, inp):
        with tf.variable_scope("discriminator", reuse=tf.AUTO_REUSE) as model_scope:
            x = tfl.Flatten()(inp)
            
            x = tfl.Dense(256)(x)
            x = self.batchnorm()(x)
            x = tf.nn.leaky_relu(x)
            x = self.dropout(0.25)(x)
            
            x = tfl.Dense(256)(x)
            x = self.batchnorm()(x)
            x = tf.nn.leaky_relu(x)
            x = self.dropout(0.25)(x)
            
            x = tfl.Dense(64)(x)
            x = self.batchnorm()(x)
            x = tf.nn.leaky_relu(x)
            x = self.dropout(0.25)(x)
            
            x = tfl.Dense(1, activation=None)(x)
            
        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=model_scope.name)
        self.post_call(model_scope)
        
        return x
