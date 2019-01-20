import tensorflow as tf
from tensorflow import layers as tfl
from .model import Model, AbstractDecoderModel, AbstractDiscriminatorModel
from tensorflow.keras.layers import UpSampling2D

class DecoderModel(AbstractDecoderModel):
    def pad(self, num_pad=1):
        return lambda x: tf.pad(x,
                                tf.constant([[0,0],[num_pad,num_pad],[num_pad,num_pad],[0,0]]),
                                mode="SYMMETRIC")
    def __call__(self, inp):
        with tf.variable_scope("decoder", reuse=tf.AUTO_REUSE) as model_scope:
            x = tfl.Dense(11*11*16)(inp)
            x = self.batchnorm()(x)
            x = tf.nn.leaky_relu(x)

            x = tf.reshape(x, [-1, 11, 11, 16])

            x = tfl.Conv2D(32, 3, padding="valid")(x)
            x = self.batchnorm()(x)
            x = tf.nn.leaky_relu(x)

            x = UpSampling2D()(x)
            x = tfl.Conv2D(128, 5, padding="valid")(x)
            x = self.batchnorm()(x)
            x = tf.nn.leaky_relu(x)

            x = self.pad()(x)
            x = tfl.Conv2D(64, 3, padding="valid")(x)
            x = self.batchnorm()(x)
            x = tf.nn.leaky_relu(x)

            x = UpSampling2D()(x)

            x = self.pad()(x)
            x = tfl.Conv2D(64, 3, padding="valid")(x)
            x = tf.nn.leaky_relu(x)
            
            x = self.pad()(x)
            x = tfl.Conv2D(1, 3, padding="valid")(x)
        
        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=model_scope.name)
        self.post_call(model_scope)
        
        return x

class DiscriminatorModel(AbstractDiscriminatorModel):
    def __call__(self, inp):
        with tf.variable_scope("discriminator", reuse=tf.AUTO_REUSE) as model_scope:
            x = tfl.Conv2D(32, 3, 2)(inp)
            x = self.batchnorm()(x)
            x = tf.nn.leaky_relu(x)

            x = tfl.Conv2D(128, 3, 2)(x)
            x = self.batchnorm()(x)
            x = tf.nn.leaky_relu(x)
            x = self.dropout(0.25)(x)

            x = tfl.Conv2D(256, 3, 2)(x)
            x = self.batchnorm()(x)
            x = tf.nn.leaky_relu(x)
            x = self.dropout(0.25)(x)
            
            x = tfl.Flatten()(x)
            x = tfl.Dense(1, activation=None)(x)
            
        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=model_scope.name)
        self.post_call(model_scope)
        
        return x
