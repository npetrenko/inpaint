import tensorflow as tf
from tensorflow import layers as tfl
from .model import Model, AbstractDecoderModel
from .conv import DiscriminatorModel
from tensorflow.keras.layers import UpSampling2D

class DecoderModel(AbstractDecoderModel):
    def pad(self, pad_size=1):
        return lambda x: tf.pad(x,
                                tf.constant([[0,0],[pad_size,pad_size],[pad_size,pad_size],[0,0]]),
                                mode="SYMMETRIC")
    def __call__(self, inp):
        with tf.variable_scope("decoder", reuse=tf.AUTO_REUSE) as model_scope:
            x = tfl.Dense(9*9*8)(inp)
            x = self.batchnorm()(x)
            x = tf.nn.leaky_relu(x)

            x = tf.reshape(x, [-1, 9, 9, 8])

            x = tfl.Conv2D(16, 3, padding="valid")(x)
            x = self.batchnorm()(x)
            x = tf.nn.leaky_relu(x)

            x = tfl.Conv2DTranspose(128, 5, 2, padding="same")(x)
            x = self.batchnorm()(x)
            x = tf.nn.leaky_relu(x)
            print(x)

            x = self.pad()(x)
            x = tfl.Conv2D(64, 3, padding="valid")(x)
            x = self.batchnorm()(x)
            x = tf.nn.leaky_relu(x)
            print(x)

            x = tfl.Conv2DTranspose(64, 3, 2, padding="same")(x)
            x = tf.nn.leaky_relu(x)
            print(x)
            
            x = self.pad()(x)
            x = tfl.Conv2D(1, 3, padding="valid")(x)
        
        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=model_scope.name)
        self.post_call(model_scope)
        
        return x
