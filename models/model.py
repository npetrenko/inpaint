import tensorflow as tf
import tensorflow.layers as tfl

class Model:
    def __init__(self, learning_phase):
        self.summaries = []
        self.learning_phase = learning_phase
        
    def batchnorm(self):
        return lambda x: tfl.batch_normalization(x, training=self.learning_phase)
    
    def post_call(self, model_scope):
        if not hasattr(self, "model_scope"):
            self.model_scope = model_scope
    
    def dropout(self, rate=0.5):
        return lambda x: tfl.dropout(x, rate=rate, training=self.learning_phase)
    
    def add_summary(self, summ):
        self.summaries += [summ]
        
    def get_merged_summaries(self):
        with tf.variable_scope(self.model_scope):
            return tf.summary.merge([x() for x in self.summaries])

    def get_variables(self):
        return self.variables

    def get_update_ops(self):
        return tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=self.model_scope.name)

class AbstractDecoderModel(Model):
    def __init__(self, learning_phase, train_config=None):
        super().__init__(learning_phase)
        self.train_config = train_config
    
    def batchnorm(self):
        if self.train_config != None and self.train_config.dec_BN:
            return super().batchnorm()
        else:
            return lambda x: x

class AbstractDiscriminatorModel(Model):
    def __init__(self, dropout_phase, batchnorm_phase, train_config=None):
        super().__init__(dropout_phase)
        self.batchnorm_phase = batchnorm_phase
        self.train_config = train_config
    
    def batchnorm(self):
        if self.train_config != None and self.train_config.disc_BN:
            return lambda x: tfl.batch_normalization(x, training=self.batchnorm_phase)
        else:
            return lambda x: x
