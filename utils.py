import tensorflow as tf
import numpy as np

class TBImUploader:
    def __init__(self, decoder_model, train_config, num_rows):
        self.num_rows = num_rows
        self.num_images = num_rows**2
        noise = np.random.normal(size=[self.num_images, train_config.latent_dim]).astype(np.float32)
        
        model_output = decoder_model(tf.constant(noise))
        flattened = self.flatten(model_output)
        self.summary = tf.summary.image("img", flattened)
        
    def flatten(self, img_t):
        exp = tf.reshape(img_t, [self.num_rows,self.num_rows*28,28,1])
        exp = [exp[i] for i in range(self.num_rows)]
        exp = tf.concat(exp, axis=1)
        exp = tf.expand_dims(exp, 0)
        return exp
    
    def post_summary(self):
        sm = self.summary.eval({dec_learning_phase: False})
        writer.add_summary(sm, global_step.eval())

def get_new_dir(base_dir, prefix):
    import os
    for ix in range(1000):
        cand = os.path.join(base_dir, prefix + str(ix))
        if not os.path.exists(cand):
            return cand
