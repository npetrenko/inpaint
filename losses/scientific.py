import tensorflow as tf

def build_gan_losses(decoder_model, discriminator_model, true_image_tensor, train_config):
    num_true = tf.shape(true_image_tensor)[0]

    decoder_output = decoder_model(tf.random_normal([num_true, train_config.latent_dim]))
    num_false = tf.shape(decoder_output)[0]
    
    def build_disc_gain():
        disc_input = tf.concat([true_image_tensor, decoder_output], axis=0)
        disc_logits = discriminator_model(disc_input)
        
        def label_noise(ln):
            return tf.random_uniform([ln, 1], maxval=train_config.label_noise_rate)

        labels = tf.concat([tf.ones([num_true,1]), 
                            tf.zeros([num_false,1])], axis=0)
        
        labels_noisy = labels + tf.concat([-label_noise(num_true), 
                                           label_noise(num_false)], axis=0)

        disc_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels_noisy, logits=disc_logits)
        disc_gain = -tf.reduce_mean(disc_loss)
        
        def accuracy(labels, preds):
            tmp = tf.cast(tf.equal(labels, tf.cast(preds, tf.float32)), tf.float32)
            return tf.reduce_mean(tmp)
        
        accuracy = accuracy(labels, tf.nn.sigmoid(disc_logits)>0.5)
        discriminator_model.add_summary(lambda: tf.summary.scalar("disc_acc", accuracy))
        discriminator_model.add_summary(lambda: tf.summary.scalar("disc_loss", -disc_gain))
        return disc_gain
    
    disc_gain = build_disc_gain()
    
    disc_input = tf.concat([decoder_output, true_image_tensor], axis=0)
    disc_output = discriminator_model(disc_input)[:num_false]
    false_samples_logp = tf.log(tf.nn.sigmoid(disc_output))    
    dec_gain = tf.reduce_mean(false_samples_logp)
    
    decoder_model.add_summary(lambda: tf.summary.scalar("dec_loss", -dec_gain))
    return dec_gain, disc_gain
