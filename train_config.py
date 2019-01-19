import tensorflow as tf

class TrainConfig:
    def __init__(self, global_step_tensor):
        self.global_step_tensor = global_step_tensor
        self.sess = tf.get_default_session()
        
        self.disc_rate = [1., 1.]
        self.disc_init_steps = 10000
        self.is_disc_init = True
        self.batch_size = 64
        
        self.dec_rate = [1., .7]
        self.dec_init_steps = 10000
        self.is_dec_init = True
        
        self.label_noise_rate = 0.2
        
        self.disc_BN = True
        self.dec_BN = True
        self.disc_dropout_during_dec_training = True

        self.latent_dim = 32

        self.loss_type = "scientific"
        if self.loss_type not in {"scientific", "working"}:
            raise RuntimeError("Incorrect loss type!")

        self.model_type = "conv"
        if self.model_type not in {"dense", "conv"}:
            raise RuntimeError("Incorrect model type!")

    def to_string(self):
        train_dic = {"disc_rate": self.disc_rate,
                     "dec_rate": self.dec_rate,
                     "batch_size": self.batch_size,
                     "disc_init_steps": self.disc_init_steps,
                     "dec_init_steps": self.dec_init_steps,
                     "label_noise_rate": self.label_noise_rate,
                     "disc_BN": self.disc_BN,
                     "dec_BN": self.dec_BN,
                     "latent_dim": self.latent_dim,
                     "loss_type": self.loss_type,
                     "model_type": self.model_type,
                     "disc_dropout_during_dec_training": self.disc_dropout_during_dec_training}
        return str(train_dic)

    def update_step_number(self):
        self.gs = self.sess.run(self.global_step_tensor)
        return self.gs

    def get_dec_rate(self):
        if self.get_dec_init():
            return self.dec_rate[0]
        else:
            return self.dec_rate[1]

    def get_disc_rate(self):
        if self.get_disc_init():
            return self.disc_rate[0]
        else:
            return self.disc_rate[1]

    def get_dec_init(self):
        if self.is_dec_init:
            self.is_dec_init = self.update_step_number() < self.dec_init_steps
            return self.is_dec_init
        else:
            return False

    def get_disc_init(self):
        if self.is_disc_init:
            self.is_disc_init = self.update_step_number() < self.disc_init_steps
            return self.is_disc_init
        else:
            return False
