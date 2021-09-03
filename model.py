import tensorflow as tf
import tensorflow.keras as keras

class Model(keras.Model):
    def __init__(self,batch_size,generator_input=100):
        super(Model, self).__init__()
        self.discriminator_loss_metric = keras.metrics.Mean(name="d_loss")
        self.generator_loss_metric = keras.metrics.Mean(name="g_loss")
        self.generator_input = generator_input
        self.batch_size = batch_size
        self.generator = keras.Sequential([keras.layers.Conv2DTranspose(1024,(4,4)),
                                            keras.layers.BatchNormalization(),
                                            keras.layers.ReLU(),
                                            keras.layers.Conv2DTranspose(512,(5,5),strides=(2,2),padding="same"),
                                            keras.layers.BatchNormalization(),
                                            keras.layers.ReLU(),
                                            keras.layers.Conv2DTranspose(256,(5,5),strides=(2,2),padding="same"),
                                            keras.layers.BatchNormalization(),
                                            keras.layers.ReLU(),
                                            keras.layers.Conv2DTranspose(128,(5,5),strides=(2,2),padding="same"),
                                            keras.layers.BatchNormalization(),
                                            keras.layers.ReLU(),
                                            keras.layers.Conv2DTranspose(128,(5,5),strides=(2,2),padding="same"),
                                            keras.layers.BatchNormalization(),
                                            keras.layers.ReLU(),
                                            keras.layers.Conv2DTranspose(3,(5,5),strides=(2,2),padding="same",activation="tanh")
        ])
        self.discriminator = keras.Sequential([keras.layers.Conv2D(128,(5,5),strides=(2,2),padding="same",use_bias=False),
                                                # keras.layers.BatchNormalization(),
                                                keras.layers.LayerNormalization(),
                                                keras.layers.LeakyReLU(alpha=0.2),

                                                keras.layers.Conv2D(128,(5,5),strides=(2,2),padding="same",use_bias=False),
                                                # keras.layers.BatchNormalization(),
                                                keras.layers.LayerNormalization(),
                                                keras.layers.LeakyReLU(alpha=0.2),

                                                keras.layers.Conv2D(256,(5,5),strides=(2,2),padding="same",use_bias=False),
                                                # keras.layers.BatchNormalization(),
                                                keras.layers.LayerNormalization(),
                                                keras.layers.LeakyReLU(alpha=0.2),

                                                keras.layers.Conv2D(512,(5,5),strides=(2,2),padding="same",use_bias=False),
                                                # keras.layers.BatchNormalization(),
                                                keras.layers.LayerNormalization(),
                                                keras.layers.LeakyReLU(alpha=0.2),

                                                keras.layers.Conv2D(1024,(5,5),strides=(2,2),padding="same",use_bias=False),
                                                # keras.layers.BatchNormalization(),
                                                keras.layers.LayerNormalization(),
                                                keras.layers.LeakyReLU(alpha=0.2),

                                                # keras.layers.Conv2D(1,(4,4),activation="sigmoid",use_bias=False)
                                                keras.layers.Conv2D(1,(4,4),use_bias=False)
        ])
        self.discriminator_opt = keras.optimizers.Adam(learning_rate=5e-5,beta_1=0.)
        self.generator_opt = keras.optimizers.Adam(learning_rate=5e-5,beta_1=0.)
        # self.optimizer = keras.optimizers.Adam()
    def generate(self,inputs):
        return self.generator(inputs,training=False)

    def train_step(self,data):
        x_real = tf.cast(data,tf.float32)
        x_real-=127.5
        x_real/=128
        x_fake = self.generator(tf.random.uniform((self.batch_size,1,1,self.generator_input),minval=-1, maxval=1),training=True)

        with tf.GradientTape() as tape_d:
            d_pred_real = self.discriminator(x_real,training=True)
            d_pred_fake = self.discriminator(x_fake,training=True)
            # real_loss = BCE(tf.ones_like(d_pred_real),d_pred_real)
            # fake_loss = BCE(tf.zeros_like(d_pred_fake),d_pred_fake)
            # discriminator_loss = (real_loss+fake_loss)/2
            real_loss = tf.math.reduce_mean(d_pred_real)
            fake_loss = tf.math.reduce_mean(d_pred_fake)
            # discriminator_loss=-fake_loss+real_loss
            diff = x_fake-x_real
            interpolate = x_real + (tf.random.uniform((self.batch_size, 1,1,1),0,1) * diff)
            gradient = tf.gradients(self.discriminator(interpolate), [interpolate])[0]
            slope = tf.sqrt(tf.math.reduce_sum(tf.square(gradient),axis=1))
            gp = tf.reduce_mean((slope - 1) ** 2)
            discriminator_loss=-fake_loss+real_loss+gp
        gradients_d = tape_d.gradient(discriminator_loss, self.discriminator.trainable_variables)
        self.discriminator_opt.apply_gradients(zip(gradients_d, self.discriminator.trainable_variables))
        with tf.GradientTape() as tape_g:
            x_fake_g = self.generator(tf.random.uniform((self.batch_size,1,1,self.generator_input),minval=-1, maxval=1),training=True)
            g_pred = self.discriminator(x_fake_g,training=True)
            # generator_loss = BCE(tf.ones_like(g_pred),g_pred)
            generator_loss = tf.math.reduce_mean(g_pred)
        gradients_g = tape_g.gradient(generator_loss, self.generator.trainable_variables)
        self.generator_opt.apply_gradients(zip(gradients_g, self.generator.trainable_variables))

        self.discriminator_loss_metric.update_state(discriminator_loss)
        self.generator_loss_metric.update_state(generator_loss)
        return {"d_loss": discriminator_loss, "d_loss_mean": self.discriminator_loss_metric.result(), "g_loss": generator_loss, "g_loss_mean": self.generator_loss_metric.result()}

def test():
    model = Model(1)
    print(model.generator(tf.random.normal((10,1,1,128))).shape)
    print(model.discriminator(tf.random.normal((10,128,128,3))).shape)
if __name__=="__main__":
    test()