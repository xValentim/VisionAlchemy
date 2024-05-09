import tensorflow as tf
from tensorflow import keras
from functools import partial

DefaultConv2D = partial(tf.keras.layers.Conv2D, kernel_size=3, strides=1,
                        padding="same", kernel_initializer="he_normal",
                        use_bias=False)

DefaultDeConv2D = partial(tf.keras.layers.Conv2DTranspose, kernel_size=3, strides=2,
                          padding="same", kernel_initializer="he_normal",
                          use_bias=False)

class ResidualUnit(tf.keras.layers.Layer):
    def __init__(self, filters, strides=1, activation="relu", **kwargs):
        super().__init__(**kwargs)
        self.activation = tf.keras.activations.get(activation)
        self.main_layers = [
            DefaultConv2D(filters, strides=strides),
            tf.keras.layers.BatchNormalization(),
            self.activation,
            DefaultConv2D(filters),
            tf.keras.layers.BatchNormalization()
        ]
        self.skip_layers = []
        if strides > 1:
            self.skip_layers = [
                DefaultConv2D(filters, kernel_size=1, strides=strides),
                tf.keras.layers.BatchNormalization()
            ]

    def call(self, inputs):
        Z = inputs
        for layer in self.main_layers:
            Z = layer(Z)
        skip_Z = inputs
        for layer in self.skip_layers:
            skip_Z = layer(skip_Z)
        return self.activation(Z + skip_Z)
    
class ResidualDeConvUnit(tf.keras.layers.Layer):
    def __init__(self, filters, strides=1, activation="relu", **kwargs):
        super().__init__(**kwargs)
        self.activation = tf.keras.activations.get(activation)
        self.main_layers = [
            DefaultDeConv2D(filters, strides=strides),
            tf.keras.layers.BatchNormalization(),
            self.activation,
            DefaultConv2D(filters),
            tf.keras.layers.BatchNormalization()
        ]
        self.skip_layers = []
        if strides > 1:
            self.skip_layers = [
                DefaultDeConv2D(filters, kernel_size=1, strides=strides),
                tf.keras.layers.BatchNormalization()
            ]
        
    def call(self, inputs):
        Z = inputs
        for layer in self.main_layers:
            Z = layer(Z)
        skip_Z = inputs
        for layer in self.skip_layers:
            skip_Z = layer(skip_Z)
        return self.activation(Z + skip_Z)
    
class Sampling(tf.keras.layers.Layer):
    def call(self, inputs):
        mean, log_var = inputs
        return tf.random.normal(tf.shape(log_var)) * tf.exp(log_var / 2) + mean
    
class EncoderResnetLike(keras.Model):
    def __init__(self, input_shape, codings_size):
        super(EncoderResnetLike, self).__init__()
        self.codings_size = codings_size
        self.inputs = keras.layers.Input(shape=input_shape)
        self.Z = DefaultConv2D(64, kernel_size=3, strides=2)(self.inputs)
        self.Z = keras.layers.Activation("relu")(self.Z)
        self.Z = keras.layers.MaxPool2D(pool_size=3, strides=2, padding="same")(self.Z)

        prev_filters = 64
        for filters in [64] * 1 + [128] * 2 + [256] * 2 + [512] * 1:
            strides = 1 if filters == prev_filters else 2
            self.Z = ResidualUnit(filters, strides=strides)(self.Z)
            prev_filters = filters

        self.Z = keras.layers.GlobalAvgPool2D()(self.Z)
        self.Z = keras.layers.Flatten()(self.Z)

        self.codings_mean = keras.layers.Dense(codings_size)(self.Z)  # μ
        self.codings_log_var = keras.layers.Dense(codings_size)(self.Z)  # γ
        self.codings = Sampling()([self.codings_mean, self.codings_log_var])
        self.variational_encoder = keras.Model(
            inputs=[self.inputs], outputs=[self.codings_mean, self.codings_log_var, self.codings])
    
    def call(self, inputs):
        return self.variational_encoder(inputs)
    
class DecoderResnetLike(keras.Model):
    def __init__(self, input_shape, codings_size):
        super(DecoderResnetLike, self).__init__()
        self.codings_size = codings_size
        self.decoder_inputs = keras.layers.Input(shape=[self.codings_size])
        self.x = keras.layers.Dense(4 * 4 * (512))(self.decoder_inputs)
        self.x = keras.layers.Reshape((4, 4, 512))(self.x)
        self.x = keras.layers.Conv2DTranspose(512, 3, padding='same', activation='relu')(self.x)
        self.x = keras.layers.UpSampling2D(2)(self.x)
        prev_filters = 512
        for filters in ([64] * 1 + [128] * 2 + [256] * 2 + [512] * 1)[::-1]:
            strides = 1 if filters == prev_filters else 2
            self.x = ResidualDeConvUnit(filters, strides=strides)(self.x)
            prev_filters = filters

        self.x = keras.layers.Conv2DTranspose(32, 3, padding='same', activation='relu')(self.x)
        self.x = keras.layers.UpSampling2D(2)(self.x)
        self.x = keras.layers.Conv2DTranspose(3, 3, padding='same', activation='sigmoid')(self.x)
        self.outputs = keras.layers.Reshape([128, 128, 3])(self.x)
        self.variational_decoder = keras.Model(inputs=[self.decoder_inputs], outputs=[self.outputs])
    
    def call(self, inputs):
        return self.variational_decoder(inputs)
    

class VariationalAutoEnconder(keras.Model):
    def __init__(self, input_shape, codings_size):
        super(VariationalAutoEnconder, self).__init__()
        self.encoder = EncoderResnetLike(input_shape=input_shape, codings_size=codings_size)
        self.decoder = DecoderResnetLike(input_shape=input_shape, coddings_size=codings_size)
        
        _, _, self.codings = self.encoder(self.encoder.inputs)
        self.reconstructions = self.decoder(self.codings)
        self.variational_ae = keras.Model(inputs=[self.encoder.inputs], outputs=[self.reconstructions])
        
    def add_dkl_loss(self):
        self.latent_loss = -0.5 * tf.reduce_sum(
        1 + self.encoder.codings_log_var - tf.exp(self.encoder.codings_log_var) - tf.square(self.encoder.codings_mean),
        axis=-1)
        self.variational_ae.add_loss(tf.reduce_mean(self.latent_loss) / (128.0 * 128.0))
        
    def call(self, inputs):
        x = self.encoder(inputs)
        x = self.decoder(x)
        return x
    