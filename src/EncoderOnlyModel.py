import tensorflow as tf
from keras import layers


class EncoderOnlyModel(tf.keras.Model):
    def __init__(self, target_vocab_size, num_heads=2, d_model=128, dff=512, rate=0.1):
        super(EncoderOnlyModel, self).__init__()

        self.d_model = d_model

        self.enc_layers = tf.keras.Sequential([
            layers.Dense(d_model, activation='relu')
        ])

        self.mha = layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model)

        self.ffn = tf.keras.Sequential([
            layers.Dense(dff),
            layers.Dense(d_model, activation='relu')
        ])

        self.layer_norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layer_norm2 = layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = layers.Dropout(rate=rate)
        self.dropout2 = layers.Dropout(rate=rate)

        self.final_layer = layers.Dense(target_vocab_size)
        self.global_average_pooling = layers.GlobalAveragePooling1D()

    def call(self, inputs, training=True):
        inputs = self.enc_layers(inputs)
        outputs = self.mha(inputs, inputs)

        outputs = self.dropout1(outputs, training=training)
        inputs += outputs
        inputs = self.layer_norm1(inputs)

        outputs = self.ffn(inputs)

        # Dropout and residual connection
        outputs = self.dropout2(outputs, training=training)
        inputs += outputs
        inputs = self.layer_norm2(inputs)

        outputs = self.final_layer(inputs)
        outputs = self.global_average_pooling(outputs)
        probabilities = tf.nn.softmax(outputs)

        return probabilities
