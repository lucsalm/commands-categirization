import numpy as np
import tensorflow as tf
from tensorflow import keras

class DisplayOutputs(keras.callbacks.Callback):
    def __init__(self, batch, commands):
        self.batch = batch
        self.commands = commands

    def on_epoch_end(self, epoch, logs=None):
        audio = self.batch[0]
        target_label = self.batch[1].numpy()
        batch_size = tf.shape(audio)[0]
        predicted_label = self.model.predict(audio)
        for i in range(batch_size):
            command_target = self.commands[np.argmax(target_label[i, :])]
            command_predicted = self.commands[np.argmax(predicted_label[i, :])]
            print(f"Target: {command_target}\nPrediction: {command_predicted}\n")
