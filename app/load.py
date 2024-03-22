import os
import pathlib
import numpy as np
import tensorflow as tf
from keras.metrics import CategoricalAccuracy
from keras.losses import CategoricalCrossentropy
from keras.optimizers import Adam
from app.model import EncoderOnlyModel

AUTOTUNE = tf.data.AUTOTUNE


class LoadData():
    def __init__(self):
        self.commands = self.get_commands()
        self.base_data_path = 'files/data'
        self.data_path = f'{self.base_data_path}/mini_speech_commands'
        self.data_dir = pathlib.Path(self.data_path)
        self.train_files, self.validation_files, self.test_files = None, None, None
        # if not self.data_dir.exists():
        #     self.download_data(base_data_path)
        # self.train_files, self.validation_files, self.test_files = self.randomize_data()

    def get_commands(self):
        return {0: 'down', 1: 'go', 2: 'left', 3: 'no', 4: 'right', 5: 'stop', 6: 'up', 7: 'yes'}

    def get_data(self):
        if not self.data_dir.exists():
            origin = "http://storage.googleapis.com/download.tensorflow.org/" \
                     "data/mini_speech_commands.zip"
            tf.keras.utils.get_file('mini_speech_commands.zip', origin=origin, extract=True, cache_dir='.',
                                    cache_subdir=self.base_data_path)
        self.train_files, self.validation_files, self.test_files = self.randomize_data()

    def randomize_data(self):
        file_names = sorted(tf.io.gfile.glob(str(self.data_dir) + '/*/*'))
        quantity_for_commands = 1000
        train_files, validation_files, test_files = [], [], []
        files = [train_files, validation_files, test_files]
        for i in range(0, len(file_names), quantity_for_commands):
            np.random.seed(1)
            files_command = file_names[i:i + quantity_for_commands]
            np.random.shuffle(files_command)
            train_files += files_command[:800]
            validation_files += files_command[800:900]
            test_files += files_command[900:]
        for file_set in files:
            np.random.seed(1)
            np.random.shuffle(file_set)
        return files

    def get_label_by_command(self, parts):
        commands_values = tf.constant(list(self.get_commands().values()))
        index = tf.where(tf.equal(commands_values, parts))[0][0]
        len_commands = tf.shape(commands_values)[0]
        return tf.one_hot(index, len_commands, dtype=tf.float32)

    def get_command_by_label(self, one_hot):
        one_hot_np = one_hot.numpy()
        index = np.argmax(one_hot_np)
        return tf.convert_to_tensor(self.get_commands()[index])

    def get_label(self, file_path):
        parts = tf.strings.split(input=file_path, sep=os.path.sep)
        return self.get_label_by_command(parts[-2])

    def decode_audio(self, audio_binary):
        audio, _ = tf.audio.decode_wav(contents=audio_binary)
        return tf.squeeze(audio, axis=-1)

    def get_spectrogram(self, waveform):
        input_len = 16000
        waveform = waveform[:input_len]
        zero_padding = tf.zeros([16000] - tf.shape(waveform), dtype=tf.float32)
        waveform = tf.cast(waveform, dtype=tf.float32)
        equal_length = tf.concat([waveform, zero_padding], 0)
        spectrogram = tf.signal.stft(equal_length, frame_length=255, frame_step=128)
        spectrogram = tf.abs(spectrogram)
        return spectrogram

    def get_stft_and_label(self, file_path):
        label = self.get_label(file_path)
        audio_binary = tf.io.read_file(file_path)
        waveform = self.decode_audio(audio_binary)
        spectrogram = self.get_spectrogram(waveform)
        return spectrogram, label

    def create_ds_batch(self, files, batch_size):
        data = tf.data.Dataset.from_tensor_slices(files) \
            .map(map_func=self.get_stft_and_label, num_parallel_calls=AUTOTUNE)
        return data.batch(batch_size=batch_size) if batch_size is not None else data

    def get_data_train(self, batch_size):
        train_files = self.train_files
        train = self.create_ds_batch(files=train_files, batch_size=batch_size)
        return train

    def get_data_validation(self, batch_size):
        val_files = self.validation_files
        validation = self.create_ds_batch(files=val_files, batch_size=batch_size)
        return validation

    def get_data_test(self, batch_size):
        test_files = self.test_files
        test = self.create_ds_batch(files=test_files, batch_size=batch_size)
        return test

    def load_weights_predict(self, model_path):
        num_heads, d_model, dff, dropout_rate = 2, 128, 512, 0.1
        model = EncoderOnlyModel(num_heads=num_heads, d_model=d_model, dff=dff,
                                 target_vocab_size=len(self.get_commands()), rate=dropout_rate)
        model_weights_file = os.listdir(model_path)[0]
        model.build((None, None, 129))
        model.load_weights(os.path.join(model_path, model_weights_file))
        model.compile(optimizer=Adam(), loss=CategoricalCrossentropy(), metrics=[CategoricalAccuracy()])
        return model
