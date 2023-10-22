import os
from LoadData import LoadData
import matplotlib.pyplot as plt
from keras.optimizers import Adam
from DisplayOutputs import DisplayOutputs
from keras.callbacks import ModelCheckpoint
from EncoderOnlyModel import EncoderOnlyModel
from keras.metrics import CategoricalAccuracy
from keras.losses import CategoricalCrossentropy

batch_size, test_batch_size = 32, 4
epochs, num_heads, d_model, dff, dropout_rate = 400, 2, 128, 512, 0.1

class Train():
    def __init__(self):
        self.load_data = LoadData()

    def train_model(self):
        train = self.load_data.get_data_train(batch_size=batch_size)
        validation = self.load_data.get_data_validation(batch_size=batch_size)
        test = self.load_data.get_data_test(batch_size=test_batch_size)
        batch_test = next(iter(test))

        model = EncoderOnlyModel(num_heads=num_heads, d_model=d_model, dff=dff, target_vocab_size=len(self.load_data.get_commands()), rate=dropout_rate)

        checkpoint_path = "files/checkpoints"
        os.makedirs(checkpoint_path, exist_ok=True)
        checkpoints = os.listdir(checkpoint_path)
        checkpoint = checkpoints[0] if len(checkpoints) > 0 else None

        model.build((None, 124, 129))
        if checkpoint:
            model.load_weights(os.path.join(checkpoint_path, checkpoint))

        model.compile(optimizer=Adam(), loss=CategoricalCrossentropy(), metrics=[CategoricalAccuracy()])

        display_cb = DisplayOutputs(batch=batch_test, commands=self.load_data.get_commands())
        ckpt_cb = ModelCheckpoint(filepath=f'{checkpoint_path}/model.h5', save_best_only=True, save_weights_only=True, monitor='val_categorical_accuracy', mode='max', save_freq='epoch')

        history = model.fit(train, validation_data=validation, callbacks=[display_cb, ckpt_cb], epochs=epochs)

        plt.figure()
        plt.plot(history.history['loss'], label='Train Loss')
        plt.title("Treinamento")
        plt.xlabel('Épocas')
        plt.ylabel('Perda')
        plt.savefig(f'files/report/train_loss.png', format='png', bbox_inches='tight', pad_inches=0)
        plt.clf()
        plt.plot(history.history['val_categorical_accuracy'], label='Validation Accuracy')
        plt.title("Validação")
        plt.xlabel('Épocas')
        plt.ylabel('Precisão')
        plt.savefig(f'files/report/validation_accuracy.png', format='png', bbox_inches='tight', pad_inches=0)
        plt.clf()

Train().train_model()
