import os
import tensorflow as tf
from keras.callbacks import ModelCheckpoint
from LoadData import LoadData
from DisplayOutputs import DisplayOutputs
from EncoderOnlyModel import EncoderOnlyModel
import matplotlib.pyplot as plt

epochs = 400
batch_size = 32
test_batch_size = 4
num_heads = 2
d_model = 128
dff = 512
dropout_rate = 0.1


def train_model():
    load_data = LoadData()
    commands = load_data.get_commands()
    train = load_data.get_data_train(batch_size=batch_size)
    validation = load_data.get_data_validation(batch_size=batch_size)
    test = load_data.get_data_test(batch_size=test_batch_size)
    batch_test = next(iter(test))

    model = EncoderOnlyModel(
        num_heads=num_heads,
        d_model=d_model,
        dff=dff,
        target_vocab_size=len(commands),
        rate=dropout_rate
    )

    checkpoint_path = "files/checkpoints"
    os.makedirs(checkpoint_path, exist_ok=True)
    checkpoints = os.listdir(checkpoint_path)
    checkpoint = checkpoints[0] if len(checkpoints) > 0 else None

    model.build((None, 124, 129))
    if checkpoint:
        model.load_weights(os.path.join(checkpoint_path, checkpoint))

    loss_object = tf.keras.losses.CategoricalCrossentropy()
    optimizer = tf.keras.optimizers.Adam()
    accuracy = tf.metrics.CategoricalAccuracy()

    model.compile(optimizer=optimizer, loss=loss_object, metrics=[accuracy])

    display_cb = DisplayOutputs(batch=batch_test, commands=commands)
    ckpt_cb = ModelCheckpoint(
        filepath=f'{checkpoint_path}/model.h5',
        save_best_only=True,
        save_weights_only=True,
        monitor='val_categorical_accuracy',
        mode='max',
        save_freq='epoch',
    )

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

    print("Finalizado")


train_model()
