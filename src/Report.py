import os
import numpy as np
from keras.losses import CategoricalCrossentropy
from keras.metrics import CategoricalAccuracy
from keras.optimizers import Adam
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from LoadData import LoadData
from EncoderOnlyModel import EncoderOnlyModel

len_train, len_validation, len_test = 6400, 800, 800
num_heads, d_model, dff, dropout_rate = 2, 128, 512, 0.1

class Report():
    def __init__(self):
        self.load_data = LoadData()

    def generate_report(self):
        train_input, train_output = self.load_data.get_data_train(len_train).as_numpy_iterator().next()
        validation_input, validation_output = self.load_data.get_data_validation(
            len_validation).as_numpy_iterator().next()
        test_input, test_output = self.load_data.get_data_test(len_test).as_numpy_iterator().next()

        model = EncoderOnlyModel(num_heads=num_heads, d_model=d_model, dff=dff,
                                 target_vocab_size=len(self.load_data.get_commands()), rate=dropout_rate)

        model_path = "files/checkpoints"
        model_weights_file = os.listdir(model_path)[0]
        model.build((None, None, 129))
        model.load_weights(os.path.join(model_path, model_weights_file))
        model.compile(optimizer=Adam(), loss=CategoricalCrossentropy(), metrics=[CategoricalAccuracy()])

        train_precision = model.evaluate(x=train_input, y=train_output)[1]
        validation_precision = model.evaluate(x=validation_input, y=validation_output)[1]
        test_precision = model.evaluate(x=test_input, y=test_output)[1]

        train_predictions = np.argmax(model.predict(train_input), axis=1)
        validation_predictions = np.argmax(model.predict(validation_input), axis=1)
        test_predictions = np.argmax(model.predict(test_input), axis=1)

        train_confusion = np.array(confusion_matrix(np.argmax(train_output, axis=1), train_predictions))
        validation_confusion = np.array(confusion_matrix(np.argmax(validation_output, axis=1), validation_predictions))
        test_confusion = np.array(confusion_matrix(np.argmax(test_output, axis=1), test_predictions))

        fig = plt.figure()
        report_path = "files/report"
        os.makedirs(report_path, exist_ok=True)
        report = f';Treinamento;Validação;Teste;Geral\nPrecisão;{np.mean(train_precision):.4f};{np.mean(validation_precision):.4f};{np.mean(test_precision):.4f}'
        with open(os.path.join(report_path, "report.csv"), 'w') as file:
            file.write(report)

        self.save_confusion(tittle="Treinamento", confusion=train_confusion)
        self.save_confusion(tittle="Validação", confusion=validation_confusion)
        self.save_confusion(tittle="Teste", confusion=test_confusion)

    def save_confusion(self, tittle, confusion):
        sns.heatmap(confusion, annot=True, fmt='d', cmap='YlGnBu', cbar=False)
        plt.title(tittle)
        plt.xlabel(f'Valores Previstos')
        plt.ylabel(f'Valores Reais')
        class_names = self.load_data.get_commands().values()
        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks + 0.5, class_names)
        plt.yticks(tick_marks + 0.5, class_names)
        plt.savefig(f'files/report/confusion-{str(tittle).lower()}.png', format='png', bbox_inches='tight',
                    pad_inches=0)
        plt.clf()

Report().generate_report()
