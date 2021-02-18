import numpy as np
import pandas as pd
import torch

# Обработка данных
from torch.nn import functional as F
from joblib import load
from classifier import Classifier


class Model:

    def __init__(self, model_name, classes):
        print('Класс Model загружается')
        # Имя модели
        self.model_name = model_name

        # Пути к файлам
        self.model_folder = './model/'

        # Константы
        self.raw_ni = 14
        self.fft_ni = 14

        # Классы
        self.classes = classes  # ['back', 'forward', 'left', 'neutral', 'right']
        self.num_classes = len(classes)

        self.load_normalization()
        self.load_model()
        print('Загрузка завершена')

    def load_normalization(self):
        """
        Load preprocessing transformers
        Returns
        -------
        out : None
        """
        self.preprocess_X = load(self.model_folder + 'preprocess_X.joblib')
        self.preprocess_X_fft = load(self.model_folder + 'preprocess_X_fft.joblib')

    def load_model(self):
        """
        Load model
        Returns
        -------
        out : None

        """
        self.model = Classifier(self.raw_ni, self.fft_ni, self.num_classes)
        self.model.load_state_dict(torch.load(self.model_folder + self.model_name))
        self.model.eval()

    def process_data(self, X):
        """

        Parameters
        ----------
        X: ndarray
            array shape of (128,14)

        Returns
        -------
        out: string

        """

        X = X[np.newaxis,:]

        X_p = self.preprocess_X.transform(X).astype(np.float32)
        X_fft = self.preprocess_X_fft.transform(X).astype(np.float32)

        X_t = torch.tensor(X_p, dtype=torch.float32)
        X_fft_t = torch.tensor(X_fft, dtype=torch.float32)

        out = self.model.forward(X_t, X_fft_t)

        preds = F.log_softmax(out, dim=1).argmax(dim=1)

        return preds.item()


if __name__ == '__main__':
    # Тест на первую запись в файле
    print('Начало теста')
    n = 0  # Номер записи для проверки
    data = pd.read_csv('concentrate_t.csv')
    test = data.iloc[n * 128: n * 128 + 128].values

    # Пример использования
    mdl = Model('best.pth', ['concentrate', 'relax'])
    result = mdl.process_data(test)

    print('Значение в данных: ', data.iloc[n * 128, 0])
    print('Значение выданное нейросетью', result)
