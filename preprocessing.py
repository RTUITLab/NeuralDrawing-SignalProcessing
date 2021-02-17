from sklearn.base import TransformerMixin
import numpy as np


class FFT(TransformerMixin):
    def __init__(self):
        pass

    def absfft(self, arr):
        """
        apply an abs(fft()) function to array
        Parameters
        ----------
        arr : ndarray

        Returns
        -------
        out : ndarray
        """
        return np.abs(np.fft.rfft(arr))

    def fit(self, X, y=None, **fit_params):
        return self

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y, fit_params=fit_params)
        return self.transform(X, y)

    def transform(self, X, y=None):
        return np.apply_along_axis(self.absfft, 1, X)


class Scale(TransformerMixin):
    def __init__(self):
        pass

    def scale(self, arr):
        """
        subtracts mean by each record for each column
        Parameters
        ----------
        arr: ndarray

        Returns
        -------
        out : ndarray
        """
        out = np.zeros_like(arr)
        out[:] = arr[:]
        for i in range(arr.shape[2]):
            np_arr = arr[:, :, i]
            mean_arr = np.mean(np_arr, 1)
            np_arr = np_arr - mean_arr[:, np.newaxis]
            out[:, :, i] = np_arr
        return out

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y=None, fit_params=fit_params)
        return self.transform(X, y=None)

    def transform(self, X, y=None):
        return self.scale(X)

    def fit(self, X, y=None, **fit_params):
        return self


class Normalize(TransformerMixin):
    def __init__(self):
        pass

    def get_mean(self, arr):
        mean = arr.mean(axis=(0, 1))
        return mean

    def get_std(self, arr):
        std = arr.std(axis=(0, 1))
        return std

    def normalize(self, arr, mean, std):
        """
        function subtract mean by channelwise and divides by std
        Parameters
        ----------
        arr: ndarray

        Returns
        -------
        out: ndarray

        """
        out = arr - mean[np.newaxis, :]
        out /= std[np.newaxis, :]
        return out

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y=None, fit_params=fit_params)
        return self.transform(X)

    def transform(self, X, y=None):
        return self.normalize(X, mean=self.mean, std=self.std)

    def fit(self, X, y=None, **fit_params):
        self.mean = self.get_mean(X)
        self.std = self.get_std(X)


class Transpose(TransformerMixin):
    def __init__(self):
        pass

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y=None, fit_params=fit_params)
        return self.transform(X)

    def transform(self, X, y=None):
        return X.transpose(0, 2, 1)

    def fit(self, X, y=None, **fit_params):
        return self