import numpy as np

def load(path):
    data = np.genfromtxt(path, dtype = float, delimiter = ',')

    temp = data[:, :1].T
    features = data[:, 1:].T

    labels = np.zeros((int(temp.max() + 1), temp.shape[1]))

    for i in range(temp.shape[1]):
        labels[int(temp[0, i]), i] = 1

    return features, labels

def scale(data):
    bias = data.mean(axis = 1)[:, None]

    zoom = (data.max(axis = 1) - data.min(axis = 1))[:, None]
    zoom = zoom + (zoom == 0)

    return (data - bias) / zoom, bias, zoom