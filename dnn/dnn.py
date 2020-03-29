from dnn.data_process import*
from dnn.functions import*

import numpy as np
import math

class Dnn:

    def __init__(self):
        self.activation = {
            "hidden layer": "ReLU",
            "output layer": "sigmoid"
        }

        self.layers = []
        self.keep_probs = []

        self.w = []
        self.b = []
        self.vdw = []#momentum param
        self.vdb = []#momentum param
        self.sdw = []#RMSprop param
        self.sdb = []#RMSprop param

        self.mean = []#mean value of features
        self.zoom = []#zoom value of features

        self.alpha = 0.02#learning rate
        self.beta1 = 0.9#momentum param
        self.beta2 = 0.999#RMSprop param


    def set_hidden_layer_units(self, units):
        self.layers = units


    def set_keep_probs(self, keep_probs):
        self.keep_probs = keep_probs


    def set_hidden_activation(self, activation):
        self.activation["hidden layer"] = activation


    def set_output_activation(self, activation):
        self.activation["output layer"] = activation


    def init_params(self, features, labels):
        self.layers = [features.shape[0]] + self.layers + [labels.shape[0]]

        for i in range(len(self.layers) - 1):
            sigma = math.sqrt(2 / self.layers[i])

            self.w.append(np.random.normal(0, sigma, (self.layers[i + 1], self.layers[i])))
            self.b.append(np.random.normal(0, sigma, (self.layers[i + 1], 1)))

            self.vdw.append(self.w[-1] - self.w[-1])
            self.vdb.append(self.b[-1] - self.b[-1])
            self.sdw.append(self.w[-1] - self.w[-1])
            self.sdb.append(self.b[-1] - self.b[-1])


    def get_loss(self, features, labels):
        self.layers = [features]
        self.for_prop()

        self.layers[-1] += (self.layers[-1] == 1) * -1e-16 + (self.layers[-1] == 0) * 1e-16

        loss = -(labels * np.log(self.layers[-1]) + (1 - labels) * np.log(1 - self.layers[-1]))
        return np.sum(loss) / loss.shape[0] / loss.shape[1]


    def get_acc(self, features, labels):
        self.layers = [features]
        self.for_prop()

        r1, c1 = np.where(labels.T == np.max(labels.T, axis = 1)[:, None])
        r2, c2 = np.where(self.layers[-1].T == np.max(self.layers[-1].T, axis = 1)[:, None])

        return np.sum(c1 == c2) / c1.shape[0]


    def dropout(self, layer, keep_prob):
        mark = np.random.rand(layer.shape[0], layer.shape[1]) <= keep_prob
        return layer * mark / keep_prob


    def for_prop(self, drop_flag = False):
        for i in range(len(self.w)):
            if drop_flag and len(self.keep_probs):
                self.layers[-1] = self.dropout(self.layers[-1], self.keep_probs[i])

            activation = self.activation["output layer"] if i == len(self.w) - 1 else self.activation["hidden layer"]
            self.layers.append(activations[activation](np.dot(self.w[i], self.layers[-1]) + self.b[i]))


    def back_prop(self, labels, t):
        error = np.array([[]])
        for i in reversed(range(len(self.w))):
            if i == len(self.w) - 1:##未完待续
                gradient = output_layer_gradients[self.activation["output layer"]](labels, self.layers[i + 1])
                error = gradient
            else:
                gradient = activation_gradients[self.activation["hidden layer"]](self.layers[i + 1])
                error = gradient * (np.dot(error.T, self.w[i + 1])).T

            dw = np.dot(error, self.layers[i].T) / self.layers[i].shape[1]
            db = np.sum(error, axis = 1)[:, None] / self.layers[i].shape[1]

            self.vdw[i] = self.beta1 * self.vdw[i] + (1 - self.beta1) * dw
            vdw = self.vdw[i] / (1 - pow(self.beta1, t))
            self.vdb[i] = self.beta1 * self.vdb[i] + (1 - self.beta1) * db
            vdb = self.vdb[i] / (1 - pow(self.beta1, t))

            self.sdw[i] = self.beta2 * self.sdw[i] + (1 - self.beta2) * np.square(dw)
            sdw = self.sdw[i] / (1 - pow(self.beta2, t))
            self.sdb[i] = self.beta2 * self.sdb[i] + (1 - self.beta2) * np.square(db)
            sdb = self.sdb[i] / (1 - pow(self.beta2, t))

            self.w[i] = self.w[i] - self.alpha * vdw / (np.sqrt(sdw + 1e-8))
            self.b[i] = self.b[i] - self.alpha * vdb / (np.sqrt(sdb + 1e-8))


    def backup_params(self):
        last_params = {}
        last_params["w"] = self.w.copy()
        last_params["b"] = self.b.copy()
        last_params["vdw"] = self.vdw.copy()
        last_params["vdb"] = self.vdb.copy()
        last_params["sdw"] = self.sdw.copy()
        last_params["sdb"] = self.sdb.copy()

        return last_params


    def restore_params(self, last_params):
        self.w = last_params["w"].copy()
        self.b = last_params["b"].copy()
        self.vdw = last_params["vdw"].copy()
        self.vdb = last_params["vdb"].copy()
        self.sdw = last_params["sdw"].copy()
        self.sdb = last_params["sdb"].copy()


    def train(self, features, labels, epoch_size = 16, batch_size = 128):
        features, self.mean, self.zoom = scale(features)

        self.init_params(features, labels)

        t = 1
        last_params = self.backup_params()
        last_loss = self.get_loss(features, labels)

        for epoch in range(epoch_size):
            for i in range(int(features.shape[1] / batch_size)):
                label = labels[:, i * batch_size:(i + 1) * batch_size]
                self.layers = [features[:, i * batch_size:(i + 1) * batch_size]]
                self.for_prop(True)
                self.back_prop(label, t)

                t += 1

                progress = (i + 1) / int(features.shape[1] / batch_size) * 100

                print("\rEpoch " + str(epoch) + ": " + '=' * int(progress / 2) + ">" + str(int(progress)) + "%", end = '')

            loss = self.get_loss(features, labels)

            if loss > last_loss:
                self.restore_params(last_params)
                self.alpha = self.alpha * 0.5
                print("\r" +' ' * 70, end = '')
                continue

            last_params = self.backup_params()
            last_loss = loss
            self.alpha = self.alpha * 1.05

            print("\nloss: " + str(round(loss, 8)) + ", train acc: " + str(round(self.get_acc(features, labels) * 100, 2)) + "%, rate: " + str(self.alpha))