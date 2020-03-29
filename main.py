from dnn import dnn

import numpy as np

if __name__ == "__main__":
    model = dnn.Dnn()

    model.set_hidden_layer_units([30])
    #model.set_keep_probs([1, 0.8])
    model.set_hidden_activation("leakyReLU")
    model.set_output_activation("sigmoid")

    features, labels = dnn.load("datas/MNIST/mnist_train.csv")

    model.train(features, labels, batch_size = 192, epoch_size = 16)

    features, labels = dnn.load("datas/MNIST/mnist_test.csv")

    print("test acc: " + str(model.get_acc(features, labels)))