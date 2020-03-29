import numpy as np

activations = {
    "sigmoid": lambda x: 1 / (1 + np.exp(-x)),
    "tanh": lambda x: np.tanh(x),
    "ReLU": lambda x: np.maximum(0, x),
    "leakyReLU": lambda x: np.maximum(0.125 * x, x),
    "softmax": lambda x: np.exp(x) / np.sum(np.exp(x), axis=1)[:, None]
}

loss_functions = {
    "L1_loss": lambda y, ty: np.abs(ty - y),
    "L2_loss": lambda y, ty: np.square(ty - y),
    "Bin_cross_entropy": lambda y, ty: -y * np.log(ty) - (1 - y) * np.log(1 - ty),
    "Mul_cross_entropy": lambda y, ty: -y * np.log(ty)
}

activation_gradients = {
    "sigmoid": lambda y: y * (1 - y),
    "tanh": lambda y: 1 - np.square(y),
    'ReLU': lambda y: y > 0,
    "leakyReLU": lambda y: (y > 0) + (y <= 0) * 0.125
}

output_layer_gradients = {
    "sigmoid": lambda y, ty: ty - y,
    "softmax": lambda y, ty: ty - y
}

loss_fun_gradients = {
    "L1_loss": lambda y, ty: (ty >= y) - (ty < y),
    "L2_loss": lambda y, ty: 2 * (ty - y),
    "Bin_cross_entropy": lambda y, ty: -y / ty + (1 - y) / (1 - ty),
    "Mul_cross_entropy": lambda y, ty: -y / ty
}