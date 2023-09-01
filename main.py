from tensorflow import keras
from matplotlib import pyplot
from os import path

ANIMALS = [
    "butterfly", 
    "cat", 
    "chicken", 
    "cow", 
    "dog", 
    "elephant", 
    "horse", 
    "sheep",
    "spider", 
    "squirrel"
]

ACTIVATION_FUNCTIONS = [
    "elu",
    "exponential", 
    "gelu", 
    "hard_sigmoid", 
    "linear", 
    "mish", 
    "relu", 
    "selu",
    "sigmoid", 
    "softmax", 
    "softplus", 
    "softsign", 
    "swish", 
    "tanh"
]

def load_data(directory_name = "categorical"):
    data = keras.utils.image_dataset_from_directory(
        directory_name,
        label_mode = "categorical"
    )
    data = data.map(lambda x, y: (x/255, y))
    train_size = int(len(data) * .7)
    test_size = int(len(data) * .1) + 1
    val_size = int(len(data) * .2) + 1
    training_data = data.take(train_size)
    testing_data = data.skip(train_size).take(test_size)
    validation_data = data.skip(train_size + test_size).take(val_size)
    return training_data, testing_data, validation_data
