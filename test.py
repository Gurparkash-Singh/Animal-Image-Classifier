from os import path
from main import ANIMALS, Model, Models
from matplotlib import pyplot
from numpy import expand_dims, argmax
from tensorflow import image, keras

def get_image(filename):
    img = pyplot.imread(filename)
    img = image.resize(img, (256, 256))
    pyplot.axis('off')
    pyplot.imshow(img.numpy().astype(int))
    pyplot.show()
    return expand_dims(img/255, 0)

def process_yhat(yhat):
    yhat = yhat.replace("[", "")
    yhat = yhat.replace("]", "")
    yhat = yhat.split(" ")
    index = 0
    while index < len(yhat):
        yhat[index] = yhat[index].strip()
        if yhat[index] == "":
            yhat.pop(index)
            index -= 1
        else:
            yhat[index] = float(yhat[index])
        index += 1
    return yhat

def get_max(array):
    index = 0
    max_value = array[0]
    for idx, value in enumerate(array):
        if value > max_value:
            index = idx
            max_value = value
    return index, max_value

def main(filename):
    print(path.join("models", "tanh_5.keras"))
    model = keras.models.load_model(path.join("models", "tanh_5.keras"))
    img = get_image(filename)
    yhat = model.predict(img)
    yhat = str(yhat)
    yhat = process_yhat(yhat)
    index, _ = get_max(yhat)
    print(yhat)
    print(index)
    print(f"The Animal is a {ANIMALS[index]}")
    return

if __name__ == "__main__":
    main("dog.jpg")
    main("cat.jpg")
