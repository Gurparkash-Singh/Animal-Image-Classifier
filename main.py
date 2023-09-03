from tensorflow import keras
from matplotlib import pyplot
from os import path

SEP = "=" * 100

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

class Model(keras.Sequential):
    def __init__(self, name, activations):
        super().__init__(name=name)
        self.history = None
        self.calculations = {
            "precision": None,
            "recall": None,
            "accuracy": None,
            "loss": None,
            "average": None
        }
        self.activations = activations
        self.figure = None
        self._create_model()
        return

    def _create_model(self):
        super().add(
            keras.layers.AveragePooling2D(3, 3, input_shape = (256,256,3))
        )
        super().add(
            keras.layers.Conv2D(64, (3,3), activation = self.activations[0])
        )
        super().add(keras.layers.MaxPooling2D())

        super().add(
            keras.layers.Conv2D(32, (3,3), activation = self.activations[1])
        )
        super().add(keras.layers.MaxPooling2D())

        super().add(keras.layers.Normalization())

        super().add(
            keras.layers.Conv2D(64, (6,6), activation = self.activations[2])
        )
        super().add(keras.layers.MaxPooling2D())

        super().add(
            keras.layers.Conv2D(32, (6,6), activation = self.activations[3])
        )
        super().add(keras.layers.MaxPooling2D())

        super().add(keras.layers.Flatten())
        super().add(keras.layers.Dropout(0.75))
        super().add(
            keras.layers.Dense(
                256,
                activation = self.activations[4]
            )
        )
        super().add(
            keras.layers.Dense(
                128,
                activation = self.activations[5]
            )
        )
        super().add(
            keras.layers.Dense(
                10,
                activation = self.activations[6],
                kernel_regularizer = keras.regularizers.L1(0.01),
                activity_regularizer = keras.regularizers.L2(0.1)
            )
        )
        super().compile(
            optimizer = 'adam',
            loss = "categorical_crossentropy",
            metrics = [
                "accuracy"
            ]
        )
        return
    
    def train(self, data, epoch, logdir = "logs"):
        tensorboard_callback = keras.callbacks.TensorBoard(log_dir = logdir)
        self.history = super().fit(
            data["training"],
            epochs = epoch,
            validation_data = data["validation"],
            callbacks = tensorboard_callback
        )
        self._performance_plot()
        self.figure.savefig(f"./results/{self.__str__()}.png")
        self._evaluate(data)
        self.calculations["average"] = (
            self.calculations["accuracy"]
            + self.calculations["precision"]
            + self.calculations["recall"]
        ) / 3
        self.save(self.__str__())
        return
    
    def _evaluate(self, data):
        print("Evaluating Data")
        loss, _ = super().evaluate(data["testing"])

        precision = keras.metrics.Precision()
        recall = keras.metrics.Recall()
        accuracy = keras.metrics.CategoricalAccuracy()

        print("Calculating Precision, Recall and Accuracy")
        for batch in data["testing"].as_numpy_iterator():
            x, y = batch
            yhat = super().predict(x)
            precision.update_state(y, yhat)
            recall.update_state(y, yhat)
            accuracy.update_state(y, yhat)
        
        self.calculations["precision"] = precision.result().numpy()
        self.calculations["recall"] = recall.result().numpy()
        self.calculations["accuracy"] = accuracy.result().numpy()
        self.calculations["loss"] = loss
        return
    
    def save(self, filename, directory = "models"):
        super().save(path.join(directory, f"{filename}.keras"))
        return
    
    def _performance_plot(self):
        figure = pyplot.figure()
        pyplot.title(self.__str__())
        pyplot.plot(self.history.history["loss"], color = "red", label = "loss")
        pyplot.plot(
            self.history.history["val_loss"], color = "orange", label = "val_loss"
        )
        pyplot.plot(
            self.history.history["accuracy"], color = "green", label = "accuracy"
        )
        pyplot.plot(
            self.history.history["val_accuracy"],
            color = "blue",
            label = "val_accuracy"
        )
        pyplot.legend(loc = "upper left")
        self.figure = figure
        return
    
    def __str__(self):
        if self.name == None:
            s = f"{self.activations}"
        else:
            s = self.name
        return s

class Models:
    def __init__(self, epoch, best_epoch):
        """
        epoch = number of epoch for all models
        best-epoch = number of epoch for best performing models
        """
        self.best_models = {
            "precision": None,
            "recall": None,
            "accuracy": None,
            "loss": None,
            "average": None
        }
        self.epoch = 2
        self.best_epoch = 2
        self.data = None
        self.models = []
        self.best_metric = None
        self._load_data()
        self._evaluate_metrics()
        return
    
    def _load_data(self, directory_name = "images"):
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
        data = {
            "training": training_data, 
            "testing": testing_data, 
            "validation": validation_data
        }
        self.data = data
        return
    
    def _evaluate_metrics(self):
        with open(f"./results/result.txt", "wt") as file:
            file.write(f"Result: \n")
            for activation in ACTIVATION_FUNCTIONS:
                activations = [activation] * 7
                model = Model(f"{activation}_all", activations)
                model.train(self.data, self.epoch)
                self.models.append(model)
                self.write_metrics(file, model, model.calculations)
            self._models_evaluate()
            self.write_metrics(file, "Best Models", self.best_models)
            for key in self.best_models:
                self.best_models[key].train(self.data, self.best_epoch)
            self._models_evaluate()
            for key in self.best_models:
                if key != "accuracy" and key != "loss":
                    if self.best_models[key] == self.best_models["accuracy"]:
                        self.best_metric = key
            self.write_metrics(file, "Best Models", self.best_models)
            if self.best_metric == None:
                print("No Best Metric Selected")
                return
            self._create_best_model()
        return
    
    def _create_best_model(self):
        with open(f"./results/result.txt", "at") as file:
            for i in len(self.models[0].activations):
                activations = self.best_models[self.best_metric].activations
                for function in ACTIVATION_FUNCTIONS:
                    activations[i] = function
                    model = Model(f"{function}_{i}", activations)
                    model.train(self.data, self.epoch)
                    self.models.append(model)
                    self.write_metrics(file, model, model.calculations)
                self._models_evaluate()
                self.write_metrics(file, "Best Models", self.best_models)
            self._models_evaluate()
            self.best_models[self.best_metric].train(self.data, self.best_epoch)
            self.write_metrics(file, "Best Models", self.best_models)
        return
    
    def write_metrics(self, file, title, data):
        file.write(f"{SEP}\n{title}\n{SEP}\n")
        file.write(f"Precision:     {data['precision']}\n")
        file.write(f"Recall:        {data['recall']}\n")
        file.write(f"Accuracy:      {data['accuracy']}\n")
        file.write(f"Loss:          {data['loss']}\n")
        file.write(f"Average:       {data['average']}\n")
        return
    
    def _models_evaluate(self):
        max_accuracy = self.models[0]
        max_precision = self.models[0]
        max_recall = self.models[0]
        least_loss = self.models[0]
        best_average = self.models[0]
        for model in self.models:
            if model.calculations["average"] > best_average.calculations["average"]:
                best_average = model
            if model.calculations["accuracy"] > max_accuracy.calculations["accuracy"]:
                max_accuracy = model
            if model.calculations["recall"] > max_recall.calculations["recall"]:
                max_recall = model
            if model.calculations["precision"] > max_precision.calculations["precision"]:
                max_precision = model
            if model.calculations["loss"] < least_loss.calculations["loss"]:
                least_loss = model
        self.best_models["precision"] = max_precision
        self.best_models["recall"] = max_recall
        self.best_models["accuracy"] = max_accuracy
        self.best_models["loss"] = least_loss
        self.best_models["average"] = best_average
        return
    
if __name__ == "__main__":
    models = Models(2, 2)
