import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class NeuralNetwork():
    def __init__(self):
        self.data_path = None
        self.df = None
        
        self.target = "class"
        self.train_df = None
        self.valid_df = None
        self.train_labels = None
        self.valid_labels = None
        self.train_dataset = None
        self.valid_dataset = None

        self.model = None
        self.history = None

    def load_data(self, data_path) -> tuple[str, bool]:
        self.data_path = data_path
        str_res = None
        if self.data_path:
            self.df = pd.read_csv(self.data_path, sep=";")
            str_res = ("[INFO] Data load successfully",True)
        else:
            str_res = ("[INFO] No data found",False)
        return str_res
    
    def prepare_data(self):
        # Categorical Columns
        categorical_columns = ["f02","f03","f06","f07","f09","f11"]
        self.df = pd.get_dummies(self.df, columns=categorical_columns)

        # Separating to Train and Valid
        self.train_df = self.df.sample(frac=0.85, random_state=0)
        self.valid_df = self.df.drop(self.train_df.index)

        self.train_labels = self.train_df.pop(self.target).values
        self.valid_labels = self.valid_df.pop(self.target).values

        # Scaling
        stats = self.train_df.describe().transpose()
        self.train_df = (self.train_df - stats['mean']) / stats['std']
        self.valid_df = (self.valid_df - stats['mean']) / stats['std']

        # Converting to numpy
        self.train_data = self.train_df.to_numpy()
        self.valid_data = self.valid_df.to_numpy()

        # Preparing dataset
        batch_size = 32
        buffer = 50
        self.train_dataset = self.prepare_dataset(self.train_data, self.train_labels, batch_size, buffer)
        self.valid_dataset = self.prepare_dataset(self.valid_data, self.valid_labels, batch_size, buffer)

    # Converting numpy to tf dataset
    def prepare_dataset(self, data, label, batch, shuffle_buffer):
        dataset = tf.data.Dataset.from_tensor_slices((data, label))
        dataset = dataset.shuffle(shuffle_buffer)
        dataset = dataset.batch(batch).prefetch(1)
        return dataset
    
    def build_model(self, layers:list[tuple[int,str]]):
        self.prepare_data()
        # Building model
        self.model = tf.keras.Sequential()
        for layer in layers:
            neurons, activation = layer
            self.model.add(tf.keras.layers.Dense(neurons, activation=activation))
    
    def compile_model(self, epochs:int):
        if self.model is not None:
            self.model.compile(
                loss=tf.keras.losses.binary_crossentropy,
                optimizer=tf.keras.optimizers.SGD(),
                metrics=['accuracy', 'mae']
            )

            self.history = self.model.fit(
                self.train_dataset,
                epochs=epochs,
                validation_data = self.valid_dataset,
            )

if __name__ == "__main__":
    nn = NeuralNetwork()
    epochs = 30
    layers = [
        (11, "relu"),
        (512, "relu"),
        (128, "relu"),
        (1, "sigmoid"),
    ]
    nn.load_data("training_data.csv")
    nn.build_model(layers)
    nn.compile_model(epochs)



# # load data
# url = 'https://raw.githubusercontent.com/shravanc/datasets/master/heart.csv'
# # df = pd.read_csv(url)
# df = pd.read_csv("training_data.csv", sep=";")
# print(df.head())

# # TARGET = 'target'
# TARGET = "class"

# # Categorical Columns
# # categorical_columns = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'ca', 'thal']
# categorical_columns = ["f02","f03","f06","f07","f09","f11"]
# df = pd.get_dummies(df, columns=categorical_columns)
# print(df.head())

# # Separating to Train and Valid
# train_df = df.sample(frac=0.85, random_state=0)
# valid_df = df.drop(train_df.index)

# train_labels = train_df.pop(TARGET).values
# valid_labels = valid_df.pop(TARGET).values

# # Scaling
# stats = train_df.describe().transpose()
# train_df = (train_df - stats['mean']) / stats['std']
# valid_df = (valid_df - stats['mean']) / stats['std']
# print(train_df.head())

# # storing data for future prediction
# prediction_data_0 = np.array(valid_df.iloc[0])[np.newaxis]
# result_0 = valid_labels[0]
# prediction_data_1 = np.array(valid_df.iloc[1])[np.newaxis]
# result_1 = valid_labels[1]
# print(result_0, result_1)

# # Converting to numpy
# train_data = train_df.to_numpy()
# valid_data = valid_df.to_numpy()


# # Converting numpy to tf dataset
# def prepare_dataset(data, label, batch, shuffle_buffer):
#     dataset = tf.data.Dataset.from_tensor_slices((data, label))
#     dataset = dataset.shuffle(shuffle_buffer)
#     dataset = dataset.batch(batch).prefetch(1)
#     return dataset


# # Preparing dataset
# batch_size = 32
# buffer = 50
# train_dataset = prepare_dataset(train_data, train_labels, batch_size, buffer)
# valid_dataset = prepare_dataset(valid_data, valid_labels, batch_size, buffer)

# # Building Model
# model = tf.keras.Sequential([
#     tf.keras.layers.Dense(512, activation='relu'),
#     tf.keras.layers.Dense(128, activation='relu'),
#     tf.keras.layers.Dense(1, activation='sigmoid')
# ])

# model.compile(
#     loss=tf.keras.losses.binary_crossentropy,
#     optimizer=tf.keras.optimizers.SGD(),
#     metrics=['accuracy', 'mae']
# )

# history = model.fit(
#     train_dataset,
#     epochs=30,
#     validation_data=valid_dataset
# )

# # Plotting Metrics Curve
# plots = ['accuracy', 'mae', 'loss']
# for plot in plots:
#     metric = history.history[plot]
#     val_metric = history.history[f"val_{plot}"]

#     epochs = range(len(metric))

#     plt.figure(figsize=(15, 10))
#     plt.plot(epochs, metric, label=f"Training {plot}")
#     plt.plot(epochs, val_metric, label=f"Validation {plot}")
#     plt.legend()
#     plt.title(f"Training and Validation for {plot}")
#     plt.show()

# prediction_0 = model.predict(prediction_data_0)
# prediction_1 = model.predict(prediction_data_1)
# print(f"Expected Result: {result_0}, Prediction Result: {prediction_0}")
# print(f"Expected Result: {result_1}, Prediction Result: {prediction_1}")