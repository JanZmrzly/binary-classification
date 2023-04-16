import datetime
import tensorflow as tf
import pandas as pd
import numpy as np

class NeuralNetwork():
    def __init__(self):
        self.data_path = None
        self.df = None
        self.data_prepared = False
        
        self.target = "class"
        self.train_df = None
        self.valid_df = None
        self.train_labels = None
        self.valid_labels = None
        self.train_dataset = None
        self.valid_dataset = None

        self.model = None
        self.history = None
        self.model_trained = False
        self.progress = 0
        self.accuracy = 0

    def load_data(self, data_path) -> tuple[str, bool]:
        self.data_path = data_path
        str_res = None
        if self.data_path:
            self.df = pd.read_csv(self.data_path, sep=";")
            str_res = ("[INFO] Data load successfully",True)
            self.data_prepared = False
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
        self.data_prepared = True

    # Converting numpy to tf dataset
    def prepare_dataset(self, data, label, batch, shuffle_buffer):
        dataset = tf.data.Dataset.from_tensor_slices((data, label))
        dataset = dataset.shuffle(shuffle_buffer)
        dataset = dataset.batch(batch).prefetch(1)
        return dataset
    
    def build_model(self, layers:list[tuple[int,str]]):
        if self.data_prepared is False:
            self.prepare_data()
        # Building model
        self.model = tf.keras.Sequential()
        for layer in layers:
            neurons, activation = layer
            self.model.add(tf.keras.layers.Dense(neurons, activation=activation))
    
    def compile_model(self, epochs:int):
        self.progress = 0

        if self.model is not None:
            self.model.compile(
                loss=tf.keras.losses.binary_crossentropy,
                optimizer=tf.keras.optimizers.SGD(),
                metrics=['accuracy', 'mae']
            )
            
            def get_epoch(epoch:int, logs):
                self.progress = round(((epoch+1) / epochs) * 100)

            callbacks = [tf.keras.callbacks.LambdaCallback(on_epoch_end=lambda epoch, logs: 
                                                           get_epoch(epoch, logs))]

            self.history = self.model.fit(
                self.train_dataset,
                epochs=epochs,
                validation_data = self.valid_dataset,
                callbacks=callbacks,
                verbose=0
            )
            self.model_trained = True
   
    def binary_accurancy(self):
        if self.df is not None and self.data_prepared is False:
            self.prepare_data()
        if self.model is not None:
            ba = tf.keras.metrics.BinaryAccuracy()
            evaluate_data = self.train_dataset.concatenate(self.valid_dataset)
            y_test = evaluate_data.map(lambda x, y: y)
            y_test = np.concatenate(list(y_test.as_numpy_iterator()))
            y_pred = self.model.predict(evaluate_data, verbose=0)
            y_pred = np.concatenate(y_pred)
            try:
                ba.update_state(y_test, y_pred)
                self.accuracy = round(ba.result().numpy(), 3)
                return self.accuracy
            except tf.errors.InvalidArgumentError:
                # [TODO]
                pass

    def save_model(self, path):
        dt = datetime.datetime.now()
        time = dt.strftime('%Y%m%d_%H%M')
        if self.history:
            self.model.save(f"{path}/{time}_Model")

    def load_model(self, path):
            try:
                self.model = tf.keras.models.load_model(path)
            except OSError:
                # [TODO]: Wrong folder loaded
                pass
                

if __name__ == "__main__":
    nn = NeuralNetwork()
    epochs = 30
    layers = [
        (11, "relu"),
        (512, "relu"),
        (128, "relu"),
        (2, "tanh")
    ]
    nn.load_data("training_data.csv")
    nn.build_model(layers)
    nn.compile_model(epochs)
    nn.binary_accurancy()