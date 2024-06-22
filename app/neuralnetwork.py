import optuna
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import backend as K

@tf.keras.utils.register_keras_serializable()
def recall_positive_class(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

class NeuralNetwork:
    def __init__(self, input_shape, num_classes, learning_rate, num_layers, units, activations):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.num_layers = num_layers
        self.units = units
        self.activations = activations
        self.model = None
    
    def build_model(self):
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Dense(self.units[0], activation=self.activations[0], input_shape=(self.input_shape,)))
        
        for i in range(1, self.num_layers):
            model.add(tf.keras.layers.Dense(self.units[i], activation=self.activations[i]))
        
        model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
        
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=[recall_positive_class])
        
        self.model = model

    def train(self, X_train, y_train, X_valid, y_valid, epochs, batch_size):
        history = self.model.fit(X_train, y_train, validation_data=(X_valid, y_valid), epochs=epochs, batch_size=batch_size)
        return history.history['val_recall_positive_class'][-1]

    def evaluate(self, X_test, y_test):
        loss, recall = self.model.evaluate(X_test, y_test)
        print(f"Test recall (class 1): {recall:.4f}")
        return recall

    def predict(self, X_new):
        predictions = self.model.predict(X_new)
        return predictions
    
    def predict_ones_and_zeros(self, rows):
        predictions_c = self.predict(rows)
        predicted_classes_c = (predictions_c > 0.5).astype('int32')
        return predicted_classes_c