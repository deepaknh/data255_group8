import pandas as pd
from keras.models import Sequential
from keras.layers import Embedding, Dropout, BatchNormalization, Bidirectional, GRU, Dense, LSTM
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from sklearn.metrics import classification_report
from keras.models import load_model
import tensorflow as tf
import os

class BiGRU_model:
    """
    This class defines a BiGRU-based model for binary classification tasks.
    """

    def __init__(self, input_dim, output_dim, input_length):
        """
        Initializes the BiGRU_model with specified input dimension, output dimension, and input length.
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_length = input_length
        self.model = self.create_deep_model()

    def create_deep_model(self):
        """
        Creates and configures a deep BiGRU model for binary classification.
        """
        model = Sequential()
        model.add(Embedding(input_dim=self.input_dim, output_dim=100, input_length=self.input_length))
        model.add(Dropout(0.3))
        model.add(Bidirectional(GRU(32, return_sequences=True, kernel_regularizer=tf.keras.regularizers.l2(0.01))))
        model.add(Dropout(0.3))
        model.add(BatchNormalization())
        model.add(Bidirectional(GRU(32, kernel_regularizer=tf.keras.regularizers.l2(0.01))))
        model.add(Dropout(0.3))
        model.add(BatchNormalization())
        model.add(Dense(self.output_dim, activation='sigmoid'))  # Binary classification
        return model

    def train_and_monitor_model(self, model, X_train, y_train, X_val, y_val, epochs=50, batch_size=64):
        """
        Trains the model and applies callbacks for early stopping and learning rate reduction.
        """
        optimizer = Adam(learning_rate=0.0001)
        model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-5)

        initial_history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping, reduce_lr],
            verbose=1
        )
        return initial_history

    def plot_history(self, history, title):
        """
        Plots the training and validation accuracy and loss graphs.
        """
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title(title + ' Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epochs')
        plt.legend(['Train', 'Validation'], loc='upper left')

        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title(title + ' Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
        plt.tight_layout()

        plt.savefig(title)
        plt.show()

    def eval_and_print_metrics(self, y_pred, y_val):
        """
        Evaluates and prints various performance metrics of the model.
        """
        # Calculate metrics with different averages
        micro_f1 = f1_score(y_val, y_pred, average='micro')
        macro_f1 = f1_score(y_val, y_pred, average='macro')
        weighted_f1 = f1_score(y_val, y_pred, average='weighted')

        micro_precision = precision_score(y_val, y_pred, average='micro')
        macro_precision = precision_score(y_val, y_pred, average='macro')
        weighted_precision = precision_score(y_val, y_pred, average='weighted')

        micro_recall = recall_score(y_val, y_pred, average='micro')
        macro_recall = recall_score(y_val, y_pred, average='macro')
        weighted_recall = recall_score(y_val, y_pred, average='weighted')

        accuracy = accuracy_score(y_val, y_pred)

        # Print metrics
        print(f"Weighted-averaged F1 score: {weighted_f1:.3f}")
        print(f"Weighted-averaged Precision: {weighted_precision:.3f}")
        print(f"Weighted-averaged Recall: {weighted_recall:.3f}")
        print(f"Accuracy: {accuracy:.3f}")
        print("-" * 10)
        print()

    def test_model(self, X_padded_test, y_test, model_type):
        """
        Tests the model on the test data and prints the classification report.
        """
        if self.model is None:
            raise ValueError("Model not trained. Please train the model first.")

        # Make predictions
        predictions = self.model.predict(X_padded_test)

        # Convert probabilities to labels
        if model_type == 'EI':
            # Convert probabilities to labels
            predicted_labels = ['E' if prob < 0.5 else 'I' for prob in predictions]
        elif model_type == 'SN':
            # Convert probabilities to labels
            predicted_labels = ['S' if prob > 0.5 else 'N' for prob in predictions]
        elif model_type == 'TF':
            # Convert probabilities to labels
            predicted_labels = ['T' if prob > 0.5 else 'F' for prob in predictions]
        elif model_type == 'JP':
            # Convert probabilities to labels
            predicted_labels = ['J' if prob < 0.5 else 'P' for prob in predictions]

        # Print classification report
        report = classification_report(y_test, predicted_labels)
        print(f"TEST Classification Report: {model_type}")
        print(report)

        # Evaluate and print metrics
        self.eval_and_print_metrics(predicted_labels, y_test)

