"""
In this module, we define the trainer in TensorFlow.
"""
__title__: str = "TFTrainer"
__version__: str = "1.0.0"
__author__: str = "Brice Petit"
__license__: str = "MIT"

# ----------------------------------------------------------------------------------------------- #
# ------------------------------------------- IMPORTS ------------------------------------------- #
# ----------------------------------------------------------------------------------------------- #

# Imports standard libraries
from typing import NoReturn

# Imports third party libraries
import tensorflow as tf

# Imports from src
from config import DASH_NB, MODELS_DIR
from models import TFSeq2Point, TFUNetNilm
from .base_trainer import BaseTrainer

# Change the default float type
# tf.keras.backend.set_floatx('float64')

# ----------------------------------------------------------------------------------------------- #
# ------------------------------------------- CLASSES ------------------------------------------- #
# ----------------------------------------------------------------------------------------------- #


class TFTrainer(BaseTrainer):
    """
    Class for a TFSeq2Point model. The idea of this class is to create, train, validate and
    test the model. This class will plot the history and the results of the model. It is also
    possible to save and load the model.
    """
    def __init__(
            self, model_name: str, model_type: str, in_length: int, out_length: int,
            learning_rate: float, epochs: int, batch_size: int, patience: int, verbose: int = 1
    ):
        """
        Constructor / Initializer of the Seq2Point class.

        :param model_name:          Name of the model.
        :param model_type:          Type of the model (Seq2Point, UNetNilm or).
        :param in_length:           Length of the input sequence.
        :param out_length:          Length of the output sequence.
        :param learning_rate:       Learning rate.
        :param epochs:              Number of epochs.
        :param batch_size:          Size of the batch.
        :param patience:            Patience for the early stopping.
        :param verbose:             Verbosity of the model.
        """
        super().__init__(
            model_name, model_type, in_length, out_length, learning_rate, epochs, batch_size,
            patience, 'TensorFlow'
        )
        self.verbose = verbose
        if self.model_type == "Seq2Point":
            self.model = TFSeq2Point(self.in_length, self.out_length)
            # self.workers = 2
            self.workers = 4
        elif self.model_type == "UNetNilm":
            self.model = TFUNetNilm(self.in_length, self.out_length)
            # self.workers = 3
            self.workers = 6
        self.model.build((None, self.in_length, 1))
        self.model.compile(
            loss='mse',  # bien en loss function
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
            metrics=[
                tf.metrics.RootMeanSquaredError(),  # bien pour avoir la distance en métrique
                tf.metrics.MeanAbsoluteError(),
            ],
            jit_compile=False
        )
        self.callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=self.patience,
                verbose=0,
                mode="min",
                restore_best_weights=True
            )
        ]
        print(self.model.summary())

    def train(self, train, val=None) -> NoReturn:
        """
        Function to train the model.

        :param train:   Training data generator.
        :param val:     Validation data generator.
        """
        print('-' * DASH_NB, 'Training the model...', '-' * DASH_NB)
        # Init the losses to plot the history
        # Train the model
        history = self.model.fit(
            train,
            batch_size=self.batch_size,
            validation_data=val,
            verbose=self.verbose,
            epochs=self.epochs,
            callbacks=self.callbacks,
            use_multiprocessing=True,
            workers=self.workers
        )
        print('-' * DASH_NB, 'Model trained!', '-' * DASH_NB)
        print('-' * DASH_NB, 'Plotting history...', '-' * DASH_NB)
        self.plot_history(history.history)
        print('-' * DASH_NB, 'History plotted!', '-' * DASH_NB)

    def evaluate(self, test):
        """
        Function to evaluate the model.

        :param test:    Test data generator.

        :return:        Return the evaluation result.
        """
        print('-' * DASH_NB, 'Evaluating the model...', '-' * DASH_NB)
        eval_res = self.model.evaluate(
            test,
            batch_size=self.batch_size,
            verbose=self.verbose,
            use_multiprocessing=True,
            workers=self.workers
        )
        print('-' * DASH_NB, 'Evaluation Done!', '-' * DASH_NB)
        print('-' * DASH_NB, 'Evaluation Result:', '-' * DASH_NB)
        print(f'Loss: {eval_res[0]}, RMSE: {eval_res[1]}, MAE: {eval_res[2]}')
        return eval_res

    def predict(self, test):
        """
        Function to predict with the model, and return the predictions.
        Predictions will be saved in a file, and the results will be plotted.

        :param test:    Test data generator.

        :return:            Return the prediction result.
        """
        print('-' * DASH_NB, 'Predicting the model...', '-' * DASH_NB)
        y_pred = self.model.predict(
            test,
            batch_size=self.batch_size,
            verbose=self.verbose,
            use_multiprocessing=True,
            workers=self.workers
        )
        print('-' * DASH_NB, 'Predictions Done!', '-' * DASH_NB)
        print('-' * DASH_NB, 'Predictions Result:', '-' * DASH_NB)
        self.plot_results(y_pred, test)
        self.save_predictions(y_pred)
        return y_pred

    def save_model(self) -> NoReturn:
        """
        Function to save a model.
        """
        print('-' * DASH_NB, 'Saving the model...', '-' * DASH_NB)
        self.model.save(
            f"{MODELS_DIR}/{self.model_name}_{self.in_length}_{self.out_length}"
            f"_{self.model_type}_TF", save_format='tf'
        )
        print('-' * DASH_NB, 'Model saved!', '-' * DASH_NB)

    def load_model(self) -> NoReturn:
        """
        Function to load a model.
        """
        print('-' * DASH_NB, 'Loading the model...', '-' * DASH_NB)
        self.model = (
            tf.keras.models.load_model(
                f"{MODELS_DIR}/{self.model_name}_{self.in_length}_{self.out_length}"
                f"_{self.model_type}_TF"
            )
        )
        print('-' * DASH_NB, 'Model loaded!', '-' * DASH_NB)
