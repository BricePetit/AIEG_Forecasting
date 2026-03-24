"""
In this module, we define the trainer in PyTorch.
"""
__title__: str = "sklearn_trainer"
__version__: str = "1.0.0"
__author__: str = "Brice Petit"
__license__: str = "MIT"

# ----------------------------------------------------------------------------------------------- #
# ------------------------------------------- IMPORTS ------------------------------------------- #
# ----------------------------------------------------------------------------------------------- #

# Imports standard libraries
import pickle
from typing import NoReturn

# Imports third party libraries
from sklearn.ensemble import HistGradientBoostingRegressor  


# Imports from src
from .base_trainer import BaseTrainer
from config import DASH_NB, MODELS_DIR

# ----------------------------------------------------------------------------------------------- #
# ------------------------------------------- CLASSES ------------------------------------------- #
# ----------------------------------------------------------------------------------------------- #


class SkLearnTrainer(BaseTrainer):
    """
    Class for a PyTorchSeq2Models model. The idea of this class is to create, train, validate and
    test the model. This class will plot the history and the results of the model. It is also
    possible to save and load the model.
    According to the model that we want, it will be possible to create a Seq2Point model or a
    Seq2Seq model.
    """
    def __init__(
            self, model_name: str, model_type: str, in_length: int, out_length: int, params: dict,
            learning_rate: float, epochs: int, batch_size: int, patience: int, shuffle: bool = False
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
        :param shuffle:             Boolean to shuffle the indices.
        """
        super().__init__(
            model_name, model_type, in_length, out_length, learning_rate, epochs, batch_size,
            patience, 'SkLearn'
        )
        self.model = HistGradientBoostingRegressor(**params)
        self.metrics = None
        self.shuffle = shuffle

    def train(self, train, val=None, example_seen=0):
        """
        Function to train the model.

        :param train:           Data for the training set.
        :param val:             Data for the validation set.
        :param example_seen:    Number of examples seen.
        """
        pass

    def evaluate(self, test):
        """
        Function to evaluate the model.

        :param test:    Test data.

        :return:        Return the evaluation result.
        """
        pass

    def predict(self, test):
        """
        Function to predict with the model, and return the predictions.
        Predictions will be saved in a file, and the results will be plotted.

        :param test:    Test data generator.

        :return:        Return the prediction result.
        """
        pass

    def save_model(self) -> NoReturn:
        """
        Function to save the model.
        """
        print('-' * DASH_NB, 'Saving the model...', '-' * DASH_NB)
        pickle.dump(self.model, open(
            f"{MODELS_DIR}/{self.model_name}_{self.in_length}_{self.out_length}"
            f"_{self.model_type}_SkLearn.sav", 'wb'
        ))
        print('-' * DASH_NB, 'Model saved!', '-' * DASH_NB)

    def load_model(self) -> NoReturn:
        """
        Function to load the model.
        :return:
        """
        print('-' * DASH_NB, 'Loading the model...', '-' * DASH_NB)
        self.model = pickle.load(open(
            f"{MODELS_DIR}/{self.model_name}_{self.in_length}_{self.out_length}"
            f"_{self.model_type}_SkLearn.sav", 'rb'
        ))
        print('-' * DASH_NB, 'Model loaded!', '-' * DASH_NB)
