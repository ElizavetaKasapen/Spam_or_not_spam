from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import ConfusionMatrixDisplay

class ModelEstimator:

    def __init__(self):
        pass

    def confusion_matrix(self, actual, predicted):
        """
        Method to create confusion matrix 
        :param actual: Series, actual values
        :param predicted: Series, predicted values
        :return: ndarray, confusion matrix
        """
        matrix = confusion_matrix(actual, predicted)
        return matrix

    def accuracy (self, actual, predicted):
        """
        Method to create confusion matrix 
        :param actual: Series, actual values
        :param predicted: Series, predicted values
        :return: float, accuracy score
        """
        return accuracy_score(actual, predicted)

    def f1(self, actual, predicted):
        """
        Method to create confusion matrix 
        :param actual: Series, actual values
        :param predicted: Series, predicted values
        :return: float, f1 score
        """
        return f1_score(actual, predicted)

    def recall(self, actual, predicted):
        """
        Method to create confusion matrix 
        :param actual: Series, actual values
        :param predicted: Series, predicted values
        :return: float, recall score
        """
        return recall_score(actual, predicted)

    def precision(self, actual, predicted):
        """
        Method to create confusion matrix 
        :param actual: Series, actual values
        :param predicted: Series, predicted values
        :return: float, precision score
        """
        return precision_score(actual, predicted)
