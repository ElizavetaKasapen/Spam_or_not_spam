"""
Code to use models
"""
import pickle
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from imblearn.combine import SMOTETomek 

class BasicModel:
    """
    Basic class to inherit
    """
    def __init__(self):
        pass

    def fit_model(self, features, results, **kwargs):
        """
        Train your model
        :param inputs: list, X
        :param outputs: list, Y
        :param kwargs:
        :return: None
        """
        pass

    def predict(self, features, **kwargs):
        """
        Use trained model
        :param inputs: list, X
        :param kwargs:
        :return:
        """
        pass

"""
Class for NaiveBayesModel 
"""

class  NaiveBayesModel(BasicModel):
    
    def __init__(self):
        BasicModel.__init__(self)
        self.__classifier = MultinomialNB()
        self.__smt = SMOTETomek()

    def fit_model(self, features, results, **kwargs):   
        """
        Method to fit and save model
        :param features: DataFrame, DataFrame of features
        :param results: Series, Series of results
        """
        model = self.__classifier.fit(features, results)
        self._save_model(model)

    def predict(self, features, **kwargs):
        """
        Method to load model and make prediction
        :param features: DataFrame, DataFrame of features
        :return: ndarray, array of results
        """
        model = self._load_model()
        result = model.predict(features)
        return result

    def _save_model(self, model, filename = "data/models/NaiveBayesModel.sav"):  
        """
        Method to save model 
        :param model: MultinomialNB, fitting model
        :param filename: str, path to model
        """
        with open(filename, 'wb') as save_file:
            pickle.dump(model, save_file)
    
    def _load_model(self, filename = "data/models/NaiveBayesModel.sav"):  
        """
        Method to load model 
        :param filename: str, path to model
        :return: MultinomialNB, fitting model
        """
        with open(filename, 'rb') as load_file:
            model = pickle.load(load_file)
            return model

    def train_test_split(self, features, results, test_size = 0.4):
        """
        Method to split dataset to train and test datasets
        :param features: DataFrame, features for prediction
        :param results: Series, result of prediction
        :return: tuple of DataFrames and Series, train and test features and results
        """
        x_train, x_test, y_train, y_test = train_test_split(
            features, results, test_size = test_size, random_state = 0)
        return x_train, x_test, y_train, y_test

    def train_test_valid_split(self, features, results):
        """
        Method to split dataset to train, test and validation datasets
        :param features: DataFrame, features for prediction
        :param results: Series, result of prediction
        :return: tuple of DataFrames and Series, train, test and validation features and results
        """
        x_train, x_test, y_train, y_test = train_test_split(features, results)
        x_test, x_valid, y_test, y_valid = train_test_split(x_test, y_test, test_size = 0.5)
        return x_train, x_test, x_valid, y_train, y_test, y_valid    

    def smote(self, features, results):
        """
        Method to balance classes
        :param features: DataFrame, features for prediction
        :param results: Series, result of prediction
        :return: tuple of ndarrays, features and results with balanced classes
        """
        x_bal, y_bal = self.__smt.fit_resample(features, results)
        return x_bal, y_bal
