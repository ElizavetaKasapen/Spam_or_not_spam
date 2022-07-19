"""
Interface class
"""

from processors.text_processor import TextProcessor
from processors.encoders import DataEncoder
from models.models import NaiveBayesModel
from models_estimation.models_estimation import ModelEstimator
from data_loader import DataLoader
from datetime import datetime

class ModelInterface:

    def __init__(self, data_config, model_config):
        self.__path = data_config["path"]
        self.__label_column = data_config["label_column"]
        self.__message_column = data_config["message_column"]
        self.__data_loader = DataLoader()
        self.__txt_processor = TextProcessor()
        self.__encoder = DataEncoder()
        self.__model = model_config["model"]()  
        self.__estimation = ModelEstimator()  

    def _get_encoded_data(self): 
        """
        Method to get and process data
        :return: tuple of DataFrame and Series, features and results 
        """
        data_frame = self.__data_loader.download_csv(self.__path)  
        processed_messages = self.__txt_processor.process_text_series(data_frame[self.__message_column])
        encoded_labels = self.__encoder.encode_labels(data_frame[self.__label_column])
        encoded_features = self.__encoder.encode_text_data(processed_messages)
        return encoded_features, encoded_labels 

    def train(self): 
        """
        Train method
        """
        encoded_features, encoded_results = self._get_encoded_data()
        x_train, x_test, y_train, y_test = self.__model.train_test_split(encoded_features, encoded_results)
        x_train_balanced, y_train_balanced = self.__model.smote(x_train, y_train)
        self.__model.fit_model(x_train_balanced, y_train_balanced)
        prediction = self.__model.predict(x_test)
        self.save_model_estimation(prediction, y_test, "NaiveBayesModel")

    def predict(self, text):
        """
        Method for class prediction
        :param text: str, text for analysis
        :return: str, spam or ham
        """
        processed_text = self.__txt_processor.process_text(text)
        encode_text = self.__encoder.encode_text_data(processed_text)
        prediction = self.__model.predict(encode_text)
        return self.__encoder.decode_label(prediction)

    def _get_estimation_scores(self, prediction, y_test):
        """
        Method to calculate estimation scores
        :param prediction: ndarray, array of predicted values
        :param y_test: Series, actual values
        :return: tuple, tuple of scores in string format
        """
        accuracy = str(self.__estimation.accuracy(y_test,prediction))
        recall = str(self.__estimation.recall(y_test,prediction))
        precision = str(self.__estimation.precision(y_test,prediction))
        f1 = str(self.__estimation.f1(y_test,prediction))
        confusion_matrix = self.__estimation.confusion_matrix(y_test,prediction)
        represent_confusion_matrix = "true negatives: {}\nfalse negatives: {}\ntrue positives: {}\nfalse positives: {}".format(confusion_matrix[0][0], 
                                    confusion_matrix[1][0], confusion_matrix[1][1], confusion_matrix[0][1])
        return accuracy, recall, precision, f1, represent_confusion_matrix

    def save_model_estimation(self, prediction, y_test, model_name):
        """
        Method to get and save estimation scores in txt file
        :param prediction: ndarray, array of predicted values
        :param y_test: Series, actual values
        param model_name: str, model name
        """       
        accuracy, recall, precision, f1, represent_confusion_matrix = self._get_estimation_scores(prediction, y_test)
        current_datetime = datetime.now()
        time = current_datetime.strftime("%d-%m-%y_%H:%M")
        path = "data/models_estimation/{0}_{1}.txt".format(model_name, time)
        text_to_save = "{0}\n\nAccuracy: {1}\nRecall: {2}\nPrecision: {3}\nF1: {4}\nConfusion matrix: \n{5}".format(model_name,
                        accuracy, recall, precision, f1, represent_confusion_matrix)
        with open(path, 'w') as file:
            file.write(text_to_save)

if __name__ == '__main__':
    ham = "Hey! How are you? How is your family? What about your babies?"
    spam = "U have won the £750 Pound prize!"
    data_config = dict(path = "data/csv_data/spam_or_not_spam_data.csv", label_column = "Category", message_column = "Message")
    model_config = dict(model = NaiveBayesModel)
    interface = ModelInterface(data_config, model_config)  
    interface.train()
    result = interface.predict(ham)
    print(result)
    result = interface.predict(spam)
    print(result)
