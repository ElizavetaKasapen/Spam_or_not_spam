

#from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import pickle
from processors.tfidf_encoder import TfidfEncoder
"""
Class for data encoders
"""

class DataEncoder:

    def __init__(self):
        self.__vectorizer = TfidfEncoder() 
        self.__label_encoder = {'spam': 1, 'ham': 0}  

    def encode_text_data(self, data):
        """
        Method to transform text series or string to td-idf matrix
        :param data: series of str or str, processed message(s)
        :return: DataFrame, td-idf matrix
        """
        features = self.__vectorizer.fit_transform(data)
        tfidf_df_matrix = self._features_to_df(features)
        return tfidf_df_matrix 

    def _features_to_df(self, trasformed_features):
        """
        Method to transform td-idf matrix to DataFrame
        :param trasformed_features: csr_matrix, features
        :return: DataFrame, td-idf matrix
        """
        vectors_dense = trasformed_features.todense()
        dense_list = vectors_dense.tolist()
        names = self.__vectorizer.get_feature_names_out()
        tfidf_df_matrix = pd.DataFrame(dense_list, columns=names)
        return tfidf_df_matrix 

    def encode_labels(self, series, name = 'spam'):  
        """
        Method to add encoder where 1 = spam, 0 = ham
        :param series: series, category label
        :return: series, zero-one series
        """
        encoded = [self.__label_encoder.get(label) for label in series]
        encoded = pd.Series(encoded, name = name)
        return encoded

    def decode_label(self, label):
        """
        Method to decode label 
        :param label: ndarray, category label
        :return: str, spam or ham
        """
        if label == 0:
            return "ham!"
        return "spam!"

   