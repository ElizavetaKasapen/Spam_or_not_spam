

from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

class TfidfEncoder:

    def __init__(self, path = "data/models/text_encoder.sav"):
        self.__path = path
        self.__vectorizer, self.__pretrained = self._get_encoder(self.__path)

    def _get_encoder(self, path):
        """
        Method to get encoder 
        :param path: str, path to encoder
        :return: tuple(TfidfVectorizer, boolean) td-idf vectorizer and model status (true if model pretrained)
        """
        try:
            with open(path, 'rb') as load_file:
                vectorizer = pickle.load(load_file)
                pretrained = True
        except IOError:
            vectorizer = TfidfVectorizer() 
            pretrained = False
        return vectorizer, pretrained
        
    def _save_text_encoder(self, encoder, path):  
        """
        Method to save text encoder 
        :param encoder: TfidfVectorizer, td-idf vectorizer
        :param path: str, path to encoder
        """
        with open(path, 'wb') as save_file:
            pickle.dump(encoder, save_file)
        self.__pretrained = True

    def fit_transform(self, data):
        """
        Method to fit and transform data 
        :param data: Series or str, data to transform
        :return: csr_matrix, features
        """
        if self.__pretrained:   
            if isinstance(data, str):
                data = [data]
            features = self.__vectorizer.transform(data)
        else:
            features = self.__vectorizer.fit_transform(data)
            self._save_text_encoder(self.__vectorizer, self.__path)
        return features
    
    def get_feature_names_out(self):
        """
        Method to get feature names
        :return: ndarray, names of features
        """
        return self.__vectorizer.get_feature_names_out()
    