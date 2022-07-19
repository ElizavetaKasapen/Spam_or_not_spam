"""
Code to process texts
"""


import pandas as pd
#to del punctuation marks
import re
#for Lemmatizer and stopwords
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer 


class TextProcessor:  

    def __init__(self):
        self.__reg_exp_to_del_punctuation = "[^a-zA-Z]"
        self.__lm = WordNetLemmatizer()
        self._text_processors = [self._del_punctuation, self._to_words_list,
                                self._del_stopwords, self._lemmatizing, ' '.join]
        self._load_nltk_data()
    
    def _load_nltk_data(self):
        """
        Method to load nltk data
        """       
        nltk.download('stopwords')
        nltk.download('wordnet')

    def _del_punctuation(self,text):  
        """
        Method to delete punctuation marks in string
        :param text: str, message
        :return: str, text without punctuation
        """
        clear_text = re.sub(self.__reg_exp_to_del_punctuation, ' ', text)
        return clear_text

    def _to_words_list(self, text):  
        """
        Method to convert string to words list in lower case
        :param text: str, message
        :return: str, list of words
        """
        words_list = text.lower().split()
        return words_list

    def _del_stopwords(self, words_list):  
        """
        Method to delete stopwords
        :param words_list: list, list of words
        :return: list, list without stopwords
        """
        filtered_list = [word for word in words_list if word not in stopwords.words('english')]
        return filtered_list

    def _lemmatizing(self, words_list):  
        """
        Method to delete stopwords
        :param words_list: list, list of words
        :return: list, list without stopwords
        """
        lemmatized_words = [self.__lm.lemmatize(word) for word in words_list]
        return lemmatized_words

    def process_text(self, text):  #I want to use it for data processing in prediction()
        """
        Method to process text
        :param text: str, message
        :return: str, processed message
        """
        for processor in self._text_processors:
            text = processor(text)
        return text

    def process_text_series(self, series, name = 'prepared_messages'):
        """
        Method to process text series
        :param series: series of str, messages
        :return: Series, processed messages
        """
        list_of_prepared_messages = [self.process_text(message) for message in series]
        prepared_messages_series = pd.Series(list_of_prepared_messages, name = name)
        return prepared_messages_series