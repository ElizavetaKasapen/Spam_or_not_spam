import pandas as pd

class DataLoader:

    def __init__(self):
        pass

    def download_csv(self, path):
        """
        Method to download csv file
        :param path: str, path to file
        :return: DataFrame, data from file
        """
        try:
            dataset = pd.read_csv(path)
            return dataset
        except IOError:
            print("Unreachable file!")   