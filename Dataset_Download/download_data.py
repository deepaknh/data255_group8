import gdown

class Download_Data:
    """
    Class to download data from their respective locations
    """
    def download_training_data(self):
        """
        Method to download the training dataset
        """
        output_file = 'TRAINING'
        url = 'https://drive.google.com/uc?id=1aMiL-Jg_3OdaSbAkHTYz_3D0wBSR23Sa'
        gdown.download(url, output_file + ".csv", quiet=False)

    def download_testing_data(self):
        """
        Method to download the testing dataset
        """
        output_file = 'TESTING'
        url = 'https://drive.google.com/uc?id=1HygEBiHwTgvUH42z8H4CAwkN84Wf4_bW'
        gdown.download(url, output_file + ".csv", quiet=False)