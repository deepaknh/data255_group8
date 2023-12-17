# Personality Trait Prediction Using Text From Social Media

Personality classification from social media has gained attention and has been implemented by various neural networks. However, numerous papers mention the limitation of the amount of labeled data. To address this issue, this project will employ semi-supervised learning using a stochastic gradient descent classifier. Stochastic gradient descent classifier will initially process a small dataset with MBTI labels and then assign labels on unlabeled text data from Twitter. For this project, four models will be used: LSTM, GRU, BILSTM, and BiGRU. The models will be trained with the augmented data, evaluated, and compared for its effectiveness in classifying MBTI personality types. In this project’s final examinations, the BiLSTM model achieved the best accuracy results of 0.85, 0.89, 0.79 and 0.765 for the E/I, S/N, T/F, J/P dimensions, respectively.

**HOW TO RUN CODE?**
1. Download code.zip and extract files
2. Open main.py and pass argument to run (!python main.py --model_type X --model_architecture Y --download_data)

In the above arugumet pass any one model name in X=[EI, SN, TF, JP] and 
  any one model architecture in Y =[BiLSTM, LSTM, BiGRU, GRU]
  Eg: !python main.py --model_type JP --model_architecture BiGRU --download_data
	
The above argument does the following:
1. Downloads the datasets.
2. Preprocesses the dataset.
3. Training.
4. Testing.
5. Finally displays the test results and saves the graphs in the directorary as .png file.
=====================================================================================
**DATASET DOWNLOAD LINKS:

'https://www.kaggle.com/datasets/datasnaek/mbti-type'

'https://www.kaggle.com/datasets/kazanova/sentiment140'

=====================================================================================
___________________________________________________________________________________________________________________________________
