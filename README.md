
**HOW TO RUN CODE?:

1. Download code.zip and extract files
2. Open main.py and pass argument to run 

JupyterNotebook: !python main.py --model_type X --model_architecture Y --download_data
Local m/c cmd: python main.py --model_type X --model_architecture Y --download_data

- In the above arugumet pass any one model name in X=[EI, SN, TF, JP] and 
  any one model architecture in Y =[BiLSTM, LSTM, BiGRU, GRU]
  Eg: !python main.py --model_type JP --model_architecture BiGRU --download_data
	
NOTE: The above argument does the following:
	1. Downloads the datasets.
	2. Preprocesses the dataset.
	3. Training.
	4. Testing.
	5. Finally displays the test results and saves the graphs in the directorary as .png file.
	
=================================================================================================================================

**DATASET DOWNLOAD LINKS:

'[links]https://www.kaggle.com/datasets/datasnaek/mbti-type'
'[links]https://www.kaggle.com/datasets/kazanova/sentiment140'

=================================================================================================================================

**CONTRIBUTIONS:
_____________________________________________________________________________________
|*Tasks								|				Contributions				    |
|___________________________________|_______________________________________________|
|Background and Literature Survey	|Sung Jun Bok, Deepak Halliyavar, Juyeon Kim	|
|Data Collection					|Deepak Halliyavar, Juyeon Kim					|
|Data Preprocessing					|Sung Jun Bok, Deepak Halliyavar				|
|Data Augmentation					|Sung Jun Bok, uyeon Kim						|
|Modeling							|Sung Jun Bok, Deepak Halliyavar, Juyeon Kim	|
|Experimentation					|Sung Jun Bok, Deepak Halliyavar, Juyeon Kim	|
|Hyperparameter Tuning				|Sung Jun Bok, Deepak Halliyavar, Juyeon Kim	|
|LaTeX documentation				|Sung Jun Bok, Juyeon Kim						|
|Code management and OOPs' dev.		|Deepak Halliyavar								|
|___________________________________|_______________________________________________|