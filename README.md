# Detection of sleep apnea (Unicam Project)

## Overview
This repository contains the code and data for a deep learning project focused on apnea detection from audio spectrograms. The goal of this project is to train a deep learning model capable of classifying whether a given spectrogram represents an apnea or not.

## Dataset
The dataset used for training and evaluation is collected from the following source: https://www.scidb.cn/en/detail?dataSetId=778740145531650048. The dataset includes two types of files:
1) RML files: contains a list of annotations corresponding to the apnea segments in the edf files.
2) EDF files: contains the audio recordings in EDF format. Each EDF file corresponds to a segment of audio, and its annotations are specified in the corresponding rml file.

## Data Preprocessing
Download the dataset from the provided link and place the url_list.txt file in the root directory of this project.
Execute the data download script (download_prepare_dataset.py) to retrieve the annotation files and corresponding edf files. The script will use the information in rml file to download the required files.
Extract audio segments corresponding to apnea from the EDF files using the audio extraction and cutting script (extract_audio.py). The extracted audio segments will be used to generate spectrograms, but before is applied a noise reduce filter.

These spectrograms will be used as input data for the deep learning model.

## Model
The deep learning model used for this project is based on the VGG19 architecture. The VGG19 model is used as a feature extractor, and additional layers are added to adapt it for the apnea detection task.
The VGG19 base model is used to extract relevant features from the spectrograms, followed by two fully connected layers for classification. The output layer uses the sigmoid activation function to predict the probability of the spectrogram belonging to either an apnea or non-apnea class.

## Model Training
The model is trained using the following configurations:

1) Loss function: Sparse Categorical Crossentropy
2) Optimizer: Adam with a learning rate of 1e-4
3) Metrics: Accuracy
The training process includes early stopping, which monitors the validation accuracy and stops training if there is no improvement for a specified number of epochs.

## Evaluation
After training the model, the model is evaluated with this metrics:
1) Accuracy
2) Loss 
3) Recall
4) F1
5) Precision
