# CancerScreeningCNNs
ConvNet Implementations using Keras, for Kaggle Cancer Screening Image Classification Competition

Descriptions of files:

Preprocess.py - preprocess the training and test data samples from kaggle into 224x224x3 numpy arrays.  creates .npy files

train_classifier.py - takes the output of vgg's last conv block to train the last dense layers of for our bottleneck approach to fine tuning VGG

train_top.py - takes the pretrained vgg weights along wiht weights from train_classifier to train entire fine-tuned model (slow learning rate)

scratch_model.py - creates a CNN from scratch with randomly initialized weights, modeled after VGG (less layers)

train_scratch.py - script to train given model for a given number of epochs on the data

ensemble.py - script to train 5 randomly initialized scratch models and ensemble (simple average) their results

summary.py - print summaries of given models

predict.py - generate a csv file with predictions from a model, to be uploaded to Kaggle

vgg.py - first attempts at fine-tuning vgg
