# Smile Detection
### By Beyer, Mechtijev, Marte

We have three different applications.

## Requirements for all methods
* python3
* opencv-python
* tensorflow
* pandas
* numpy
* matplot
* sklearn
* glob
* os
* time
## Additional notes
* Depending on your OS (unix / windows), you will have to adjust the paths in the code.
* The images of the dataset are not included in our submission. The positives would need to be copied into the `\dataset\positives\` directory, the negative images into `\dataset\negatives\`.


## Live Smile Detection using Webcam
Detects faces and smiles using your primary webcam. Take a screenshot with `s`, quit the application with `q`.
### How to run
from the root folder
` python3 liveSmileDetection.py`.

## Haar Classifier
Detects smiles from the given dataset. It takes about 20sec (on our machines). It will show the confusion matrix at the end. 
### How to run
from the root folder
` python3 detectSmiles.py`.

## TensorFlow / Keras - Training
A model is already trained (located in the `\model` directory). However if you wish to train it yourself, it can be trained. It might take about 5 minutes to run. A GPU is not necessary but recommended. The dataset splitting is called from the application. It will save a new model in the `\model` directory and it will show two graphs, one with the accuracy over the epochs and one with the loss.
### How to run
from the root folder
` python3 TensorFlow/trainModel.py`.

## TensorFlow / Keras - Prediction
This application will predict, if an image contains a smile or frown. It will show the image with the prediction for each class as a label. 
### How to run
from the root folder
` python3 TensorFlow/predict.py`.

If you wish to predict another image, adjust the path in line `28` to your desired image path.

`predictImage("[imagePath]")`

## Other files
All files in the `\misc` directory are either utility files or miscellaneous files. The cascade classifiers are located in this directory aswell, but will be loaded from the cv2 path.