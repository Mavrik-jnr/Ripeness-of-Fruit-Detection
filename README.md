# Ripeness Of Fruit Detection
The project simply detects the fruits previously trained on the Tensorflow Object Detection API and then on the detected ROI, 30 Ensemble Support Vector Classifiers determine the ripeness of the detected fruit's ROI which is then colour coded and expressed as percentages.


In order to use the Detection Scripts, the models have to be in the same directory or the path to the models should be set.
In the training files are the necessary scripts to run and train the classification models for each of the fruits (CLASSIFIER.ipynb).
The Detection model contains the pretrained weights for this project (located in assets and variables), and the trained model itself(saved model.pb) lastly the label map needed to make predictions(labelmap.pbtxt). Five classses were trained and the classes with respect to the labelmap text to int:

Ripe Bunch -'1' (Banana bunch)
Unripe Bunch -'2' (Banana bunch)
Orange - '3'
Banana -'4'
Pineapple -'5'

The classification models file contains the compressed previously trained SVC models. used as the backend and applied to the detected ROI, the classes include 'Ripe' and 'Unripe'


I would appreciate any further training on the models, and i would constantly try to update and fix any detection or software bugs in the future.

