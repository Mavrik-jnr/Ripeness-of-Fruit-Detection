import numpy as np
# import matplotlib.pyplot as plt
import os
import cv2
import pickle
import random
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report

datadir = r"C:\Users\LG\Desktop\Ripe Fruit Detector"
Categories = ['Ripe', 'Unripe']
training_data = []
test_images = ['Test']
test_data = []
# model_name = input('To Begin Training, Please give the name of the Fruit(.pkl). Please remember to switch me off! : ')
#Functions
def train ():

    for category in Categories:
        path = os.path.join(datadir, category)
        class_num = Categories.index(category)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path, img))
                img_hsv = cv2.cvtColor(img_array, cv2.COLOR_BGR2HSV)
                img_hsv[:,:,1] = 255
                img_array = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)
                new_array = cv2.resize(img_array, (100, 100))
                # here i changed flatten to reshaped (1, -1)
                image = np.array(new_array).flatten()
                # image = np.array(image).reshape(1,-1)
                training_data.append([image, class_num])
                # print(class_num)
            except:
                pass

def test ():
    for category in test_images:
        path = os.path.join(datadir, category)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path, img))

                # img_hsv = cv2.cvtColor(img_array, cv2.COLOR_BGR2HSV)
                # img_hsv[:, :, 1] = 255
                # img_array = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)

                new_array = cv2.resize(img_array, (100, 100))
                # here i changed flatten to reshaped (1, -1)
                image = np.array(new_array).flatten()
                image = np.array(image).reshape(1,-1)

                test_data.append([image])
            except:
                pass

#-------------------------------------------#

#TRAINING DATA PREPARATION

train()
random.shuffle(training_data)
print(len(training_data))
with open('training_data.pkl', 'wb') as f:
    pickle.dump(training_data,f)
print ('Saved Successfully!')



### SPLITTING DATA INTO ATTRIBUTES AND LABELS ###
# with open('training_data.pkl', 'rb') as f:
#     training_data = pickle.load(f)
#
# x = []
# y = []
# for features, label in training_data:
#     x.append(features)
#     y.append(label)
# with open('x.pkl', 'wb') as feat:
#     pickle.dump(x,feat)
# print ('Saved Attributes Successfully!')
#
# with open('y.pkl', 'wb') as lab:
#     pickle.dump(y,lab)
# print ('Saved labels Successfully!')

###### LOADING Training dataset for splitting ####

with open('x.pkl', 'rb')as feat:
    X = pickle.load(feat)
with open('y.pkl ', 'rb') as lab:
    Y = pickle.load(lab)




### TRAINING Time bro ###
xtrain, xtest, ytrain, ytest = train_test_split(X, Y, test_size=0.22)
model_p = SVC(C=1 , kernel = 'linear', gamma='auto')
model = CalibratedClassifierCV(model_p)
model.fit(xtrain, ytrain)
#k nearest implementation
# knn = KNeighborsClassifier(n_neighbors=2)
# knn.fit(xtrain,ytrain)
# accuracy = knn.score(xtest, ytest)
### EVALUATION ###

Accuracy = model.score(xtest, ytest)
predictions = model.predict(xtest)
print(classification_report(ytest, predictions ))

print ('Accuracy: ', Accuracy)
# print ('KNN: ', accuracy)
# if Accuracy>accuracy:
#         print('SVM MASTER-RACE')
# else:
#     print('welp')
### Saving the damn model ###

# with open(model_name, 'wb') as mod:
#     pickle.dump(model, mod)
# print (f'Saved {model_name} Successfully!')


### Testing images to confirm stuff, i.e importing the image directly ###
# img = cv2.imread('IMG_20201218_171647_152.jpg')
# img = cv2.resize(img, (100,100))
# img = np.array(img).reshape(1,-1)

# cv2.imshow('wawu',img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

#### for testing if you have multiple real world datasets ####

# test()
# print(len(test_data))
# with open('real.pkl', 'wb') as t:
#     pickle.dump(test_data,t)
# print ('Saved Successfully!')
# #
# # # -------------------------
# with open(model_name, 'rb') as moda:
#     model = pickle.load(moda)
# with open('Orange Model1.pkl', 'rb') as mod:
#     model1 = pickle.load(mod)
# with open('real.pkl', 'rb') as ril:
#     real = pickle.load(ril)
#
# for p in real:
#     p = np.array(p).reshape(1,-1)
#     # p.reshape(1,-1)
#
#     # print('Saturation model: ', Categories[model1.predict(p)[0]],model1.predict_proba(p) )
#     print('Normal model: ', Categories[model.predict(p)[0]], model.predict_proba(p))
# #
