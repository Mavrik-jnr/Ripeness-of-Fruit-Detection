#MAIN MODULE CONTAINING FUNCTIONS FOR THE:
#Video_detection.py
#Image_detection.py
#Live_detection.py

###### WARNING: It would save a lot of time,and generally be more convenient if ran at once in the main RFD.ipynb.####
###### This is due to this main RFD module having to reload the 'detection_model1' object just for a single use. #####


import numpy as np
import tensorflow as tf
import cv2


from object_detection.utils import ops as utils_ops


# patch tf1 into `utils.ops`
utils_ops.tf = tf.compat.v1

# Patch the location of gfile
tf.gfile = tf.io.gfile



# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = r'C:\Users\LG\Documents\models\research\object_detection\Images\labelmap.pbtxt'


#The problem.
detection_model1 = tf.saved_model.load(r'C:\Users\LG\Documents\models\research\object_detection\inference_graph1\saved_model')



def run_inference_for_single_image(model, image):
    
    image = np.asarray(image)
  # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
    input_tensor = tf.convert_to_tensor(image)
  # The model expects a batch of images, so add an axis with `tf.newaxis`.
    input_tensor = input_tensor[tf.newaxis,...]
    
  # Run inference
    model_fn = model.signatures['serving_default']
    output_dict = model_fn(input_tensor)

  # All outputs are batches tensors.
  # Convert to numpy arrays, and take index [0] to remove the batch dimension.
  # We're only interested in the first num_detections.
    num_detections = int(output_dict.pop('num_detections'))
    output_dict = {key:value[0, :num_detections].numpy() 
                 for key,value in output_dict.items()}
    output_dict['num_detections'] = num_detections

  # detection_classes should be ints.
    output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)
   
  # Handle models with masks:
    if 'detection_masks' in output_dict:
    # Reframe the the bbox mask to the image size.
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
              output_dict['detection_masks'], output_dict['detection_boxes'],
               image.shape[0], image.shape[1])      
        detection_masks_reframed = tf.cast(detection_masks_reframed > 0.5,
                                       tf.uint8)
        output_dict['detection_masks_reframed'] = detection_masks_reframed.numpy()
    
    return output_dict


# In[9]:


# here the models are loaded. please ensure the models are in the same directory.
#Summary:
#Banana, Training accuracy: 99%, Validation accuracy: 88%
#Pineapple, Training accuracy: 98%, Validation accuracy: 92%
#Orange, Training accuracy: 99%, Validation accuracy: 95%

def decompress_pickle(file):
    import pickle
    import bz2
    import _pickle as cPickle
    data = bz2.BZ2File(file, 'rb')
    data = cPickle.load(data)

    return data

model = decompress_pickle('orange.pbz2')
model0 = decompress_pickle('banana.pbz2')
model1 = decompress_pickle('pineapple.pbz2')


class Boxes():


    def __init__(self, i, image):
        Categories = ['Ripe:', 'Unripe:']
        self.image = image
        shape = list(self.image.shape)
        im_height = int(shape[0])
        im_width =int(shape[1])
        output_dict = run_inference_for_single_image(detection_model1, self.image)
        self.ymin = int(output_dict['detection_boxes'][i][0]*im_height)
        self.xmin = int(output_dict['detection_boxes'][i][1]*im_width)
        self.ymax = int(output_dict['detection_boxes'][i][2]*im_height)
        self.xmax = int(output_dict['detection_boxes'][i][3]*im_width)
        self.roi = self.image[self.ymin:self.ymax,self.xmin:self.xmax]
        self.label = str(output_dict['detection_classes'][i])
        self.ripeness = '9'
        self.colour = 3
        
#         if  (self.label == '1') or(self.label == '2') or (self.label == '4') :
#             pass
        #Orange
        if (self.label == '3'):
            sample = cv2.resize(self.roi, (100, 100))
            sample = sample.flatten()
            sample = np.array(sample).reshape(1,-1)
            self.label = Categories[model.predict(sample)[0]]

            if self.label == 'Ripe:':
                
                self.colour = (0,165,255)
                self.ripeness = str(model.predict_proba(sample)[0][0]*100)[:2]+ '%'
            else:
                self.ripeness = str(model.predict_proba(sample)[0][1]*100)[:2]+ '%'
                self.colour = (32,80,1)
        else:
             pass
#         #Banana
        if  (self.label == '1') or(self.label == '2') or (self.label == '4'):
            sample = cv2.resize(self.roi, (100, 100))
            sample = sample.flatten()
            sample = np.array(sample).reshape(1,-1)
            self.label = Categories[model0.predict(sample)[0]]


            if self.label == 'Ripe:':
                
                self.colour = (0,255,255)
                self.ripeness = str(model0.predict_proba(sample)[0][0]*100)[:2]+ '%'
            else:
                self.ripeness = str(model0.predict_proba(sample)[0][1]*100)[:2]+ '%'
                self.colour = (0,255,0)
#          Pineapple
        if (self.label == '5'):
            sample = cv2.resize(self.roi, (100, 100))
            sample = sample.flatten()
            sample = np.array(sample).reshape(1,-1)
            self.label = Categories[model1.predict(sample)[0]]

            if self.label == 'Ripe:':
                
                self.colour = (0,0,255)
                self.ripeness = str(model1.predict_proba(sample)[0][0]*100)[:2]+ '%'
            else:
                self.ripeness = str(model1.predict_proba(sample)[0][1]*100)[:2]+ '%'
                self.colour = (255,0,0)







