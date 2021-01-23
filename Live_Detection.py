## This is the Live_Detection part of the project that allows you to be able to use a camera connected to your computer and get the detection live on it.

#WARNING: It would save a lot of time,and generally be more convenient if this is ran all at once in the main RPF.ipynb.
# This is due to RFD module having reloading the 'detection_model1' object just for a single use.

from RFD import run_inference_for_single_image
from RFD import Boxes
from RFD import detection_model1
import cv2
cap = cv2.VideoCapture(0)
Categories = ['Ripe:', 'Unripe:']


while True:
    ret, image_np = cap.read()
    # Actual detection.
    output_dict= run_inference_for_single_image(detection_model1, image_np)

    # Detection_Boxes
    box0 = Boxes(0, image_np)
    box1 = Boxes(1, image_np)
    box2 = Boxes(2, image_np)

    try:
        if output_dict['detection_scores'][0 ]>= 0.5:

            cv2.rectangle(image_np ,(box0.xmin ,box0.ymin) ,(box0.xmax ,box0.ymax) ,box0.colour ,2)
            cv2.rectangle(image_np ,(box0.xmin ,box0.ymin -60) ,(box0.xmax ,box0.ymin) ,box0.colour ,-1)
            cv2.putText(image_np, box0.label, (box0.xmin, box0.ymin -40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255 ,255 ,255), 2)
            cv2.putText(image_np, box0.ripeness, (box0.xmin, box0.ymin -10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255 ,255 ,255), 2)

        if output_dict['detection_scores'][1 ]>= 0.5:

            cv2.rectangle(image_np ,(box1.xmin ,box1.ymin) ,(box1.xmax ,box1.ymax) ,box1.colour ,2)
            cv2.rectangle(image_np ,(box1.xmin ,box1.ymin -60) ,(box1.xmax ,box1.ymin) ,box1.colour ,-1)
            cv2.putText(image_np, box1.label, (box1.xmin, box1.ymin -40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255 ,255 ,255), 2)
            cv2.putText(image_np, box1.ripeness, (box1.xmin, box1.ymin -10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255 ,255 ,255), 2)


        if output_dict['detection_scores'][2 ]>= 0.5:

            cv2.rectangle(image_np ,(box2.xmin ,box2.ymin) ,(box2.xmax ,box2.ymax) ,box2.colour ,2)
            cv2.rectangle(image_np ,(box2.xmin ,box2.ymin -60) ,(box2.xmax ,box2.ymin) ,box2.colour ,-1)
            cv2.putText(image_np, box2.label, (box2.xmin, box2.ymin -40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255 ,255 ,255), 2)
            cv2.putText(image_np, box2.ripeness, (box2.xmin, box2.ymin -10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255 ,255 ,255), 2)
    except:
        pass
    # You can change the resize of the displayed webcam here.
    cv2.imshow('object detection', cv2.resize(image_np, (800 ,600) ,-1))
    if cv2.waitKey(1) & 0xFF == 27:
        cap.release()
        cv2.destroyAllWindows()
        break
