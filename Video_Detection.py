from RFD import run_inference_for_single_image
from RFD import Boxes
from RFD import detection_model1
import cv2
# This is the Video_Detection part of the project that allows you to be able to pass a video and get the detection on it.
#WARNING: It would save a lot of time,and generally be more convenient if this is ran all at once in the main RPF.ipynb.
# This is due to RFD module having reloading the 'detection_model1' object just for a single use.



#paste name/path to video to be read
cap = cv2.VideoCapture('Unripe.mp4')
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))


# fourcc = cv2.VideoWriter_fourcc(*'MJPG')
#this is the video writer object. set the name of the processed video.
out = cv2.VideoWriter('unrapeban.mp4', cv2.VideoWriter_fourcc('X', 'V', 'I', 'D'), 30, (frame_width, frame_height))
Categories = ['Ripe:', 'Unripe:']

#looping over them frames from the VideoCapture object
while True:
    ret, image = cap.read()
    # Actual detection.
    try:
        output_dict = run_inference_for_single_image(detection_model1, image)
        box0 = Boxes(0, image)
        box1 = Boxes(1, image)
        box2 = Boxes(2, image)
    except:
        pass
#Condition Box0
    if (output_dict['detection_classes'][0] == 5) & (output_dict['detection_scores'][0] >= 0.5):
        image = cv2.rectangle(image, (box0.xmin, box0.ymin), (box0.xmax, box0.ymax), box0.colour, 2)
        image = cv2.rectangle(image, (box0.xmin, box0.ymin - 60), (box0.xmax, box0.ymin), box0.colour, -1)
        image = cv2.putText(image, box0.label, (box0.xmin, box0.ymin - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8,(255, 255, 255), 2)
        image = cv2.putText(image, box0.ripeness, (box0.xmin, box0.ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8,(255, 255, 255), 2)

    if (output_dict['detection_classes'][0] == 3) & (output_dict['detection_scores'][0] >= 0.5):
        image = cv2.rectangle(image, (box0.xmin, box0.ymin), (box0.xmax, box0.ymax), box0.colour, 2)
        image = cv2.rectangle(image, (box0.xmin, box0.ymin - 60), (box0.xmax, box0.ymin), box0.colour, -1)
        image = cv2.putText(image, box0.label, (box0.xmin, box0.ymin - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8,(255, 255, 255), 2)
        image = cv2.putText(image, box0.ripeness, (box0.xmin, box0.ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8,(255, 255, 255), 2)

    if (output_dict['detection_classes'][0] == 2) & (output_dict['detection_scores'][0] >= 0.1):
        image = cv2.rectangle(image, (box0.xmin, box0.ymin), (box0.xmax, box0.ymax), box0.colour, 2)
        image = cv2.rectangle(image, (box0.xmin, box0.ymin - 60), (box0.xmax, box0.ymin), box0.colour, -1)
        image = cv2.putText(image, box0.label, (box0.xmin, box0.ymin - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8,(255, 255, 255), 2)
        image = cv2.putText(image, box0.ripeness, (box0.xmin, box0.ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8,(255, 255, 255), 2)

# Condition for Box1
    if (output_dict['detection_classes'][1] == 5) & (output_dict['detection_scores'][1] >= 0.5):
        image = cv2.rectangle(image, (box1.xmin, box1.ymin), (box1.xmax, box1.ymax), box1.colour, 2)
        image = cv2.rectangle(image, (box1.xmin, box1.ymin - 60), (box1.xmax, box1.ymin), box1.colour, -1)
        image = cv2.putText(image, box1.label, (box1.xmin, box1.ymin - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8,(255, 255, 255), 2)
        image = cv2.putText(image, box1.ripeness, (box1.xmin, box1.ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8,(255, 255, 255), 2)

    if (output_dict['detection_classes'][1] == 3) & (output_dict['detection_scores'][1] >= 0.5):
        image = cv2.rectangle(image, (box1.xmin, box1.ymin), (box1.xmax, box1.ymax), box1.colour, 2)
        image = cv2.rectangle(image, (box1.xmin, box1.ymin - 60), (box1.xmax, box1.ymin), box1.colour, -1)
        image = cv2.putText(image, box1.label, (box1.xmin, box1.ymin - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8,(255, 255, 255), 2)
        image = cv2.putText(image, box1.ripeness, (box1.xmin, box1.ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8,(255, 255, 255), 2)

    if  (output_dict['detection_classes'][1] == 2) & (output_dict['detection_scores'][1] >= 0.6):
        image = cv2.rectangle(image, (box1.xmin, box1.ymin), (box1.xmax, box1.ymax), box1.colour, 2)
        image = cv2.rectangle(image, (box1.xmin, box1.ymin - 60), (box1.xmax, box1.ymin), box1.colour, -1)
        image = cv2.putText(image, box1.label, (box1.xmin, box1.ymin - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8,(255, 255, 255), 2)
        image = cv2.putText(image, box1.ripeness, (box1.xmin, box1.ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8,(255, 255, 255), 2)

#Condition Box2
    if (output_dict['detection_classes'][2] == 5) & (output_dict['detection_scores'][2] >= 0.5):
        image = cv2.rectangle(image, (box2.xmin, box2.ymin), (box2.xmax, box2.ymax), box2.colour, 2)
        image = cv2.rectangle(image, (box2.xmin, box2.ymin - 60), (box2.xmax, box2.ymin), box2.colour, -1)
        image = cv2.putText(image, box2.label, (box2.xmin, box2.ymin - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8,(255, 255, 255), 2)
        image = cv2.putText(image, box2.ripeness, (box2.xmin, box2.ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8,(255, 255, 255), 2)

    if (output_dict['detection_classes'][2] == 3) & (output_dict['detection_scores'][2] >= 0.5):
        image = cv2.rectangle(image, (box2.xmin, box2.ymin), (box2.xmax, box2.ymax), box2.colour, 2)
        image = cv2.rectangle(image, (box2.xmin, box2.ymin - 60), (box2.xmax, box2.ymin), box2.colour, -1)
        image = cv2.putText(image, box2.label, (box2.xmin, box2.ymin - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8,(255, 255, 255), 2)
        image = cv2.putText(image, box2.ripeness, (box2.xmin, box2.ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8,(255, 255, 255), 2)

    if  (output_dict['detection_classes'][2] == 2) & (output_dict['detection_scores'][2] >= 0.6):

        image = cv2.rectangle(image, (box2.xmin, box2.ymin), (box2.xmax, box2.ymax), box2.colour, 2)
        image = cv2.rectangle(image, (box2.xmin, box2.ymin - 60), (box2.xmax, box2.ymin), box2.colour, -1)
        image = cv2.putText(image, box2.label, (box2.xmin, box2.ymin - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8,(255, 255, 255), 2)
        image = cv2.putText(image, box2.ripeness, (box2.xmin, box2.ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8,(255, 255, 255), 2)

#processed frames are then saved
    if (ret == True):
        out.write(image)
        cv2.imshow('object detection', cv2.resize(image, (800, 600), -1))
        if cv2.waitKey(1) & 0xFF == 27:
            cap.release()
            out.release()
            cv2.destroyAllWindows()
    else:
        break
        cap.release()
        cv2.destroyAllWindows()
        out.release()
#         break

out.release()
