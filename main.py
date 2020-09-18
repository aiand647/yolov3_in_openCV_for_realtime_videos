# MIT licence @ aiand647

from cv2 import cv2
import numpy as np
from load_model import load_yolo
from draw_boxes import put_boxes
from boxes import box_list

#setup camera input an window size

CAMERA_PORT = 0
WIDTH = 1280
HEIGHT = 720

WEIGHTS = "/{path to weight file}"
CFG = "/{path to cfg file}"
NAMES = "/{path to coco.names}"

cam = cv2.VideoCapture(CAMERA_PORT)

# Check if the webcam is opened correctly
if not cam.isOpened():
    raise IOError("Cannot open camera")


cam.set(3, WIDTH) # set video width
cam.set(4, HEIGHT) # set video height

if __name__=="__main__":

    classes, output_layers, net = load_yolo(weights_path=WEIGHTS, cfg_path=CFG, names_path=NAMES)

    while True:  
        
        ret, img = cam.read()

        # Detecting objects
        blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(output_layers)

        boxes, confidences, class_ids, class_id = box_list(outs, WIDTH, HEIGHT)

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

        put_boxes(indexes, boxes, classes, class_ids, class_id, img, confidences)

        cv2.imshow("Image", img)
        key = cv2.waitKey(1) & 0xff # Press 'ESC' for exiting video
    
        if key == 27:
            break
    
    cv2.destroyAllWindows()