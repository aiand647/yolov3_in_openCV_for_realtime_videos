from cv2 import cv2
import numpy as np



def put_boxes(indexes, boxes, classes, class_ids, class_id, img, confidences):
    
    font=cv2.FONT_HERSHEY_SIMPLEX
    colors = np.random.uniform(0, 255, size=(len(classes), 3))
    for i in range(len(boxes)):
       
        if i in indexes:
           
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            color = colors[i]
            
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(img, label, (x+5,y-5), font, 1, color, 3)
            cv2.putText(img, str(round(confidences[class_id]*100,2))+"%", (x+5,y-37), font, 1, color, 3)