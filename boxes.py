import cv2
import numpy as np

def box_list(outs, width, height):
    class_ids = []
    confidences = []
    boxes = []
    
    for out in outs:
        for detection in out:
            
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            
            if confidence > 0.5:
                
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    
    return boxes, confidences, class_ids, class_id