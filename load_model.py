from cv2 import cv2
import numpy as np

#The method 

def load_yolo(weights_path="/home/ayand/Projects/facedetect/YOLO/yolov3.weights", cfg_path= "/home/ayand/Projects/facedetect/YOLO/yolov3.cfg", names_path="/home/ayand/Projects/facedetect/YOLO/coco.names"):
    net = cv2.dnn.readNet(weights_path, cfg_path)
    classes = []
    
    with open(names_path, "r") as f:
        classes = [line.strip() for line in f.readlines()]
    
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    
    return classes, output_layers, net
