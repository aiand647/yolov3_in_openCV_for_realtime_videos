# Yolov3 in openCV for realtime video
Automated localization and classification of multiple object fed directly from the webcam without GPU.

Note: hence it use only cpu the fps noted while using yolov3-tiny is useable wheres original version draws to much computaional resources and fps drops to 2-5 making it unusable in real time

Feel free to twick the code and optimize as per requierment

please download yolov3-tiny.weights from the following link:-
https://pjreddie.com/media/files/yolov3-tiny.weights

# Requierments
- Numpy
- openCV 2.0

# How to Use
- Clone the repository
- Download the weights
- Place it in model folder
- open **main.py** in a text editor and specify the paths of weight, cfg, names file
- let's run 

```
python3 main.py

```

# Samples

![Cars on Road](https://raw.githubusercontent.com/aiand647/yolov3_in_openCV_for_realtime_videos/master/samples/cars.png)
![Peoples](https://raw.githubusercontent.com/aiand647/yolov3_in_openCV_for_realtime_videos/master/samples/people.png)
![More Peoples](https://raw.githubusercontent.com/aiand647/yolov3_in_openCV_for_realtime_videos/master/samples/people2.png)

# Referance
- https://pjreddie.com/darknet/yolo/
