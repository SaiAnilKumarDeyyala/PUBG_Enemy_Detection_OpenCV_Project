# Real Time Pubg Enemy Detection Using Opencv
# ssd mobile net configuration has good balance between accuracy and speed over others
import cv2 as cv
from playsound import playsound


cap = cv.VideoCapture(r'C:\Users\Admin\PycharmProjects\PUBGEnemyDetection\PUBG.mp4')
cap.set(3,640)    # setting width
cap.set(4,480)    # setting height #aimodelmarketplace

classNames = []
classFile = 'coco.names'
with open(classFile,'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')             # reading classNames from coco.names

configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = 'frozen_inference_graph.pb'

# DetectionModel allows to set params for preprocessing input image.\
# DetectionModel creates net from file with trained weights and config, sets preprocessing input,\
# runs forward pass and return result detections.
net = cv.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0/127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)


while True:
    success, img = cap.read()
    # we just have to pass the image to net and it will do everything for us
    # and it will return the id of the object that detected and boundary box of it
    # confThreshold -	A threshold used to filter boxes by confidences.
    classIds, confidence, bbox = net.detect(img, confThreshold=0.5)

    if len(classIds) != 0:
        for classId, confidence, box in zip(classIds.flatten(), confidence.flatten(), bbox):
            if classId == 1:
                # playsound('alert.wav')
                cv.rectangle(img, box, color=(255, 0, 0), thickness=2)
                cv.putText(img, classNames[classId-1].upper(), (box[0]+10,box[1]+50), cv.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)

    cv.imshow("output",img)
    cv.waitKey(1)