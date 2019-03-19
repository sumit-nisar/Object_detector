# -*- coding: utf-8 -*-

import cv2
import numpy as np
import imutils
import time
import os

arguments = {"Input":"videos/pexs.mp4",
             "output":"output/pexs.mp4",
             "yolo":"yolo-coco/",
             "confidence":0.5,
             "threshold":0.3
             } 

labels_path=os.path.sep.join([arguments["yolo"],"coco.names"]) #accessing the file
labels = open(labels_path).read().strip().split("\n") #taking classes from files

np.random.seed(42)

colors = np.random.randint(0,255,size=(len(labels),3),dtype="uint8") #making the different colors for different objects

weights_path = os.path.sep.join([arguments["yolo"],"yolov3.weights"])
config_path = os.path.sep.join([arguments["yolo"],"yolov3.cfg"])

net = cv2.dnn.readNetFromDarknet(config_path,weights_path) #Darknet is backend working like tenserflow for math numeric calculation
layer_names = net.getLayerNames() #for bounding box prediction which is different from cnn,...
layer_names = [layer_names[i[0]-1] for i in net.getUnconnectedOutLayers()] #net.getUnconnectedOutLayers() taking the output from layer_names (see the variable explorer & output of this layer) this is called as listcomprohesion

#new_layer_names=[]
#for i in net.getUnconnectedOutLayers():
#    new_layer_names.append(layer_names[i[0]-1])

writer = None

video_stream = cv2.VideoCapture(arguments["Input"])

while True:
    
    (grabbed, frame) = video_stream.read() #grabbed value will be true and false
    #check if we have reach the end frame
    if not grabbed: #break the grabbed
        break
    
    (h,w) = frame.shape[:2] 
    
    blob = cv2.dnn.blobFromImage(frame,1/255.0,(416,416),
                                 swapRB=True,crop=False) #1/255 is scalling value,calculating mean,(416,416) is resize, swap red and green,center crop
    
    net.setInput(blob)
    start_time = time.time()
    print("start time",start_time)
    layer_outputs = net.forward(layer_names) #net.forward will give value based on layer_names format
    
    end_time = time.time()
    print("TIme taken:",end_time-start_time)
    
    boxes = []
    confidences = []
    class_ids = []
    
    for output in layer_outputs: #diiferent box
        for detection in output: #different objects
            scores = detection[5:]
            class_id = np.argmax(scores) #it will give maximum value of index
            confidence = scores[class_id]
            if confidence > arguments["confidence"]:
                box = detection[0:4] * np.array([w,h,w,h]) #scaling into standardformat of yolo
                (center_x, center_y,width,height)=box.astype("int")
                x = int(center_x - (width/2))
                y = int(center_x - (height/2))
                boxes.append([x,y,int(width),int(height)])
                confidences.append(float(confidence))
                class_ids.append(class_id)
                
    nms_boxes = cv2.dnn.NMSBoxes(boxes,confidences,arguments["confidence"],
                                 arguments["threshold"])
    
    if len(nms_boxes) > 0:
        for i in nms_boxes.flatten(): #flatten is used to  make into single list
            (x,y) = (boxes[i][0],boxes[i][1])
            (w,h) = (boxes[i][2],boxes[i][3])
            
            color = [int(c) for c in colors[class_ids[i]]]
            cv2.rectangle(frame,(x,y),(x+w,y+h),color,2)
            text = "{}: {:.4f}".format(labels[class_ids[i]],confidences[i])
            cv2.putText(frame,text,(x,y-5),cv2.FONT_HERSHEY_SIMPLEX,0.5,
                        color,2)
    
    if writer is None:
        fourcc = cv2.VideoWriter_fourcc(*"MPEG")
        writer = cv2.VideoWriter(arguments["output"],fourcc,30,
                                 (frame.shape[1],frame.shape[0]),True)
        
    writer.write(frame)
    
writer.release()
video_stream.release()

            
            



