import cv2
import numpy as np 


cap = cv2.VideoCapture('video.mp4')


min_width_rect = 80
min_height_rect = 80

count_line_position = 550

algo = cv2.bgsegm.createBackgroundSubtractorMOG()

def center_handle(x,y,w,h):
    x1 =int(w/2)
    y1 =int(h/2)
    cx = x+x1
    cy = y+y1
    return cx ,cy

detect =[]
offset = 6 
vehicle_count = 0 

while True:
    ret, frame = cap.read()
    grey = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(grey,(3,3),5)
    
    img_sub = algo.apply(blur)
    dilate = cv2.dilate(img_sub, np.ones((5,5)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    dilateada = cv2.morphologyEx(dilate, cv2.MORPH_CLOSE, kernel) 
    dilateada = cv2.morphologyEx(dilateada, cv2.MORPH_CLOSE, kernel) 
    counter , h = cv2.findContours(dilateada, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    cv2.line(frame,(25,count_line_position),(1200,count_line_position),(255,127,0),4)
   
    for (i,c) in enumerate(counter):
        (x,y,w,h) = cv2.boundingRect(c)
        val_counter = (w>=min_width_rect) and (h>=min_height_rect)
        if not val_counter:
            continue
        
        cv2.rectangle(frame,(x,y),(x+w, y+h),(0,255,0),2)
        cv2.putText(frame,"Vehicle:"+str(vehicle_count),(x,y-20),cv2.FONT_HERSHEY_SIMPLEX,1,(255,244,0),2)
        
        
        center = center_handle(x,y,w,h)
        detect.append(center)
        cv2.circle(frame, center , 4,(0,0,255),-1)
        
        for(x,y) in detect:
            if y<(count_line_position + offset) and y>(count_line_position - offset):
                vehicle_count+=1
                cv2.line(frame,(25,count_line_position),(1200,count_line_position),(0,127,255),4)
                detect.remove((x,y))
                print("Vehicle Counter:"+str(vehicle_count))
                
    cv2.putText(frame,"VEHICLE COUNTER:"+str(vehicle_count),(450,70),cv2.FONT_HERSHEY_SIMPLEX,2,(0,0,255),5)        
   
    #cv2.imshow('Detector', dilateada) 
    cv2.imshow('Video Original', frame)
    
    if cv2.waitKey(1) == 13 :
        break

cv2.destroyAllWindows()
cap.release()
    