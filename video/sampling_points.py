import cv2
import numpy as np 
import os
import pandas as pd 
import json
global p_x
global p_y
p_x = 501#-1
p_y = 167#-1
frame_width=0
frame_height=0
i=0
r,c,d = 5,5,10


filePath = '('+str(int(r*c))+')'+"data.csv"
video_path = "./videos/Trembling-1.mp4"

if os.path.exists(filePath):
    os.remove(filePath)

def mouse_event(event, x,y,flags,param):
    if event==cv2.EVENT_LBUTTONDOWN:
        global p_x, p_y
        p_x = x
        p_y = y
        print("add point", p_x, p_y)
        


def draw(frame):
    global i, total_frames
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if (p_x!=-1 and p_y!=-1):
        cv2.circle(frame, (p_x,p_y), 10, (0,0,255), 3)
        p = frame[p_y,p_x,:]
        p_gray = gray[p_y,p_x].tolist()
        

        
        avg_p_gray = 0
        for sh_y in range(-(r//2)*d, (r//2+1)*d, d):
            for sh_x in range(-(c//2)*d, (c//2+1)*d, d):
                avg_p_gray+=gray[p_y+sh_y, p_x+sh_x]
        p_gray_mean = round(avg_p_gray/(r*c),2)
        print("mean:", p_gray_mean,p_gray)
        p_color = p.tolist()
        p_gray_color = [p_gray,p_gray,p_gray]
        
        cv2.rectangle(frame, (40, 80), (100, 140), p_color, -1)
        cv2.rectangle(frame, (100, 80), (160, 140), p_gray_color, -1)
        
        if start:
            if os.path.exists(filePath):
                df = pd.read_csv(filePath, index_col=0)
                df = df.append({'data': p_gray_mean}, ignore_index=True)
                df.to_csv(filePath)
                print(i, total_frames)
            else:
                df = pd.DataFrame([p_gray], columns=['data'])
                df.to_csv(filePath)
            i+=1
        
       
        cv2.putText(frame, str(p_gray_mean)+", n:"+str(i)+"/"+str(total_frames), (160, 100), cv2.FONT_HERSHEY_SIMPLEX,
  1, (0, 255, 255), 1, cv2.LINE_AA)
        
            
        
    return frame
def save_points():
    return
    


cap = cv2.VideoCapture(video_path)


cv2.namedWindow("image")
cv2.setMouseCallback("image", mouse_event)

start = False
total_frames = 0
while (cap.isOpened()):
    ret, frame = cap.read()
    if not ret:
        break
    total_frames+=1
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
while (cap.isOpened()):
    
    ret, frame = cap.read()
    
    if ret:
        frame_height, frame_width, c = frame.shape
        if (i>total_frames):
            data={}
            data["pos"]={"x":p_x,"y":p_y}
            data["video"]=video_path
            with open('config.json', 'w') as outfile:
                json.dump(data, outfile)
            break
        frame = draw(frame)
        cv2.imshow("image", frame)
        
    else:
       if (p_x!=-1 and p_y!=-1):
           start = True
       print("restart")
       cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
       
    c = cv2.waitKey(1)
    if (c==ord('q')):
        save_points()
        break
cap.release()
cv2.destroyAllWindows()

