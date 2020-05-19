import cv2
import os 
import numpy as np

def main(name):
   capture=cv2.VideoCapture(0)

   cascade=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

   data=[]
   while True:
      ret,image=capture.read()
      if ret:
         gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
         faces=cascade.detectMultiScale(gray,1.15)
         for x,y,w,h in faces:
            cv2.rectangle(image,(x,y),(x+w,y+h),(255,255,255),5)
            myface=image[y:y+h,x:x+w,:]
            myface=cv2.resize(myface,(50,50))
            if len(data)<=100:
               print(len(data))
               data.append(myface)
         
         cv2.imshow("Saving your face",image)
         if cv2.waitKey(1) & 0xff==27 or len(data)>=100:
            break
   arr=np.array(data)
   name='%s.npy'%name
   np.save(os.path.join('Image_files',name),arr)
   capture.release()
   cv2.destroyAllWindows()
