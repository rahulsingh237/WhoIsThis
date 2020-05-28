import cv2
import numpy as np
import Image_loader as im



def recognizerLogic(name):
    iterator=0
    recog = False
    capture = cv2.VideoCapture(0)
    cascade = cv2.CascadeClassifier('E:\\My Python projects\\FaceLock\\haarcascade_frontalface_default.xml')
    font = cv2.FONT_HERSHEY_COMPLEX
    filename = '%s.npy'%name
    file='E:\\My Python projects\\FaceLock\\Image_files\\'+filename
    name1 = np.load(file).reshape(100,50*50*3)
    users = {0:name}
    labels = np.zeros((200,1))
    labels[:100] = 0.0
    labels[100:] = 1.0
    data = np.concatenate([name1])

    def distance(x1,x2):
        return np.sqrt(sum((x2-x1) ** 2))

    def knn(x,train,k=5):
        n = train.shape[0]
        distances = []
        for i in range(n):
            dist = distance(x,train[i])
            distances.append(dist)
        sortedIndex = np.argsort(distances)
        sortedIndex = sortedIndex[:k]
        nearestNeighbours = []
        for index in sortedIndex:
            nearestNeighbours.append(labels[index])
        count = np.unique(nearestNeighbours,
        return_counts=True)
        label = count[0][np.argmax(count[1])]
        return round(label)


   


    while True:
        ret, image = capture.read()
        if ret:
            faces = cascade.detectMultiScale(image)
            for x,y,w,h in faces:
                cv2.rectangle(image, (x,y),
                            (x+w,y+h), (255,255,255), 5)
                myFace = image[y:y+h,x:x+w,:]
                myFace = cv2.resize(myFace,(50,50))
                label = knn(myFace.flatten(),data)
                if label==0:
                    userName = users[label]
                    cv2.putText(image, userName,
                                (x,y), font, 1, (0,255,0), 2)
                    recog=True
            cv2.imshow('Recognizing User...',image)
            iterator = iterator+1
            if cv2.waitKey(1)    &  iterator>=15:
                break
    capture.release()
    cv2.destroyAllWindows()
    return recog
