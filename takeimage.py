import cv2
import csv
def TakeImages():
    Id=(txt.get())
    name=(txt2.get())
    
    cam=cv2.VideoCapture(0)
    harcascadePath="haarcascade_frontalface_default.xml"
    detector=cv2.CascadeClassifier(harcascadePath)
    sampleNum=0
    while(True):
        ret,img=cam.read()
        gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        faces=detector.detectMultiScale(gray,1.3,5)
        for(x,y,w,h)in faces:
            cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
#incrementingsamplenumber
            sampleNum=sampleNum+1
#savingthecapturedfaceinthedatasetfolderTrainingImage
            cv2.imwrite("TrainingImage/"+name+"."+Id+'.'+str(sampleNum)+".jpg",
            gray[y:y+h,x:x+w])
#displaytheframe
            cv2.imshow('frame',img)
#waitfor100miliseconds
        if cv2.waitKey(100)&0xFF==ord('q'):
            break
#breakifthesamplenumberismorethan100
        elif sampleNum>60:
            break
    cam.release()
    cv2.destroyAllWindows()
    res="ImagesSavedforID:"+Id+"Name:"+name
    row=[Id,name]
    with open('C:/Users/hp/Desktop/New folder/StudentDetails.csv ','a+') as csv_file:
        writer=csv.writer(csv_file)
        writer.writerow(row)
    csv_file.close()
    message.configure(text=res)
    
                      