import tkinter as tk
from tkinter import Message,Text
import cv2,os
import shutil
import csv
import numpy as np
from PIL import Image,ImageTk
import pandas as pd
import datetime
import time
import tkinter.ttk as ttk
import tkinter.font as font
import unicodedata
window=tk.Tk()
#helv36=tk.Font(family='Helvetica',size=36,weight='bold')
window.title("Face_Recogniser")
dialog_title='QUIT'
dialog_text='Areyousure?'
#answer=messagebox.askquestion(dialog_title,dialog_text)
#window.geometry('1280x720')
window.configure(background='blue')
window.grid_rowconfigure(0,weight=1)
window.grid_columnconfigure(0,weight=1)
#path="profile.jpg"
#CreatesaTkinter-compatiblephotoimage,whichcanbeusedeverywhereTkinterexpects

#img=ImageTk.PhotoImage(Image.open(path))
#TheLabelwidgetisastandardTkinterwidgetusedtodisplayatextorimageonthescreen.
#panel=tk.Label(window,image=img)
#panel.pack(side="left",fill="y",expand="no")
#cv_img=cv2.imread("img541.jpg")
#x,y,no_channels=cv_img.shape
message=tk.Label(window,text="Face-Recognition-Based-Attendance-Management System",bg="Green",fg="white",width=50,height=3,font=('times',30,'italic'))
message.place(x=200,y=20)
lbl=tk.Label(window,text="EnterID",width=20,height=2,fg="red",bg="yellow",font=('times',15,'bold'))
lbl.place(x=400,y=200)
txt=tk.Entry(window,width=20,bg="yellow",fg="red",font=('times',15,'bold'))
txt.place(x=700,y=215)
lbl2=tk.Label(window,text="EnterName",width=20,fg="red",bg="yellow" ,height=2,font=('times',15,'bold'))
lbl2.place(x=400,y=300)
txt2=tk.Entry(window,width=20,bg="yellow",fg="red",font=('times',15,'bold'))
txt2.place(x=700,y=315)
lbl3=tk.Label(window,text="Notification:",width=20,fg="red",bg="yellow",height=2,font=('times',15,'bold'))
lbl3.place(x=400,y=400)
message=tk.Label(window,text="",bg="yellow",fg="red",width=30,height=2,activebackground="yellow",font=('times',15,'bold'))
message.place(x=700,y=400)

lbl3=tk.Label(window,text="Attendance:",width=20,fg="red",bg="yellow",height=2,font=('times',15,'bold'))
lbl3.place(x=400,y=650)

message2=tk.Label(window,text="",fg="red",bg="yellow",activeforeground="green",width=30,height=2,font=('times',15,'bold'))
message2.place(x=700,y=650)

def clear():

    txt.delete(0,'end')
    res=""
    message.configure(text=res)
def clear2():
    txt2.delete(0,'end')
    res=""
    message.configure(text=res)
def is_number(s):
    try:
        float(s)
        return  True
    except ValueError:
          pass
    #unicodedata.numeric(s)


Id=(txt.get())
name=(txt2.get())
#if (is_number(Id)and name.isalpha()):
cam=cv2.VideoCapture(0)
harcascadePath="C:/Users/hp/Desktop/New folder/haarcascade_frontalface_default.xml"
detector=cv2.CascadeClassifier("C:/Users/hp/Desktop/New folder/haarcascade_frontalface_default.xml")
sampleNum=0
while(True):
    ret,img= cam.read()
    #gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY )
    faces = detector.detectMultiScale(gray,1.3,5)
    for(x,y,w,h)in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
#incrementingsamplenumber
        sampleNum=sampleNum+1
        cv2.imwrite("TrainingImage/"+name+"."+Id+'.'+str(sampleNum)+".jpg",gray[y:y+h,x:x+w])
        cv2.imshow('frame',img)
    if cv2.waitKey(100)&0xFF==ord('q'):
        break
#breakifthesamplenumberismorethan100
    elif sampleNum>60:
        break
cam.release()
cv2.destroyAllWindows()
res="ImagesSavedforID:"+Id+"Name:"+name
row=[Id,name]
with open('C:/Users/hp/Desktop/New folder/StudentDetails.csv','a+') as csvFile:
    writer=csv.writer(csvFile)
    writer.writerow(row)

    csvFile.close()
    message.configure(text=res)

if(is_number(Id)):
    res="EnterAlphabeticalName"
    message.configure(text=res)
if(name.isalpha()):
    res="EnterNumericId"
    message.configure(text=res)

#def TrainImages():
def getImagesAndLabels(path):
#getthepathofallthefilesinthefolder
    imagePaths=[os.path.join(path,f) for f in os.listdir(path)]
#print(imagePaths)
#createempthfacelis

    faces=[]
#createemptyIDlist
    Ids=[]
    for imagePath in imagePaths:
#loadingtheimageandconvertingittograyscale
        pilImage=Image.open(imagePath).convert('L')
#NowweareconvertingthePILimageintonumpyarray
        imageNp=np.array(pilImage,'uint8')
#gettingtheIdfromtheimage
        Id=int(os.path.split(imagePath)[-1].split(".")[1])
#extractthefacefromthetrainingimagesample
        faces.append(imageNp)
        Ids.append(Id)
    return faces,Ids
recognizer=cv2.face.createLBPHFaceRecognizer()#recognizer=
cv2.faces.LBPHFaceRecognizer_create()#$cv2.createLBPHFaceRecognizer()
harcascadePath="C:/Users/hp/Desktop/New folder/haarcascade_frontalface_default.xml"
detector=cv2.CascadeClassifier(harcascadePath)
faces,Id=getImagesAndLabels("TrainingImage")
recognizer.train(faces,np.array(Id))
recognizer.save("TrainingImageLabel/Trainner.yml")
res="ImageTrained"#+",".join(str(f)forfinId)
message.configure(text=res)


def TrackImages():
    recognizer=cv2.face.createLBPHFaceRecognizer()#cv2.createLBPHFaceRecognizer()
    recognizer.load("TrainingImageLabel/Trainner.yml")
    harcascadePath="haarcascade_frontalface_default.xml"
    faceCascade=cv2.CascadeClassifier(harcascadePath);
    df=pd.read_csv("StudentDetails/StudentDetails.csv")
    cam=cv2.VideoCapture(0)
    font=cv2.FONT_HERSHEY_SIMPLEX
    col_names=['Id','Name','Date','Time']
    attendance=pd.DataFrame(columns=col_names)
    while True:
        ret,im=cam.read()
        gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
        faces=faceCascade.detectMultiScale(gray,1.2,5)
        for(x,y,w,h)    in  faces:
            cv2.rectangle(im,(x,y),(x+w,y+h),(225,0,0),2)
            Id,conf=recognizer.predict(gray[y:y+h,x:x+w])  
            if(conf<50):
                ts=time.time()
                date=datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
                timeStamp=datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
                aa=df.loc[df['Id']==Id]['Name'].values
                tt=str(Id)+"-"+aa
                attendance.loc[len(attendance)]=[Id,aa,date,timeStamp  ]
            else:
                Id='Unknown'
                tt=str(Id)
            if(conf>75):
                noOfFile=len(os.listdir("ImagesUnknown"))+1
                cv2.imwrite("ImagesUnknown/Image"+str(noOfFile)+".jpg",im[y:y+h,x:x+w])
                cv2.putText(im,str(tt),(x,y+h),font,1,(255,255,255),2)
            attendance=attendance.drop_duplicates(subset=['Id'],keep='first')
            cv2.imshow('im',im)
            if(cv2.waitKey(1)==ord('q')):
                break
            ts=time.time()
            date=datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
            timeStamp=datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
            Hour,Minute,Second=timeStamp.split(":") 
            fileName="Attendance/Attendance_"+date+"_"+Hour+"-"+Minute+"-"+Second+".csv"
            attendance.to_csv(fileName,index=False)
            cam.release()
            cv2.destroyAllWindows()
    #print(attendance)
            res=attendance
            message2.configure(text=res)
clearButton=tk.Button(window,text="Clear",command=clear,fg="red",bg="yellow",width=20,height=2,activebackground=
"Red",font=('times',15,'bold'))
clearButton.place(x=950,y=200)
clearButton2=tk.Button(window,text="Clear",command=clear2,fg="red",bg="yellow",width=20,height=2,activebackground=
"Red",font=('times',15,'bold'))
clearButton2.place(x=950,y=300)
takeImg=tk.Button(window,text="TakeImages",command="TakeImages",fg="red",bg="yellow",width=20,height=3,activebackground=
"Red",font=('times',15,'bold'))
takeImg.place(x=200,y=500)
trainImg=tk.Button(window,text="TrainImages",command="TrainImages",fg="red",bg="yellow",width=20,height=3,activebackground="Red",font=('times',15,'bold'))
trainImg.place(x=500,y=500)
trackImg=tk.Button(window,text="TrackImages",command=TrackImages,fg="red",bg="yellow",width=20,height=3,activebackground=
"Red",font=('times',15,'bold'))
trackImg.place(x=800,y=500)
quitWindow=tk.Button(window,text="Quit",command=window.destroy,fg="red",bg="yellow",width=20,height=3,activebackground=
"Red",font=('times',15,'bold'))
quitWindow.place(x=1100,y=500)
copyWrite=tk.Text(window,background=window.cget("background"),borderwidth=0,font=('times',30,'italic'))
copyWrite.tag_configure("superscript",offset=10)
copyWrite.insert("insert","Developedbykusum","","TEAM","superscript")
copyWrite.configure(state="disabled",fg="red")
copyWrite.pack(side="left")
copyWrite.place(x=800,y=750)
window.mainloop()

