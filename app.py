import tkinter as tk
import customtkinter  as ctk
import torch
import numpy as np
import cv2
from PIL import Image,ImageTk

import os
os.add_dll_directory('C:/Program Files/VideoLAN/VLC')

import vlc

app=tk.Tk()
app.geometry("600x600")
app.title("Drowsiness_detection")
ctk.set_appearance_mode("dark")

vidFrame=tk.Frame(height=400,width=600)
vidFrame.pack()
vid=ctk.CTkLabel(vidFrame)
vid.pack()

counter=0
counterLabel=ctk.CTkLabel(master=app,font=("Montserrat", 16),height=40,width=120,text=counter,text_color="white",fg_color="teal")
counterLabel.pack()

def reset_counter():
    global counter
    counter=0
resetButton=ctk.CTkButton(master=app,font=("Montserrat", 16),height=40,width=120,text="Reset",text_color="white",fg_color="teal",command=reset_counter)
resetButton.pack()
cap=cv2.VideoCapture(0)

import pathlib
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

model=torch.hub.load('ultralytics/yolov5','custom',path='yolov5/runs/train/exp8/weights/best.pt',force_reload=True)

def detect():
    global counter
    ret,frame=cap.read()
    frame=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    results=model(frame)
    img=np.squeeze(results.render())

    if len(results.xywh[0])>0:
        dconf=results.xywh[0][0][4]
        dclass=results.xywh[0][0][5]

        if dconf.item() >0.85  and dclass.item() == 16.00:
            p=vlc.MediaPlayer(f"file:///{1}.wav")
            p.play()
            counter+=1


    imgarr=Image.fromarray(img)
    imgtk=ImageTk.PhotoImage(imgarr)
    vid.imgtk=imgtk
    vid.configure(image=imgtk)
    vid.after(10,detect)


detect()
app.mainloop()
counterLabel.configure(text=counter)