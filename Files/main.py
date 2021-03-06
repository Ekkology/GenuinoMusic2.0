import numpy as np
from PIL import ImageTk, Image
import librosa
import os
#imports para la gui
import tkinter as tk
from tkinter import Label, PhotoImage, ttk
from tkinter import messagebox
from tkinter.filedialog import askopenfilename
import json
from sklearn.model_selection import train_test_split
import soundfile as sf
import tensorflow as tf 
import tensorflow.keras as keras
import matplotlib.pyplot as plt
import random
import soundfile
import math
window = tk.Tk()


def choose_file():
    filename = askopenfilename()
    return filename




#Logica de prediccion 
#Se carga el Modelo  .H5
modelito = keras.models.load_model("Files\Music_Genre_10_CNN.h5")
# Audio files pre-processing

def process_input(audio_file, track_duration):
  
                SAMPLE_RATE = 22050
                NUM_MFCC = 13
                N_FTT=2048
                HOP_LENGTH=512
                TRACK_DURATION = track_duration # measured in seconds
                SAMPLES_PER_TRACK = SAMPLE_RATE * TRACK_DURATION
                NUM_SEGMENTS = 10

                samples_per_segment = int(SAMPLES_PER_TRACK / NUM_SEGMENTS)
                num_mfcc_vectors_per_segment = math.ceil(samples_per_segment / HOP_LENGTH)

                signal, sample_rate = librosa.load(audio_file, sr=SAMPLE_RATE)
  
                for d in range(10):

                    # calculate start and finish sample for current segment
                    start = samples_per_segment * d
                    finish = start + samples_per_segment

                    # extract mfcc
                    mfcc = librosa.feature.mfcc(signal[start:finish], sample_rate, n_mfcc=NUM_MFCC, n_fft=N_FTT, hop_length=HOP_LENGTH)
                    mfcc = mfcc.T

                    return mfcc

#Propiedades de la Ventana
  
window.title("GenuinoMusic")
window.geometry('400x400')
#window.iconbitmap("Files\clef.ico")
window.maxsize(width=400,height=400)

frame1 = tk.Frame(window, bg='#e01656')

frame1.pack(fill='both', expand='yes')
#-------------------------#

#
#lbl1 = tk.Label(frame1, image=photo)
#lbl1.photo = photo
#lbl1.pack()
bgframe="snow"
frame2 = tk.Frame(frame1, bg=bgframe)
frame2.place(relx=0.017, rely=0.022, relheight=0.95, relwidth=0.96)

#Widgets
imgenFondo =  PhotoImage(file = "Files\Mixing400x400.gif")
lebelFondo = Label(frame2,image=imgenFondo).place(x=0,y=0)
imgen_Fisc =  ImageTk.PhotoImage(file="Files\FISC_utp 100x100.png")
imagen_FiscTK = tk.Label(frame2, image=imgen_Fisc)
imagen_FiscTK.place(x = 250, y = 10)
img = ImageTk.PhotoImage(file="Files/utp100x99.png")
imagenutp = tk.Label(frame2, image=img)
imagenutp.place(x = 53, y = 10)

lbl0 = tk.Label(frame1, text="Bienvenido",
                bg=bgframe,
                fg='Black',
                font=("Century Gothic",40))
lbl0.place(x=58,y=150)

lbl1 = tk.Label(frame1, text="UTP ?? 2022",
                bg=bgframe,
                fg='Black',
                font=("Helvetica",8))
lbl1.place(x=161,y=365)

progressbar = ttk.Progressbar(frame1)
progressbar.place(x=95, y=240, width=200)   

    
def identificar_genero():
    audiopath = choose_file()
     
    if(audiopath != ""):
        base=os.path.basename(audiopath)
        songname = os.path.splitext(base)[0]
        try:
             
            genero = {0:"pop",1:"jazz",2:"County",3:"classical",4:"metal",5:"Hip-Hop",6:"rock",7:"blues",8:"reggae",9:"disco"}
            new_input_mfcc = process_input(audiopath,30)
            X_to_predict = new_input_mfcc[np.newaxis, ..., np.newaxis]
            prediction = modelito.predict(X_to_predict)
            predicted_index = np.argmax(prediction, axis=1)
          
            text = "La canci??n:\n"  + songname + "\nEs del g??nero: " + genero[int(predicted_index)] + "."
            messagebox.showinfo("????xito!", text)
            progressbar.step(0.2)


        except:
                messagebox.showinfo("Error", "El archivo seleccionado:\n" + base + "\nEs inv??lido o no ha podido ser le??do")
                progressbar.stop()

    else:

     messagebox.showinfo("Error", "Ning??n archivo seleccionado")


btn0 = tk.Button(frame1, text="Identificar G??nero Canci??n",
                 bg='ghost white',
                 fg='gray11',
                 font=("Helvetica",10),
                 command= identificar_genero)
btn0.place(x=113,y=280)

def click_credits():         
    tk.messagebox.showinfo("Cr??ditos","\n\tOrlando De Jes??s Montenegro Samudio\n \tMichelle Ariadne Justavino V??quez \n \tSebasti??n NULL  Pimentel S??nchez \n\n")
    
    
btn1 = tk.Button(frame1, text="Cr??ditos",
                 bg='ghost white',
                 fg='gray11',
                 font=("Helvetica",10),
                 command=click_credits)
btn1.place(x=161,y=330)


window.mainloop()