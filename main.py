import cv2
import face_recognition
import os
import numpy as np
from datetime import datetime
import pickle
import csv
from zmq import NULL


path = 'C:/Users/josuc/Desktop/Proyecto_Estructuras_2/Alumnos'

images = []
classNames = []
mylist = os.listdir(path)
for cl in mylist:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])

def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encoded_face = face_recognition.face_encodings(img)[0]
        encodeList.append(encoded_face)
    return encodeList
encoded_face_train = findEncodings(images)

def agregar_fila(archivo, fila):
    with open(archivo, 'a') as f:
        f.write(fila + '\n')

def markAttendance(name):
    datos = []
    
    with open('Attendance.txt', 'r') as f:
        for line in f:
            # Separar los elementos de cada línea por comas
            elements = line.strip().split(',')
            # Agregar el primer elemento de la línea a la lista de nombres
            datos.append(elements[0]+elements[2])
        

    now = datetime.now()
    todayDate = now.strftime('%d-%B-%Y')
    dato = name+todayDate
    print(datos)
    if not dato in datos:
            with open('Attendance.txt', mode='a') as file:

                now = datetime.now()
                time = now.strftime('%I:%M:%S:%p')
                date = now.strftime('%d-%B-%Y')
                newData = [name, time, date]
                
                # Convertir la lista newData en una cadena separada por comas
                newLine = ','.join(newData)
                
                # Agregar un salto de línea al final de la cadena
                newLine += '\n'
                
                # Escribir la nueva línea en el archivo
                file.writelines(newLine)

def mostraPersonasPresentes():
    personasPresentes = []
    with open('Attendance.txt', mode='r') as file:
        for line in file:
            elements = line.strip().split(',')
            if not elements[0] in personasPresentes:
                personasPresentes.append(elements[0])
    return personasPresentes

def __main__(): 
    # take pictures from webcam 
    cap  = cv2.VideoCapture(0)
    while True:
        success, img = cap.read()
        imgS = cv2.resize(img, (0,0), None, 0.25,0.25)
        imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
        faces_in_frame = face_recognition.face_locations(imgS)
        encoded_faces = face_recognition.face_encodings(imgS, faces_in_frame)
        for encode_face, faceloc in zip(encoded_faces,faces_in_frame):
            matches = face_recognition.compare_faces(encoded_face_train, encode_face)
            faceDist = face_recognition.face_distance(encoded_face_train, encode_face)
            matchIndex = np.argmin(faceDist)
            
            if matches[matchIndex]:
                name = classNames[matchIndex].upper().lower()
                y1,x2,y2,x1 = faceloc
                # since we scaled down by 4 times
                y1, x2,y2,x1 = y1*4,x2*4,y2*4,x1*4
                cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
                cv2.rectangle(img, (x1,y2-35),(x2,y2), (0,255,0), cv2.FILLED)
                cv2.putText(img,name, (x1+6,y2-5), cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
                
                markAttendance(name)
        cv2.imshow('webcam', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

if __name__ == '__main__':
    for persona in mostraPersonasPresentes():
        print(persona)