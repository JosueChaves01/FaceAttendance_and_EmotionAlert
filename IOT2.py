import requests
import json
import cv2
import io
from datetime import datetime
import tkinter as tk
import threading
import cognitive_face as CF
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Variables de configuración
subscription_key = '1eb956070abd44298d80d868e3ece851'  # Reemplaza con tu clave de suscripción
endpoint = 'https://faceattendancearquitectura.cognitiveservices.azure.com/'  # Reemplaza con tu endpoint
group_id = '10'  # Reemplaza con el ID del grupo al que pertenecen los rostros

# URL de la API de detección de rostros
face_api_url = f'{endpoint}/face/v1.0/detect'

# Atributos de rostro compatibles
face_attributes = 'age,gender,headPose,smile,facialHair,glasses,emotion,hair,makeup,accessories'

# Función para detectar rostros en una imagen
def detect_faces(image):
    headers = {
        'Ocp-Apim-Subscription-Key': subscription_key,
        'Content-Type': 'application/octet-stream'
    }

    params = {
        'returnFaceId': 'true',
        'returnFaceLandmarks': 'false',
        'returnFaceAttributes': face_attributes,
    }

    response = requests.post(face_api_url, headers=headers, params=params, data=image)
    faces = response.json()
    return faces

# Función para identificar un rostro en el grupo
def identify_face(face_id):
    identify_api_url = f'{endpoint}/face/v1.0/identify'
    
    headers = {
        'Ocp-Apim-Subscription-Key': subscription_key,
        'Content-Type': 'application/json'
    }
    
    data = {
        'faceIds': [face_id],
        'personGroupId': group_id,
        'confidenceThreshold': 0.5
    }
    
    response = requests.post(identify_api_url, headers=headers, json=data)
    result = response.json()
    return result

def limpiar_archivo():
    with open("Emotions.txt", 'r+') as archivo:
        lineas = archivo.readlines()
        archivo.seek(0)
        archivo.truncate()
        for linea in lineas:
            linea = ""

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
            
    # Verificar si se cumplen las condiciones para enviar una alerta por correo
    if count_anger_emotions(name) > 5 and get_max_anger_emotion_value(name) > 0.7:
        limpiar_archivo()
        print("Alerta!!")
        

def count_anger_emotions(name):
    count = 0
    with open('Emotions.txt', mode='r') as file:
        for line in file:
            elements = line.strip().split(',')
            if elements[0] == name and elements[1] == 'anger':
                count += 1
    return count

def get_max_anger_emotion_value(name):
    max_value = 0
    with open('Emotions.txt', mode='r') as file:
        for line in file:
            elements = line.strip().split(',')
            if elements[0] == name and elements[1] == 'anger':
                value = float(elements[2])
                if value > max_value:
                    max_value = value
    return max_value

def send_alert_email(name):
    # Configuración del correo electrónico
    smtp_server = 'smtp.gmail.com'  # Reemplaza con el servidor SMTP adecuado
    smtp_port = 587  # Reemplaza con el puerto SMTP adecuado
    sender_email = 'josuchazu.jc@gmail.com'  # Reemplaza con tu dirección de correo electrónico
    receiver_email = 'josuchazu.jecz@gmail.com'  # Reemplaza con la dirección de correo electrónico del receptor
    password = 'Josuchazu0102*'  # Reemplaza con tu contraseña de correo electrónico
    
    # Crear el mensaje de correo electrónico
    message = MIMEMultipart()
    message['From'] = sender_email
    message['To'] = receiver_email
    message['Subject'] = 'Alerta de emociones enojo'
    
    # Cuerpo del mensaje
    body = f'Se ha detectado que {name} ha registrado más de 5 emociones de enojo con un valor por encima de 0.7.'
    message.attach(MIMEText(body, 'plain'))
    
    # Enviar el correo electrónico
    with smtplib.SMTP(smtp_server, smtp_port) as server:
        server.starttls()
        server.login(sender_email, password)
        server.sendmail(sender_email, receiver_email, message.as_string())

def mostrarPersonasPresentes():
    personasPresentes = []
    with open('Attendance.txt', mode='r') as file:
        for line in file:
            elements = line.strip().split(',')
            if not elements[0] in personasPresentes:
                personasPresentes.append(elements[0])
    return personasPresentes

# Función principal del programa
def main():
    # Inicializar la cámara
    cap = cv2.VideoCapture(0)
    
    # Crear la ventana principal de la aplicación
    root = tk.Tk()
    root.title("Face Attendance")
    root.geometry("700x800")
    
    # Crear una etiqueta para mostrar la información de personas presentes
    lbl_personas_presentes = tk.Label(root, text="Personas presentes:")
    lbl_personas_presentes.pack()
    
    # Crear una lista para mostrar las personas presentes
    lst_personas_presentes = tk.Listbox(root)
    lst_personas_presentes.pack()
    
    # Obtener las personas presentes y mostrarlas en la lista
    personas_presentes = mostrarPersonasPresentes()
    for persona in personas_presentes:
        lst_personas_presentes.insert(tk.END, persona)
    
    # Crear el botón para salir
    btn_salir = tk.Button(root, text="Salir", command=root.destroy)
    btn_salir.pack()

    # Función para cerrar la aplicación
    def cerrar_app():
        root.destroy()
        
    def actualizar_personas_presentes():
        # Obtener las personas presentes y actualizar la lista
        personas_presentes = mostrarPersonasPresentes()
        lst_personas_presentes.delete(0, tk.END)
        for persona in personas_presentes:
            lst_personas_presentes.insert(tk.END, persona)
        
        # Volver a llamar a la función después de 1 segundo
        root.after(1000, actualizar_personas_presentes)
    
    # Llamar a la función para actualizar las personas presentes
    actualizar_personas_presentes()
    
    def procesar_fotograma():
        # Leer un fotograma de la cámara
        ret, frame = cap.read()
        
        # Convertir el fotograma a formato JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        
        # Detectar rostros en el fotograma
        faces = detect_faces(buffer.tobytes())
        
        if 'error' in faces:
            print('Error de solicitudes, la cantidad de solicitudes por minuto fue superada')
        else:
            for face in faces:
                face_id = face['faceId']
                
                # Identificar el rostro en el grupo
                result = identify_face(face_id)
                
                if 'error' in result:
                    print('Error al identificar rostro:', result['error']['message'])
                elif len(result) == 0 or len(result[0]['candidates']) == 0:
                    print('No se pudo identificar al rostro.')
                else:
                    person_id = result[0]['candidates'][0]['personId']
                    confidence = result[0]['candidates'][0]['confidence']
                    
                    # Consultar información de la persona identificada
                    person_api_url = f'{endpoint}/face/v1.0/persongroups/{group_id}/persons/{person_id}'
                    
                    headers = {
                        'Ocp-Apim-Subscription-Key': subscription_key
                    }
                    
                    response = requests.get(person_api_url, headers=headers)
                    person = response.json()
                    
                    if 'error' in person:
                        print('Error al obtener información de la persona:', person['error']['message'])
                    else:
                        name = person["name"]
                        markAttendance(name)
                        
                        # Registrar las emociones en un archivo de texto
                        emotions = face['faceAttributes']['emotion']
                        with open('Emotions.txt', mode='a') as file:
                            for emotion, value in emotions.items():
                                file.write(f'{name},{emotion},{value}\n')
                        
                        # Dibujar un rectángulo alrededor del rostro identificado
                        x = face['faceRectangle']['left']
                        y = face['faceRectangle']['top']
                        width = face['faceRectangle']['width']
                        height = face['faceRectangle']['height']
                        
                        cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)
                        
                        # Mostrar el nombre y la confianza en el marco
                        text = f'{name} ({confidence:.2f})'
                        cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        # Mostrar el fotograma en una ventana
        cv2.imshow('Face Attendance', frame)
        
        # Salir del programa al presionar la tecla 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cerrar_app()
            return
        
        # Volver a llamar a la función después de 1 milisegundo
        root.after(1, procesar_fotograma)
    
    # Llamar a la función para procesar los fotogramas
    procesar_fotograma()
    
    # Iniciar el bucle de la aplicación
    root.mainloop()

# Llamar a la función principal
if __name__ == '__main__':
    main()
