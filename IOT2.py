import requests
import json
import cv2
import io
from datetime import datetime

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

# Función principal del programa
def main():
    # Inicializar la cámara
    cap = cv2.VideoCapture(0)
    
    while True:
        # Leer un fotograma de la cámara
        ret, frame = cap.read()
        
        # Convertir el fotograma a formato JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        
        # Detectar rostros en el fotograma
        faces = detect_faces(buffer.tobytes())
        print(faces)
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
        
        # Salir del bucle si se presiona la tecla 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Liberar la cámara y cerrar las ventanas
    cap.release()
    cv2.destroyAllWindows()

# Ejecutar el programa
if __name__ == '__main__':
    main()
