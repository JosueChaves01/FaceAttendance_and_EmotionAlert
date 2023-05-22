
import sys
import requests
import json
import cv2
import numpy as np
from matplotlib import pyplot as plt
import cognitive_face as CF
from PIL import Image, ImageDraw, ImageFont
from IPython.display import display, Javascript, Image
import js2py
from datetime import datetime
from base64 import b64decode, b64encode
import PIL
import io
import html
import time
import pickle
# The key from the Azure API
SUBSCRIPTION_KEY = '1eb956070abd44298d80d868e3ece851'
# The endpoint from the Azure API
BASE_URL = 'https://faceattendancearquitectura.cognitiveservices.azure.com/face/v1.0/'
CF.BaseUrl.set(BASE_URL)
CF.Key.set(SUBSCRIPTION_KEY)


def create_group(group_id, group_name):
    """This function create a new group of people, you only need to create it once

    Parameters
    ----------
        group_id : int
            The group id number
        group_name : str
            The new group name
    """
    res = CF.person_group.create(group_id, group_name)
    print("Group created")


def delete_group(group_id):
    """This function delete a group of people, you only need to delete it once

    Parameters
    ----------
        group_id : int
            The group id number
    """

    res = CF.person_group.delete(group_id)
    print("Group deleted")


def add_picture_to_person(picture, group_id, person_id):
    """Add a picture to a person and train the new model,
    Only one person should come in the photo

    Parameters
    ----------
        picture : str
            The image path of the person

        group_id : int
            The group id of the person

        person_id : str
            Ther person id assigned by azure
    """

    # Sumarle una foto a la persona que se ha creado
    CF.person.add_face(picture, group_id, person_id)
    # print CF.person.lists(PERSON_GROUP_ID)

    # Re-entrenar el modelo porque se le agrego una foto a una persona
    CF.person_group.train(group_id)
    # Obtener el status del grupo
    response = CF.person_group.get_status(group_id)
    status = response['status']
    print(status)


def show_image(image_path):
    """This function displays an image using previously initialized libraries

    Parameters
    ----------
        image_path : str
            The image path for display the image
    """
    img = cv2.imread(image_path)
    img_cvt = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img_cvt)
    plt.show()


def show_image_with_square(image_path, analysis):
    """This function displays and image with squares in the person's face
    This function needs the image path and the Azure data to work

    Parameters
    ----------
        image_path : str
            The image path for display the image

        analysis : list
            A list with all the person's information dictionaries
    """
    img = cv2.imread(image_path)
    im2Display = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    imTemp = im2Display.copy()

    faces = []
    for face in analysis:
        fr = face['faceRectangle']  # Get the square position for all faces
        faces.append(fr)  # Save the square position in a list
        top = fr['top']
        left = fr['left']
        width = fr['width']
        height = fr['height']
        pt1 = (left, top)
        pt2 = (left+width, top+height)

        color = (23, 200, 54)  # Define the square color
        thickness = 10  # Define the thickness of the square
        cv2.rectangle(imTemp, pt1, pt2, color, thickness)  # Create the square
    plt.imshow(imTemp)
    plt.show()


def peoples_information(picture):
    """This function get the people information from an image
    This function need the image path and return the people information

    Parameters
    ----------
        picture : str
            The image path for display the image
    """
    
    headers = {'Ocp-Apim-Subscription-Key': SUBSCRIPTION_KEY,
               'Content-Type': 'application/octet-stream'}
    params = {
        'returnFaceId': 'true',
        'returnFaceLandmarks': 'false',
        'returnFaceAttributes': 'age,gender,headPose,smile,facialHair,glasses,emotion,hair,makeup,occlusion,accessories,blur,exposure,noise,name',
    }
    response = requests.post(
        BASE_URL + "detect/", headers=headers, params=params, data=picture)
    analysis = response.json()  # convert a json file to a dictionary in a list
    return analysis


def print_people(group_id):
    """Displays the people lists of a specific group

    Parameters
    ----------
        group_id : int
            The corresponding group id to show the information
    """
    for dic in CF.person.lists(group_id):
        print(dic)
    return CF.person.lists(group_id)


def recognize_person(image_path, group_id):
    """Recognize peolpe through a photo

    Parameters
    ----------
        image_path : str
            The image path for display the image

        group_id : int
            The corresponding group id to serach the information
    """
    # print(image_path, " ", group_id)

    # Detentando los rostros en las fotos
    response = peoples_information(image_path)
    # for face in response:
    # print(face['faceId'])
    nombres = []
    # Obteniendo los ids de los rostros en las fotos
    face_ids = [d['faceId'] for d in response]
    # print("id de los rostros", face_ids)
    if face_ids != []:
      # Identificar personas en con esos rostros
        identified_faces = CF.face.identify(face_ids, group_id)
        for person in identified_faces:
            faceId = person['faceId']
            candidates_list = person['candidates']
            # print("candidates_list: ", candidates_list)
            for candidate in candidates_list:
                personId = candidate['personId']
                person_data = CF.person.get(group_id, personId)
                person_name = person_data['name']
                print(person_name)
                nombres.append(person_name)
                for face in response:
                    faceId2 = face['faceId']
                    if faceId2 == faceId:
                        faceRectangle = face['faceRectangle']
                        width = faceRectangle['width']
                        top = faceRectangle['top']
                        height = faceRectangle['height']
                        left = faceRectangle['left']
                        age = face['faceAttributes']['age']

                        img = cv2.imread(image_path)
                        # Características del texto
                        texto = person_name
                        ubicacion = (left, top)
                        font = cv2.FONT_HERSHEY_TRIPLEX
                        tamañoLetra = 2
                        colorLetra = (999, 999, 1)
                        grosorLetra = 2

                        # Write text
                        cv2.putText(img, texto, ubicacion, font, tamañoLetra,
                                    colorLetra, grosorLetra)

                        im2Display = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        imTemp = im2Display.copy()

                        pt1 = (left, top)
                        pt2 = (left+width, top+height)

                        color = (23, 200, 54)
                        thickness = 10
                        cv2.rectangle(imTemp, pt1, pt2, color, thickness)

                        plt.imshow(imTemp)
                        plt.show()

                        # *******************************************
                        current_time = datetime.datetime.now()
                        date = f"{current_time.day}/{current_time.month}/{current_time.year}"
                        time = f"{current_time.hour}:{current_time.minute}"

                        emotions = face['faceAttributes']['emotion']

                        anger = emotions['anger']
                        happiness = emotions['happiness']

                        report_detect(date, time, faceId, personId,
                                      age, image_path, emotions)

                        # *******************************************
    return nombres            


def js_to_image(js_reply):
    """ Function to convert the JavaScript object into an OpenCV image
    Parameters
    ----------
        js_reply
            JavaScript object containing image from webcam
    Returns
    -------
        img
            OpenCV BGR image
    """
    # decode base64 image
    image_bytes = b64decode(js_reply.split(',')[1])
    # convert bytes to numpy array
    jpg_as_np = np.frombuffer(image_bytes, dtype=np.uint8)
    # decode numpy array into OpenCV BGR image
    img = cv2.imdecode(jpg_as_np, flags=1)

    return img


def bbox_to_bytes(bbox_array):
    """Function to convert OpenCV Rectangle bounding box image into
    base64 byte string to be overlayed on video stream

    Parameters
    ----------
        bbox_array
            Numpy array (pixels) containing rectangle to overlay on video stream.
    Returns
    -------
        bytes
            Base64 image byte string
    """
    # convert array into PIL image
    bbox_PIL = PIL.Image.fromarray(bbox_array, 'RGBA')
    iobuf = io.BytesIO()
    # format bbox into png for return
    bbox_PIL.save(iobuf, format='png')
    # format return string
    bbox_bytes = 'data:image/png;base64,{}'.format(
        (str(b64encode(iobuf.getvalue()), 'utf-8')))

    return bbox_bytes


def video_frame(label, bbox):
    """Create the video frame

    Parameters
    ----------
        label
            A descriptive term of the video frame

        bbox
            Rectangle to overlay on video stream

    """
    data = js2py.eval_js('stream_frame("{}", "{}")'.format(label, bbox))
    return data


def create_person(name, profession, picture, group_id, information1, information2):
    """Create a new person in a group, only one person should come in the photo 

    Parameters
    ----------
        name : st
            The person name

        profession : str
            The profession of the person

        picture : str
            The image_path

        group_id : int
            The person group

        information1 : str
            Some information about the person

        information2 : str
            Some information about the person   
    """
    response = CF.person.create(group_id, name, group_id)
    print(response)
    # En response viene el person_id de la persona que se ha creado
    # Obtener person_id de response
    person_id = response['personId']
    print(person_id)
    # Sumarle una foto a la persona que se ha creado
    CF.person.add_face(picture, group_id, person_id)
    print(CF.person.lists(group_id))

    # Re-entrenar el modelo porque se le agrego una foto a una persona
    CF.person_group.train(group_id)
    # Obtener el status del grupo
    response = CF.person_group.get_status(group_id)
    status = response['status']
    print(status)

    # Obtain more information about the person
    analysis = peoples_information(picture)

    # Create the instance for the person
    create_instances(group_id, analysis, person_id, name,
                     picture, information1, information2)

    class Person:
        """Class that represent a Person"""

    def __init__(self, identification, person_id, name, age, gender, image_path):
        """Person class initializer"""
        self.identification = identification
        self.person_id = person_id
        self.name = name
        self.age = age
        self.gender = gender
        self.image_path = image_path

    def __str__(self):
        """Shows the Person information"""
        return "Indentification: {}, Person ID: {}, Name: {}, Age: {}, Gender: {}, Image Path: {}, ". format(self.identification, self.person_id, self.name, self.age, self.gender, self.image_path)


class Detect():
    """Class that represent a detection of a person"""

    def __init__(self, date, time, id_detected, person_id, age, image_path, emotions):
        """dectection class initializer"""
        self.date = date
        self.time = time
        self.id_detected = id_detected
        self.person_id = person_id
        self.age = age
        self.image_path = image_path
        self.emotions = emotions

    def __str__(self):
        """Shows the detected information"""
        return f"FaceId: {self.id_detected}, Date: {self.date}, Time: {self.time}"


class Person:
    """Class that represent a Person"""

    def __init__(self, identification, person_id, name, age, gender, image_path):
        """Person class initializer"""
        self.identification = identification
        self.person_id = person_id
        self.name = name
        self.age = age
        self.gender = gender
        self.image_path = image_path

    def __str__(self):
        """Shows the Person information"""
        return "Indentification: {}, Person ID: {}, Name: {}, Age: {}, Gender: {}, Image Path: {}, ". format(self.identification, self.person_id, self.name, self.age, self.gender, self.image_path)


class Political(Person):
    """Class that represent a Political"""

    def __init__(self, identification, person_id, name, age, gender, image_path,
                 political_party, has_been_president):
        """Contructor of the class Political"""

        super().__init__(identification, person_id, name, age,
                         gender, image_path)
        self.political_party = political_party
        self.has_been_president = has_been_president

    def __str__(self):
        """Show the Political class information"""

        return super().__str__() + "Political Party: {}, Has Been President?: {}".format(self.political_party, self.has_been_president)


class Interviewer(Person):
    """Class that represent a Interviewer"""

    def __init__(self, identification, person_id, name, age, gender, image_path,
                 company_where_works, years_working):
        """Contructor of the class Interviewer"""
        super().__init__(identification, person_id, name, age,
                         gender, image_path)
        self.company_where_works = company_where_works
        self.years_working = years_working

    def __str__(self):
        """Shows the Interviewer information"""
        return super().__str__() + "Company Where Works: {}, Years Working: {}".format(self.company_where_works, self.years_working)


class Guest(Person):
    """Class that represent a Guest person"""

    def __init__(self, identification, person_id, name, age, gender, image_path,
                 profession, nationality):
        super().__init__(identification, person_id, name, age,
                         gender, image_path)
        self.profession = profession
        self.nationality = nationality

    def __str__(self):
        """Shows the Guess information"""
        return super().__str__() + "Profession: {}, Nacionality: {}".format(self.profession, self.nationality)


def create_instances(group_id, analysis, person_id, name, picture, information1, information2):
    """Create the person instance according to the group_id

    Parameters
    ----------        
        group_id : int
            The person group

        analysis :  list
            Obtained information from azure

        person_id : str
            The person id assigned by azure

        name : st
            The person name

        picture : str
            The image_path

        information1 : str
            Some information about the person

        information2 : str
            Some information about the person
    """
    # Create a politic
    if group_id == 10:
        with open("students.bin", "ab") as f:
            politician = Political(analysis[0]['faceId'], person_id, name,
                                   analysis[0]['faceAttributes']['age'],
                                   analysis[0]['faceAttributes']['gender'],
                                   picture, information1, information2)
            print(politician.name, politician.person_id)
            pickle.dump(politician, f, pickle.HIGHEST_PROTOCOL)

    show_image_with_square(picture, analysis)


def report_detect(date, time, faceId, personId, age, image_path, emotions):
    """Create a new instace of the detect class and store this in a file

    Parameters
    ----------
        date : str
            The date of creation

        time : str
            The time of creation

        faceId : str
            The person faceId

        personId : str
            The person id assigned by azure

        age : int
            The person age

        image_path : str
            The image_path of the image

        emotions : dict
            The emotions of the person
    """
    with open("detect.bin", "ab") as f:
        detect = Detect(date, time, faceId, personId, age, image_path,
                        emotions)
        pickle.dump(detect, f, pickle.HIGHEST_PROTOCOL)

def markAttendance(name):
    with open('Attendance.csv','r+') as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.now()
            time = now.strftime('%I:%M:%S:%p')
            date = now.strftime('%d-%B-%Y')
            f.writelines(f'n{name}, {time}, {date}')


def prototype():
    # initialze bounding box to empty
    nombres = []
    bbox = ''
    count = 0
    j = 1
    while True:
        # Inicializar la cámara
        cap = cv2.VideoCapture(0)

        # Leer la imagen desde la cámara
        ret, frame = cap.read()

        # Si la lectura es erronea se salta
        if not ret:
            break
        
    
        # se guarda el fotograma con el nombre más el número del contador i
        cv2.imwrite('C:/Users/josuc/Desktop/IOT/Imagenes' +
                    'nombre'+str(j)+'.jpg', frame)
        image_path = 'C:/Users/josuc/Desktop/IOT/Imagenes' + \
            'nombre'+str(j)+'.jpg'
        analysis = peoples_information(image_path)
        print("Imagen: " + 'nombre'+str(j)+'.jpg')
        
        try:
            nombres = recognize_person(image_path, 10)
        except:
            try:
                nombres = recognize_person(image_path, 20)
            except:
                try:
                    nombres = recognize_person(image_path, 30)
                except:
                    print("")
        
        for nombre in nombres:
            print(nombre)

            
        # Liberar los recursos de la cámara
        cap.release()
        
prototype()

