import tkinter as tk
import cv2
from PIL import Image, ImageTk
from main import *
class VideoCaptureApp:
    def __init__(self, window, video_source=0):
        self.window = window
        self.window.title("Video Capture")
        self.video_source = video_source

        # Inicializar el objeto de captura de video
        self.cap = cv2.VideoCapture(self.video_source)

        # Crear un objeto de lienzo Tkinter para mostrar el video
        self.canvas = tk.Canvas(window, width=self.cap.get(cv2.CAP_PROP_FRAME_WIDTH), height=self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.canvas.pack()

        # Agregar un botón para tomar una captura de pantalla
        self.btn_snapshot = tk.Button(window, text="Tomar captura", command=self.snapshot)
        self.btn_snapshot.pack(anchor=tk.CENTER, expand=True)

        # Actualizar el lienzo de video
        self.delay = 15
        self.update()
        
        
        # Configurar el cierre de la ventana
        self.window.protocol("WM_DELETE_WINDOW", self.on_close)

    def update(self):
        # Obtener un cuadro de video
        ret, frame = self.cap.read()

        if ret:
            # Convertir el marco de OpenCV a una imagen de PIL
            self.photo = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)

        # Actualizar el lienzo después de un breve retraso
        self.window.after(self.delay, self.update)

    def snapshot(self):
        # Tomar una captura de pantalla y guardarla como un archivo PNG
        ret, frame = self.cap.read()
        if ret:
            cv2.imwrite("captura.png", cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    def on_close(self):
        # Liberar el objeto de captura de video y cerrar la ventana
        self.cap.release()
        self.window.destroy()

# Crear una instancia de la ventana principal
root = tk.Tk()

# Crear una instancia de la aplicación de captura de video
app = VideoCaptureApp(root)

# Iniciar el bucle de eventos de Tkinter
root.mainloop()
