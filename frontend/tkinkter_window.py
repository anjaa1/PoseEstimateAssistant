import threading
import tkinter as tk
import traceback
from tkinter import Frame, scrolledtext

import cv2
import mediapipe as mp
from PIL import Image, ImageTk
import math


class LandmarkDetectorApp:

    pose_score = 0
    time_correct_pose = 13

    def __init__(self, master, poseEstimation_model, rasaClient):
        self.master = master
        self.master.title("Sitzhaltungsassistent")
        self.poseEstimation_model = poseEstimation_model
        self.rasaClient = rasaClient

        # Attributes for error-management
        self.PoseError = False

        # Main frame for the whole app
        self.main_frame = Frame(self.master, padx=20, pady=20)
        self.main_frame.pack(fill='both', expand=True)

        # Left frame for video display
        self.left_frame = Frame(self.main_frame, width=640)
        self.left_frame.pack(side=tk.LEFT, fill='both', expand=True)

         # Right frame for chatbot
        self.right_frame = Frame(self.main_frame, width=320)
        self.right_frame.pack(side=tk.RIGHT, fill='both', expand=False)

        # Pose detection
        self.time_correct_pose = 0
        self.time_incorrect_pose = 0
        self.time_from_start = 0
        self.goodTime_from_start = 0
        self.aligned = False

        # KPI frame
        self.KPI_frame = Frame(self.left_frame)
        self.KPI_frame.configure(background="black")
        self.KPI_frame.pack(fill=tk.X)  # Make the KPI frame as wide as the parent frame

        self.KPI_pose_time_label = tk.Label(self.KPI_frame, text=f"Korrekte Haltungsdauer: - Sekunden", bg='lightgreen')
        self.KPI_pose_time_label.config(borderwidth=2, relief="groove")
        self.KPI_pose_time_label.grid(row=0, column=0, sticky="nsew")  # Use sticky to expand label to fill the cell

        self.KPI_pose_ProcentTime_label = tk.Label(self.KPI_frame, text=f"Korrekte Haltung: 0 %")
        self.KPI_pose_ProcentTime_label.config(borderwidth=2, relief="groove")
        self.KPI_pose_ProcentTime_label.grid(row=0, column=1, sticky="nsew")  # Use sticky to expand label to fill the cell

        self.KPI_pose_align_label = tk.Label(self.KPI_frame, text="Kamera nicht ausgerichtet", bg = "red")
        self.KPI_pose_align_label.config(borderwidth=2, relief="groove")
        #self.KPI_pose_align_label.grid(row=0, column=2, sticky="nsew")  # Use sticky to expand label to fill the cell

        self.KPI_frame.columnconfigure(0, weight=1)  # Make column 0 of the KPI frame expandable
        self.KPI_frame.columnconfigure(1, weight=1)  # Make column 1 of the KPI frame expandable
        #self.KPI_frame.columnconfigure(2, weight=1)  # Make column 2 of the KPI frame expandable

        self.canvas = tk.Canvas(self.left_frame, width=640, height=480)
        self.canvas.pack()

        # Frame for Landmark detection
        self.Landmark_frame = Frame(self.left_frame)
        self.Landmark_frame.configure(background="black")
        self.Landmark_frame.pack(fill=tk.X)

        self.start_button = tk.Button(self.Landmark_frame, text="Start", command=self.start_detection, bg='lightgrey')
        self.start_button.grid(row=0, column=0, padx=1, pady=1, sticky="ew")

        self.stop_button = tk.Button(self.Landmark_frame, text="Stopp", command=self.stop_detection, state=tk.DISABLED, bg='lightgrey')
        self.stop_button.grid(row=0, column=1, padx=1, pady=1, sticky="ew")

        self.error_info_label = tk.Label(self.Landmark_frame, text= "Alles funktioniert einwandfrei. Es wurde kein Fehler erkannt.", borderwidth=2, highlightthickness=2, highlightbackground="black", highlightcolor="black")
        self.error_info_label.grid(row=1, column=0, columnspan=2, padx=1, pady=1, sticky="ew")

        self.Landmark_frame.columnconfigure(0, weight=1) # Make column 0 of the Landmark frame expandable
        self.Landmark_frame.columnconfigure(1, weight=1) # Make column 1 of the Landmark frame expandable

        self.cap = cv2.VideoCapture(0)
        self.detecting = False
        self.update()

        # Chat window configuration
        self.chat_window = scrolledtext.ScrolledText(
            self.right_frame, wrap=tk.WORD, state=tk.DISABLED, bg='#F5F5F5', fg='#333333', font=("Helvetica", 10), padx=10, pady=10
        )
        self.chat_window.tag_configure("right", justify='right', background="#DCF8C6", foreground="#333333", font=("Helvetica", 10), lmargin1=10, lmargin2=10, rmargin=10)
        self.chat_window.tag_configure("left", justify='left', background="#FFFFFF", foreground="#333333", font=("Helvetica", 10), lmargin1=10, lmargin2=10, rmargin=10)
        self.chat_window.pack(expand=False, fill='both', padx=5, pady=5)

        # Create Frame for the entry and button at the bottom
        self.bottom_frame = tk.Frame(self.right_frame)
        self.bottom_frame.pack(fill='x', padx=5, pady=5)

        self.entry = tk.Entry(self.bottom_frame, font=('Helvetica', 12), width=50)
        self.entry.pack(side=tk.LEFT, padx=5, pady=5, fill='x', expand=True)
        self.entry.bind("<Return>", self.add_message)

        # Create Button to send message
        self.send_button = tk.Button(self.bottom_frame, text="->", command=self.add_message)
        self.send_button.pack(side=tk.LEFT, padx=5, pady=5)

        # add an Initial message
        self.chat_window.config(state=tk.NORMAL)
        self.chat_window.insert(tk.END, "Ich bin ein Chatbot, der dir deine Fragen beantworten kann.\n")
        self.chat_window.config(state=tk.DISABLED)

    def add_message(self, event=None):
        message = self.entry.get()
        if message:
            self.display_message(message, "right")
            threading.Thread(target=self.fetch_response, args=(message,)).start()

    def fetch_response(self, message):
        try:
            response = self.rasaClient.send_message(message)
            self.master.after(0, self.display_message, response, "left")
        except Exception as e:
            print("Error fetching response:", e)
            traceback.print_exc()

    def display_message(self, message, tag):
        self.chat_window.config(state=tk.NORMAL)  # Enable editing to add the new message
        self.chat_window.insert(tk.END, message + "\n", tag)  # Insert the message at the end
        self.chat_window.config(state=tk.DISABLED)  # Disable editing to prevent overwriting
        self.chat_window.see(tk.END)  # Scroll to the end to show the latest message
        self.entry.delete(0, tk.END)  # Clear the entry widget

    def update(self):
        
        # Variablen für Zeit
        temp_goodtime = 0
        temp_badtime = 0

        success, video = self.cap.read()
        if not success:
            # Fehlermeldung in HMI?
            print("Null.Frames")
            raise "Camera Error: "
        
        video = cv2.cvtColor(cv2.flip(video,1), cv2.COLOR_BGR2RGB)
        
        # Starte PoseEstimation, wenn Nutzer start geklickt hat
        if self.detecting:
            video, correctPose, temp_goodtime, temp_badtime, self.aligned, self.PoseError = self.poseEstimation_model(self.cap, video) # Start PoseEstimation
            
            # Wenn PoseEstimation ohne Fehler, dann...
            if self.PoseError == False:
                # Wenn Kamera korrekt seitlich ausgerichtet, dann...
                if self.aligned:
                    self.KPI_pose_align_label.config(text="Kamera ausgerichtet", bg = "lightgreen") # Label aligned aktualisieren
                    self.error_info_label.config(text="Die Haltungserkennung funktioniert einwandfrei", highlightthickness=2, highlightbackground="green", highlightcolor="green")

                    # Korrektheit der Haltung erfassen und Zeitlabel entsprechend aktualisieren
                    if correctPose:
                        # Zeiten für gute/schlechte Haltung aktualisieren
                        self.time_correct_pose += temp_goodtime
                        self.goodTime_from_start += temp_goodtime
                        self.time_incorrect_pose = 0
                        self.time_from_start += temp_goodtime
                        self.KPI_pose_time_label.config(text=f"Korrekte Haltungsdauer: {round(self.time_correct_pose)} Sekunden", bg='lightgreen')
                        self.KPI_pose_ProcentTime_label.config(text=f"Korrekte Haltung: {round((self.goodTime_from_start/self.time_from_start)*100)} %")
                    else:
                        # Zeiten für gute/schlechte Haltung aktualisieren
                        self.time_correct_pose = 0
                        self.time_incorrect_pose += temp_badtime
                        self.time_from_start += temp_badtime
                        self.KPI_pose_time_label.config(text=f"Falsche Haltungsdauer: {round(self.time_incorrect_pose)} Sekunden", bg='red')
                        self.KPI_pose_ProcentTime_label.config(text=f"Korrekte Haltung: {round((self.goodTime_from_start/self.time_from_start)*100)} %")
                else:
                    self.KPI_pose_align_label.config(text="Kamera nicht ausgerichtet", bg = "red") # Label aligned aktualisieren
                    self.KPI_pose_time_label.config(text=f"Korrekte Haltungsdauer: - Sekunden")
                    self.error_info_label.config(text="Bitte richte die Kamera seitlich aus.", highlightthickness=2, highlightbackground="red", highlightcolor="red")
                    # Zeiten für gute/schlechte Haltung zurücksetzen
                    self.time_correct_pose = 0
                    self.time_incorrect_pose = 0

            # ... andernfalls Fehlermeldung ausgeben im dafür vorgesehenen Label
            else:
                self.error_info_label.config(text=self.PoseError, highlightthickness=2, highlightbackground="red", highlightcolor="red")
                self.stop_detection()
                # Zeiten für gute/schlechte Haltung zurücksetzen
                self.time_correct_pose = 0
                self.time_incorrect_pose = 0
        
        # Transform video-stream to tk-image format
        img = Image.fromarray(video)
        imgtk = ImageTk.PhotoImage(image=img)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)
        self.photo = imgtk  # Keep a reference to avoid garbage collection
        self.master.after(10, self.update)

    def start_detection(self):
        self.detecting = True
        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)

        # Label einblenden
        self.KPI_pose_align_label.grid(row=0, column=2, sticky="nsew")  # Use sticky to expand label to fill the cell
        self.KPI_frame.columnconfigure(2, weight=1)  # Make column 2 of the KPI frame expandable

        # Zeiten zurücksetzen
        self.time_correct_pose = 0
        self.time_incorrect_pose = 0
        self.goodTime_from_start = 0
        self.time_from_start = 0

    def stop_detection(self):
        self.detecting = False
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)

        self.KPI_pose_align_label.grid_remove()
        self.KPI_frame.columnconfigure(2, weight=0)

        self.error_info_label.config(text="Der Haltungsassistent wurde gestoppt.", highlightthickness=2, highlightbackground="black", highlightcolor="black")
