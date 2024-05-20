import tkinter as tk
import traceback
from tkinter import Frame

import cv2
import mediapipe as mp
from PIL import Image, ImageTk

# Einbinden der anderen Python-Klassen
import os
import sys
# Relative Pfadangaben haben nicht funktioniert, daher absolute Pfadangaben mit os und sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../PoseEstimation')))
from PoseEstimation import PoseEstimation

class LandmarkDetectorApp:

    pose_score = 0
    time_correct_pose = 13

    def __init__(self, master):
        self.master = master
        self.master.title("Landmark Detector")

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
        self.aligned = False

        # KPI frame
        self.KPI_frame = Frame(self.left_frame)
        self.KPI_frame.configure(background="black")
        self.KPI_frame.pack(fill=tk.X)  # Make the KPI frame as wide as the parent frame

        self.KPI_pose_time_label = tk.Label(self.KPI_frame, text=f"Correct Pose Time: {self.time_correct_pose}")
        self.KPI_pose_time_label.config(borderwidth=2, relief="groove")
        self.KPI_pose_time_label.grid(row=0, column=0, sticky="nsew")  # Use sticky to expand label to fill the cell

        self.KPI_pose_score_label = tk.Label(self.KPI_frame, text=f"Pose Score: {self.pose_score}")
        self.KPI_pose_score_label.config(borderwidth=2, relief="groove")
        self.KPI_pose_score_label.grid(row=0, column=1, sticky="nsew")  # Use sticky to expand label to fill the cell

        self.KPI_pose_align_label = tk.Label(self.KPI_frame, text="Camera not aligned", fg = "red")
        self.KPI_pose_align_label.config(borderwidth=2, relief="groove")
        self.KPI_pose_align_label.grid(row=0, column=2, sticky="nsew")  # Use sticky to expand label to fill the cell

        self.KPI_frame.columnconfigure(0, weight=1)  # Make column 0 of the KPI frame expandable
        self.KPI_frame.columnconfigure(1, weight=1)  # Make column 1 of the KPI frame expandable
        self.KPI_frame.columnconfigure(2, weight=1)  # Make column 2 of the KPI frame expandable

        self.canvas = tk.Canvas(self.left_frame, width=640, height=480)
        self.canvas.pack()

        # Buttons for landmark detection
        self.start_button = tk.Button(self.left_frame, text="Start", command=self.start_detection)
        self.start_button.pack()

        self.stop_button = tk.Button(self.left_frame, text="Stop", command=self.stop_detection, state=tk.DISABLED)
        self.stop_button.pack()

        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose()

        self.cap = cv2.VideoCapture(0)
        self.detecting = False
        self.update()

        self.chat_window = tk.Text(self.right_frame, wrap=tk.WORD, state=tk.DISABLED, bg='white', fg='black', padx=10, pady=10)
        self.chat_window.pack(expand=False, fill='both', padx=5, pady=5)

        # Create Frame for the entry and button at the bottom
        self.bottom_frame = tk.Frame(self.right_frame)
        self.bottom_frame.pack(fill='x', padx=5, pady=5)

        self.entry = tk.Entry(self.bottom_frame, font=('Arial', 12), width=50)
        self.entry.pack(side=tk.LEFT, padx=5, pady=5, fill='x', expand=True)
        self.entry.bind("<Return>", self.add_message)

        # Create Button to send message
        self.send_button = tk.Button(self.bottom_frame, text="->", command=self.add_message)
        self.send_button.pack(side=tk.LEFT, padx=5, pady=5)

        # add an Initial message
        self.chat_window.config(state=tk.NORMAL)
        self.chat_window.insert(tk.END, "Chatverlauf\n")
        self.chat_window.config(state=tk.DISABLED)

    def add_message(self, event=None):
        message = self.entry.get()
        if message:
            self.chat_window.config(state=tk.NORMAL)  # Enable editing to add the new message
            self.chat_window.insert(tk.END, message + "\n")  # Insert the message at the end
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
            
        if self.detecting:
            video_stream = PoseEstimation(self.mp_pose, self.pose, self.cap, video, True) # PoseEstimation Objekt anlegen und initialisieren
            video, temp_goodtime, temp_badtime, self.aligned = video_stream(self.cap, video) # Start
            self.time_correct_pose += temp_goodtime
            self.time_incorrect_pose += temp_badtime
        
        if self.aligned:
            self.KPI_pose_align_label.config(text="Camera aligned", fg = "green") # Label aligned aktualisieren
            self.KPI_pose_time_label.config(text=f"Correct Pose Time: {self.time_correct_pose}")
        else:
            self.KPI_pose_align_label.config(text="Camera not aligned", fg = "red") # Label aligned aktualisieren
            self.KPI_pose_time_label.config(text=f"Incorrect Pose Time: {self.time_incorrect_pose}")

        #img = Image.fromarray(video_stream(self.cap, video))
        img = Image.fromarray(video)
        imgtk = ImageTk.PhotoImage(image=img)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)
        self.photo = imgtk  # Keep a reference to avoid garbage collection
        self.master.after(10, self.update)

        """
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
            if self.detecting:
                results = self.pose.process(frame)
                if results.pose_landmarks:
                    self.draw_landmarks(frame, results.pose_landmarks)

                    # Here would be my custom MediaPipe analysis
                    # Example: analyze specific landmarks, calculate angles, etc.
                    # custom_analysis(results.pose_landmarks)

            # Convert the image to PIL format
            img = Image.fromarray(frame)
            imgtk = ImageTk.PhotoImage(image=img)
            self.canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)
            self.photo = imgtk  # Keep a reference to avoid garbage collection
        self.master.after(10, self.update)
        """

    def draw_landmarks(self, frame, landmarks):
        mp_drawing = mp.solutions.drawing_utils
        mp_drawing.draw_landmarks(frame, landmarks, self.mp_pose.POSE_CONNECTIONS)

    def start_detection(self):
        self.detecting = True
        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)

        # Update the class property in UI
        self.pose_score += 1
        self.KPI_pose_score_label.config(text=f"Pose Score: {self.pose_score}") 

    def stop_detection(self):
        self.detecting = False
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)

def main():
    try:
        root = tk.Tk()
        app = LandmarkDetectorApp(root)
        root.mainloop()
    except Exception as e:
        print("An error occurred:")
        print(e)
        print(traceback.format_exc())

if __name__ == "__main__":
    main()
