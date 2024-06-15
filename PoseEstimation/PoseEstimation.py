# Importieren der benötigten Bibliotheken
import math as m
import time

import cv2
import mediapipe as mp
from PIL import Image, ImageTk

class PoseEstimation:

    # Variablen der Klasse PoseEstimation intialisieren
    def __init__(self, captureObject = False, automode = False):
        # Frame-Zähler initialisieren
        self.good_frames = 0
        self.bad_frames = 0
        self.bad_time = 0
        self.good_time = 0
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose()
        self.error = ""
        self.captureObject = captureObject

    def __call__(self, captureObject, video):
        return self.videoFeedForHMI(captureObject, video)

    # Die Funktion offsetDistance berechnet die absolute Distanz zwischen zwei Koordinatenpunkten
    def offsetDistance (self, x1, y1, x2, y2):
        return m.sqrt((x2-x1)**2+(y2-y1)**2)
    
    # Die Funktion getAngle ermittelt den Winkel zwischen Halswirbelsäule und einer vertikalen Gerade
    def getAngle(self, x1, y1, x2, y2, mode=""):
        try:
            alpha = m.atan(abs(x2-x1)/abs(y2-y1)) # Winkel Alpha berechnen
            return int(180/m.pi) * alpha # Winkel Alpha wird von Radiant in Grad umgerechnet
        except ZeroDivisionError:
            if mode == "lowerarm":
                return 90 # Für den Lowerarm-Winkel ist Winkel 90°
            else:
                return 0 # Bei Division durch 0 ist Winkel 0°

    # Erfassen der Koordinaten der Körperhaltungspunkte 
    def getCoordinates(self, lm, Width, Heigth):
            lmPose  = self.mp_pose.PoseLandmark
            # x und y - Koordinaten der linken Schulter
            l_shldr_x = int(lm.landmark[lmPose.LEFT_SHOULDER].x * Width)
            l_shldr_y = int(lm.landmark[lmPose.LEFT_SHOULDER].y * Heigth)
            
            # x und y - Koordinaten der rechten Schulter
            r_shldr_x = int(lm.landmark[lmPose.RIGHT_SHOULDER].x * Width)
            r_shldr_y = int(lm.landmark[lmPose.RIGHT_SHOULDER].y * Heigth)
            
            # x und y - Koordinaten des linken Ohrs
            l_ear_x = int(lm.landmark[lmPose.LEFT_EAR].x * Width)
            l_ear_y = int(lm.landmark[lmPose.LEFT_EAR].y * Heigth)
            
            # x und y - Koordinaten der linken Hüfte
            l_hip_x = int(lm.landmark[lmPose.LEFT_HIP].x * Width)
            l_hip_y = int(lm.landmark[lmPose.LEFT_HIP].y * Heigth)

            # x und y - Koordinaten des linken Ellenbogens
            l_elbow_x = int(lm.landmark[lmPose.LEFT_ELBOW].x * Width)
            l_elbow_y = int(lm.landmark[lmPose.LEFT_ELBOW].y * Heigth)

            # x und y - Koordinaten des linken Handgelenks
            l_wrist_x = int(lm.landmark[lmPose.LEFT_WRIST].x * Width)
            l_wrist_y = int(lm.landmark[lmPose.LEFT_WRIST].y * Heigth)

            return l_shldr_x, l_shldr_y, r_shldr_x, r_shldr_y, l_ear_x, l_ear_y, l_hip_x, l_hip_y, l_elbow_x, l_elbow_y, l_wrist_x, l_wrist_y
    
    # Kameraausrichtung ermitteln und bewerten
    def AlignCamera(self, Shoulder_Lx, Shoulder_Ly, Shoulder_Rx, Shoulder_Ry):
        # Berechne die Offset-Distanz zwischen den zwei gemessenen Schulterpunkten
        offset = self.offsetDistance(Shoulder_Lx, Shoulder_Ly, Shoulder_Rx, Shoulder_Ry)
                
        if offset < 100:
            return True
        else:
            return False

    # Verbinde Punkte
    def drawline(self, video, start_x, start_y, end_x, end_y, color):
        cv2.line(video, (start_x, start_y), (end_x, end_y), color, 4)

    def videoFeedForHMI(self, captureObject, video_rgb):
        # FPS des Videos messen
        fps = captureObject.get(cv2.CAP_PROP_FPS)
        # Höhe und Breite des Video-Streams erfassen
        Height, Width = video_rgb.shape[:2]

        # Erfasse Körperpunkte und deren Koordinatenpunkte
        keypoints = self.pose.process(video_rgb)
        landmarks = keypoints.pose_landmarks

        # Wenn Koordinatenpunkte gefunden, dann Programm ausführen
        if landmarks != None: 
            Shoulder_Lx, Shoulder_Ly, Shoulder_Rx, Shoulder_Ry, Ear_Lx, Ear_Ly, Hip_Lx, Hip_Ly, Elbow_Lx, Elbow_Ly, Wrist_Lx, Wrist_Ly = self.getCoordinates(landmarks, Width, Height) 

            ############################################# Ausrichten der Kamera #############################################
            aligned = self.AlignCamera(Shoulder_Lx, Shoulder_Ly, Shoulder_Rx, Shoulder_Ry)

            ######################################### Berechne die Neigungswinkel ###########################################
            neck_inclination = self.getAngle(Shoulder_Lx, Shoulder_Ly, Ear_Lx, Ear_Ly)
            torso_inclination = self.getAngle(Hip_Lx, Hip_Ly, Shoulder_Lx, Shoulder_Ly)
            lowerarm_inclination = self.getAngle(Elbow_Lx, Elbow_Ly, Wrist_Lx, Wrist_Ly, "lowerarm")

            ##################################### Einzeichnen der Punkte in das Video #######################################
            cv2.circle(video_rgb, (Shoulder_Lx, Shoulder_Ly), 7, (0, 255, 255), -1) # Zeichne Kreis in Farbe Gelb
            cv2.circle(video_rgb, (Ear_Lx, Ear_Ly), 7, (0, 255, 255), -1) # Zeichne Kreis in Farbe Gelb
            cv2.circle(video_rgb, (Shoulder_Rx, Shoulder_Ry), 7, (255, 0, 255), -1) # Zeichne Kreis in Farbe Pink
            cv2.circle(video_rgb, (Hip_Lx, Hip_Ly), 7, (0, 255, 255), -1) # Zeichne Kreis in Farbe Gelb
            cv2.circle(video_rgb, (Elbow_Lx, Elbow_Ly), 7, (0, 255, 255), -1) # Zeichne Kreis in Farbe Gelb
            cv2.circle(video_rgb, (Wrist_Lx, Wrist_Ly), 7, (0, 255, 255), -1) # Zeichne Kreis in Farbe Gelb

            ############################### Bewertung der Haltung anhand der Messungen ###################################### -> hier Linien einzeln einfärben!
            self.good_frames = 0
            self.bad_frames = 0

            if neck_inclination < 30: # Gute Haltung -> Farbe Grün
                self.drawline(video_rgb, Shoulder_Lx, Shoulder_Ly, Ear_Lx, Ear_Ly, (127, 255, 0))
                self.good_frames += 1
            elif neck_inclination >= 30 and neck_inclination <= 40: # Mittelmäßige Haltung -> Farbe Orange
                self.drawline(video_rgb, Shoulder_Lx, Shoulder_Ly, Ear_Lx, Ear_Ly, (255,165,0))
                self.good_frames += 1
            else: # Schlechte Haltung -> Farbe Rot
                self.drawline(video_rgb, Shoulder_Lx, Shoulder_Ly, Ear_Lx, Ear_Ly, (255, 0, 0))
                self.bad_frames += 1

            if torso_inclination < 10: # Gute Haltung -> Farbe Grün
                self.drawline(video_rgb, Shoulder_Lx, Shoulder_Ly, Hip_Lx, Hip_Ly, (127, 255, 0))
                self.good_frames += 1
            elif torso_inclination >=10 and torso_inclination < 15: # Mittelmäßige Haltung -> Farbe Orange
                self.drawline(video_rgb, Shoulder_Lx, Shoulder_Ly, Hip_Lx, Hip_Ly, (255,165,0))
                self.good_frames += 1
            else: # Schlechte Haltung -> Farbe Rot
                self.drawline(video_rgb, Shoulder_Lx, Shoulder_Ly, Hip_Lx, Hip_Ly, (255, 0, 0))
                self.bad_frames += 1

            if lowerarm_inclination < 100 and lowerarm_inclination > 80: # Gute Haltung -> Farbe Grün
                self.drawline(video_rgb, Shoulder_Lx, Shoulder_Ly, Elbow_Lx, Elbow_Ly, (127, 255, 0))
                self.drawline(video_rgb, Wrist_Lx, Wrist_Ly, Elbow_Lx, Elbow_Ly, (127, 255, 0))
                self.good_frames += 1
            elif (lowerarm_inclination >= 100 and lowerarm_inclination <= 120) or (lowerarm_inclination >= 60 and lowerarm_inclination <= 80): # Mittelmäßige Haltung -> Farbe Orange
                self.drawline(video_rgb, Shoulder_Lx, Shoulder_Ly, Elbow_Lx, Elbow_Ly, (255,165,0))
                self.drawline(video_rgb, Wrist_Lx, Wrist_Ly, Elbow_Lx, Elbow_Ly, (255,165,0))
                self.good_frames += 1
            else: # Schlechte Haltung -> Farbe Rot
                self.drawline(video_rgb, Shoulder_Lx, Shoulder_Ly, Elbow_Lx, Elbow_Ly, (255, 0, 0))
                self.drawline(video_rgb, Wrist_Lx, Wrist_Ly, Elbow_Lx, Elbow_Ly, (255, 0, 0))
                self.bad_frames += 1

            if self.good_frames == 3:
                correctPose = True
                self.good_time = (1 / fps)
                self.bad_time = 0
            else:
                correctPose = False
                self.bad_time =  (1 / fps)
                self.good_time = 0
            
            return [video_rgb, correctPose, self.good_time, self.bad_time, aligned, False] # error = False
        else:
            self.error = "Fehler: Es konnten keine Koerperpunkte erkannt werden. Bitte Kamera erneut ausrichten"
            print("Fehler: Es konnten keine Koerperpunkte erkannt werden")

            # Attribute zurücksetzen für Abbild in HMI
            self.good_time = 0
            self.bad_time = 0
            
            return [video_rgb, False, self.good_time, self.bad_time, False, self.error] # correctPose = False | aligned = False