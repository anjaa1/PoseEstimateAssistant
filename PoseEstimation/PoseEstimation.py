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
        #self.videoFeedForHMI(captureObject, video)

    def __call__(self, captureObject, video):
        return self.videoFeedForHMI(captureObject, video)

    # Die Funktion offsetDistance berechnet die absolute Distanz zwischen zwei Koordinatenpunkten
    def offsetDistance (self, x1, y1, x2, y2):
        return m.sqrt((x2-x1)**2+(y2-y1)**2)
    
    # Die Funktion getNeckAngle ermittelt den Winkel zwischen Halswirbelsäule und einer vertikalen Gerade
    def getAngle(self, x1, y1, x2, y2):
        theta = m.acos( (y2 -y1)*(-y1) / (m.sqrt((x2 - x1)**2 + (y2 - y1)**2 ) * y1) )
        return int(180/m.pi)*theta
    
    # Erfassen der Koordinaten der Körperhaltungspunkte 
    def getCoordinates(self, keypoints, Width, Heigth):
        lm = keypoints.pose_landmarks
        if lm:
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
        
        else:
            self.error = "Fehler: Es konnten keine Koerperpunkte erkannt werden. Bitte Kamera erneut ausrichten"
            print("Fehler: Es konnten keine Koerperpunkte erkannt werden")
    
    # Kameraausrichtung ermitteln und bewerten
    def AlignCamera(self, Shoulder_Lx, Shoulder_Ly, Shoulder_Rx, Shoulder_Ry, video, Width):
        # Berechne die Offset-Distanz zwischen den zwei gemessenen Schulterpunkten
        offset = self.offsetDistance(Shoulder_Lx, Shoulder_Ly, Shoulder_Rx, Shoulder_Ry)
        aligned = False
                
        if offset < 100:
            return True
        else:
            return False

    # Verbinde Punkte
    def drawline(self, video, start_x, start_y, end_x, end_y, color):
        cv2.line(video, (start_x, start_y), (end_x, end_y), color, 4)

    # Erstelle das Videoerfassungsobjekt (0 -> Standard-Webcam des Systems)
    def videoFeed(self):
        captureObject = cv2.VideoCapture(0)
        while captureObject.isOpened():
            # Zugriff auf Video-Ressourcen der Webcam
            success, video = captureObject.read()
            if not success:
                print("Null.Frames")
                break

            # FPS des Videos messen
            fps = captureObject.get(cv2.CAP_PROP_FPS)
            # Höhe und Breite des Video-Streams erfassen
            Height, Width = video.shape[:2]

            # Transformiere BGR zu RGB
            video_rgb = cv2.cvtColor(video, cv2.COLOR_BGR2RGB)

            # Erhalte Körperpunkte 
            keypoints = self.pose.process(video_rgb)

            # Ermittel die Haupt-Koordinatenpunkte
            Shoulder_Lx, Shoulder_Ly, Shoulder_Rx, Shoulder_Ry, Ear_Lx, Ear_Ly, Hip_Lx, Hip_Ly, Elbow_Lx, Elbow_Ly, Wrist_Lx, Wrist_Ly = self.getCoordinates(keypoints, Width, Height) 

            ############################################# Ausrichten der Kamera #############################################
            self.AlignCamera(Shoulder_Lx, Shoulder_Ly, Shoulder_Rx, Shoulder_Ry, video_rgb, Width)

            ######################################### Berechne die Neigungswinkel ###########################################
            neck_inclination = self.getAngle(Shoulder_Lx, Shoulder_Ly, Ear_Lx, Ear_Ly)
            torso_inclination = self.getAngle(Hip_Lx, Hip_Ly, Shoulder_Lx, Shoulder_Ly)
            lowerarm_inclination = self.getAngle(Elbow_Lx, Elbow_Ly, Wrist_Lx, Wrist_Ly)

            ##################################### Einzeichnen der Punkte in das Video #######################################
            cv2.circle(video_rgb, (Shoulder_Lx, Shoulder_Ly), 7, (0, 255, 255), -1) # Zeichne Kreis in Farbe Gelb
            cv2.circle(video_rgb, (Ear_Lx, Ear_Ly), 7, (0, 255, 255), -1) # Zeichne Kreis in Farbe Gelb
            cv2.circle(video_rgb, (Shoulder_Rx, Shoulder_Ry), 7, (255, 0, 255), -1) # Zeichne Kreis in Farbe Pink
            cv2.circle(video_rgb, (Hip_Lx, Hip_Ly), 7, (0, 255, 255), -1) # Zeichne Kreis in Farbe Gelb
            cv2.circle(video_rgb, (Elbow_Lx, Elbow_Ly), 7, (0, 255, 255), -1) # Zeichne Kreis in Farbe Gelb
            cv2.circle(video_rgb, (Wrist_Lx, Wrist_Ly), 7, (0, 255, 255), -1) # Zeichne Kreis in Farbe Gelb

            ############################### Bewertung der Haltung anhand der Messungen ###################################### -> hier Linien einzeln einfärben!
            if neck_inclination < 20: # Gute Haltung -> Farbe Grün
                self.drawline(video_rgb, Shoulder_Lx, Shoulder_Ly, Ear_Lx, Ear_Ly, (127, 255, 0))
                self.bad_frames = 0
                self.good_frames += 1
            elif neck_inclination >= 20 and neck_inclination <= 40: # Mittelmäßige Haltung -> Farbe Orange
                self.drawline(video_rgb, Shoulder_Lx, Shoulder_Ly, Ear_Lx, Ear_Ly, (255,165,0))
                self.bad_frames = 0
                self.good_frames += 1
            else: # Schlechte Haltung -> Farbe Rot
                self.drawline(video_rgb, Shoulder_Lx, Shoulder_Ly, Ear_Lx, Ear_Ly, (255, 0, 0))
                self.bad_frames += 1
                self.good_frames = 0

            if torso_inclination < 10: # Gute Haltung -> Farbe Grün
                self.drawline(video_rgb, Shoulder_Lx, Shoulder_Ly, Hip_Lx, Hip_Ly, (127, 255, 0))
                self.bad_frames = 0
                self.good_frames += 1
            elif torso_inclination >=10 and torso_inclination < 15: # Mittelmäßige Haltung -> Farbe Orange
                self.drawline(video_rgb, Shoulder_Lx, Shoulder_Ly, Hip_Lx, Hip_Ly, (255,165,0))
                self.bad_frames = 0
                self.good_frames += 1
            else: # Schlechte Haltung -> Farbe Rot
                self.drawline(video_rgb, Shoulder_Lx, Shoulder_Ly, Hip_Lx, Hip_Ly, (255, 0, 0))
                self.bad_frames += 1
                self.good_frames = 0

            if lowerarm_inclination < 100 and lowerarm_inclination > 80: # Gute Haltung -> Farbe Grün
                self.drawline(video_rgb, Shoulder_Lx, Shoulder_Ly, Elbow_Lx, Elbow_Ly, (127, 255, 0))
                self.drawline(video_rgb, Wrist_Lx, Wrist_Ly, Elbow_Lx, Elbow_Ly, (127, 255, 0))
                self.bad_frames = 0
                self.good_frames += 1
            elif (lowerarm_inclination >= 100 and lowerarm_inclination <= 120) or (lowerarm_inclination >= 60 and lowerarm_inclination <= 80): # Mittelmäßige Haltung -> Farbe Orange
                self.drawline(video_rgb, Shoulder_Lx, Shoulder_Ly, Elbow_Lx, Elbow_Ly, (255,165,0))
                self.drawline(video_rgb, Wrist_Lx, Wrist_Ly, Elbow_Lx, Elbow_Ly, (255,165,0))
                self.bad_frames = 0
                self.good_frames += 1
            else: # Schlechte Haltung -> Farbe Rot
                self.drawline(video_rgb, Shoulder_Lx, Shoulder_Ly, Elbow_Lx, Elbow_Ly, (255, 0, 0))
                self.drawline(video_rgb, Wrist_Lx, Wrist_Ly, Elbow_Lx, Elbow_Ly, (255, 0, 0))
                self.bad_frames += 1
                self.good_frames = 0

            # Berechne die Dauer der guten bzw. schlechten Haltung
            self.good_time = (1 / fps) * self.good_frames
            self.bad_time =  (1 / fps) * self.bad_frames

            if self.good_time > 0: # Gute Haltung -> Farbe Grün
                time_string_good = 'Good Posture Time : ' + str(round(self.good_time, 1)) + 's'
                cv2.putText(video_rgb, time_string_good, (10, Height - 20), self.font, 0.9, (127, 255, 0), 2)
            else: # Schlechte Haltung -> Farbe Rot
                time_string_bad = 'Bad Posture Time : ' + str(round(self.bad_time, 1)) + 's'
                cv2.putText(video_rgb, time_string_bad, (10, Height - 20), self.font, 0.9, (255, 0, 0), 2)

            # Transformiere RGB zu BGR
            video_bgr = cv2.cvtColor(video_rgb, cv2.COLOR_RGB2BGR)

            # Zeige das Video in einem PopUp Fenster
            cv2.imshow('Mediapipe Feed', video_bgr)

            # Beenden des PopUp-Fensters mit 'q'
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        captureObject.release() # Freigeben der Webcam Ressourcen
        cv2.destroyAllWindows() # Schließen des PopUp-Fensters

        # Erstelle das Videoerfassungsobjekt (0 -> Standard-Webcam des Systems)
    
    def videoFeedForHMI(self, captureObject, video):
        # FPS des Videos messen
        fps = captureObject.get(cv2.CAP_PROP_FPS)
        # Höhe und Breite des Video-Streams erfassen
        Height, Width = video.shape[:2]

        # Transformiere BGR zu RGB
        video_rgb = cv2.cvtColor(cv2.flip(video,1), cv2.COLOR_BGR2RGB)

        # Erhalte Körperpunkte 
        keypoints = self.pose.process(video_rgb)

        # Ermittel die Haupt-Koordinatenpunkte
        Shoulder_Lx, Shoulder_Ly, Shoulder_Rx, Shoulder_Ry, Ear_Lx, Ear_Ly, Hip_Lx, Hip_Ly, Elbow_Lx, Elbow_Ly, Wrist_Lx, Wrist_Ly = self.getCoordinates(keypoints, Width, Height) 

        ############################################# Ausrichten der Kamera #############################################
        aligned = self.AlignCamera(Shoulder_Lx, Shoulder_Ly, Shoulder_Rx, Shoulder_Ry, video_rgb, Width)

        ######################################### Berechne die Neigungswinkel ###########################################
        neck_inclination = self.getAngle(Shoulder_Lx, Shoulder_Ly, Ear_Lx, Ear_Ly)
        torso_inclination = self.getAngle(Hip_Lx, Hip_Ly, Shoulder_Lx, Shoulder_Ly)
        lowerarm_inclination = self.getAngle(Elbow_Lx, Elbow_Ly, Wrist_Lx, Wrist_Ly)

        ##################################### Einzeichnen der Punkte in das Video #######################################
        cv2.circle(video_rgb, (Shoulder_Lx, Shoulder_Ly), 7, (0, 255, 255), -1) # Zeichne Kreis in Farbe Gelb
        cv2.circle(video_rgb, (Ear_Lx, Ear_Ly), 7, (0, 255, 255), -1) # Zeichne Kreis in Farbe Gelb
        cv2.circle(video_rgb, (Shoulder_Rx, Shoulder_Ry), 7, (255, 0, 255), -1) # Zeichne Kreis in Farbe Pink
        cv2.circle(video_rgb, (Hip_Lx, Hip_Ly), 7, (0, 255, 255), -1) # Zeichne Kreis in Farbe Gelb
        cv2.circle(video_rgb, (Elbow_Lx, Elbow_Ly), 7, (0, 255, 255), -1) # Zeichne Kreis in Farbe Gelb
        cv2.circle(video_rgb, (Wrist_Lx, Wrist_Ly), 7, (0, 255, 255), -1) # Zeichne Kreis in Farbe Gelb

        ############################### Bewertung der Haltung anhand der Messungen ###################################### -> hier Linien einzeln einfärben!

        if neck_inclination < 20: # Gute Haltung -> Farbe Grün
            self.drawline(video_rgb, Shoulder_Lx, Shoulder_Ly, Ear_Lx, Ear_Ly, (127, 255, 0))
            self.bad_frames = 0
            self.good_frames += 1
        elif neck_inclination >= 20 and neck_inclination <= 40: # Mittelmäßige Haltung -> Farbe Orange
            self.drawline(video_rgb, Shoulder_Lx, Shoulder_Ly, Ear_Lx, Ear_Ly, (255,165,0))
            self.bad_frames = 0
            self.good_frames += 1
        else: # Schlechte Haltung -> Farbe Rot
            self.drawline(video_rgb, Shoulder_Lx, Shoulder_Ly, Ear_Lx, Ear_Ly, (255, 0, 0))
            self.bad_frames += 1
            self.good_frames = 0

        if torso_inclination < 10: # Gute Haltung -> Farbe Grün
            self.drawline(video_rgb, Shoulder_Lx, Shoulder_Ly, Hip_Lx, Hip_Ly, (127, 255, 0))
            self.bad_frames = 0
            self.good_frames += 1
        elif torso_inclination >=10 and torso_inclination < 15: # Mittelmäßige Haltung -> Farbe Orange
            self.drawline(video_rgb, Shoulder_Lx, Shoulder_Ly, Hip_Lx, Hip_Ly, (255,165,0))
            self.bad_frames = 0
            self.good_frames += 1
        else: # Schlechte Haltung -> Farbe Rot
            self.drawline(video_rgb, Shoulder_Lx, Shoulder_Ly, Hip_Lx, Hip_Ly, (255, 0, 0))
            self.bad_frames += 1
            self.good_frames = 0

        if lowerarm_inclination < 100 and lowerarm_inclination > 80: # Gute Haltung -> Farbe Grün
            self.drawline(video_rgb, Shoulder_Lx, Shoulder_Ly, Elbow_Lx, Elbow_Ly, (127, 255, 0))
            self.drawline(video_rgb, Wrist_Lx, Wrist_Ly, Elbow_Lx, Elbow_Ly, (127, 255, 0))
            self.bad_frames = 0
            self.good_frames += 1
        elif (lowerarm_inclination >= 100 and lowerarm_inclination <= 120) or (lowerarm_inclination >= 60 and lowerarm_inclination <= 80): # Mittelmäßige Haltung -> Farbe Orange
            self.drawline(video_rgb, Shoulder_Lx, Shoulder_Ly, Elbow_Lx, Elbow_Ly, (255,165,0))
            self.drawline(video_rgb, Wrist_Lx, Wrist_Ly, Elbow_Lx, Elbow_Ly, (255,165,0))
            self.bad_frames = 0
            self.good_frames += 1
        else: # Schlechte Haltung -> Farbe Rot
            self.drawline(video_rgb, Shoulder_Lx, Shoulder_Ly, Elbow_Lx, Elbow_Ly, (255, 0, 0))
            self.drawline(video_rgb, Wrist_Lx, Wrist_Ly, Elbow_Lx, Elbow_Ly, (255, 0, 0))
            self.bad_frames += 1
            self.good_frames = 0

        # Berechne die Dauer der guten bzw. schlechten Haltung
        self.good_time += (1 / fps) * self.good_frames
        self.bad_time +=  (1 / fps) * self.bad_frames

        #print("Good Frames: ", self.good_frames, " -> Time: ", good_time)
        #print("Bad Frames: ", self.bad_frames, " -> Time: ", bad_time)

        return video_rgb, self.good_time, self.bad_time, aligned

"""
# Der Teil nur für Einzelausführung der Python-Datei relevant

def main():
    # Initialize mediapipe pose class.
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()
    PoseEstimation(mp_pose, pose)

if __name__ == "__main__":
    main()
"""