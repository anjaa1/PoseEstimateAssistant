import tkinter as tk
import traceback

from ChatClient.rasa_client import RasaClient
from frontend.tkinkter_window import LandmarkDetectorApp
from PoseEstimation.PoseEstimation import PoseEstimation


def main():
    try:
        rasaclient = RasaClient()
        poseEstimation = PoseEstimation()
        root = tk.Tk()
        app = LandmarkDetectorApp(root, poseEstimation_model=poseEstimation, rasaClient=rasaclient)
        root.mainloop()
    except Exception as e:
        print("An error occurred:")
        print(e)
        print(traceback.format_exc())

if __name__ == "__main__":
    main()