import tkinter as tk
import traceback

from ChatClient.rasa_client import RasaClient
from frontend.tkinkter_window import LandmarkDetectorApp


def main():
    try:
        rasaclient = RasaClient()
        root = tk.Tk()
        app = LandmarkDetectorApp(root)
        root.mainloop()
    except Exception as e:
        print("An error occurred:")
        print(e)
        print(traceback.format_exc())

if __name__ == "__main__":
    main()