import json

import requests
from ConfigReader import ConfigReader


class RasaClient:
    def __init__(self):
        self.config = ConfigReader().get_config()
        self.rasa_url = self.config['api_url']
        self.rasa_user = self.config['username']

        print(self.rasa_url)
        print(self.rasa_user)

    def send_message(self, message):
        payload = {
            "sender": self.rasa_user,
            "message": message
        }
        try:
            response = requests.post(self.rasa_url, json=payload)
            if response.status_code == 200:
                data = json.loads(response.text)
                response_text = data[0]['text']
                return response_text
            else:
                return "Es konnte keine Verbindung mit Rasa hergestellt werden"
        except requests.exceptions.RequestException as e:
            return f"Anfrage an Rasa konnte nicht verarbeitet werden: {e}"
    