import os

import yaml


class ConfigReader:
    def __init__(self, config_file='config.yml'):
        self.config_file = config_file
        self.config = None

    def read_config(self):
        file_path = os.path.join('ChatClient', self.config_file)
        with open(file_path, 'r') as file:
            self.config = yaml.safe_load(file)

    def get_config(self):
        if self.config is None:
            self.read_config()
        return self.config
