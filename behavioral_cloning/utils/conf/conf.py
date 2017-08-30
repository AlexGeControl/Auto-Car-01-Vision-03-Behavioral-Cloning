# Set up session:
import json

class Conf:
    def __init__(self, conf_filename):
        conf = json.load(
            open(conf_filename)
        )
        self.__dict__.update(conf)

    def __getitem__(self, key):
        return self.__dict__.get(key, None)
