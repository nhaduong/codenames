#!/usr/bin/env python
# coding=utf-8


import json
import os
import os.path
import sys
from pkg_resources import resource_string

#CONFIG_FILE = os.path.dirname(__file__) + "/config.json"
CONFIG_FILE = resource_string(__name__, 'config.json')


class Config(object):
    def __init__(self):
        #print(CONFIG_FILE)
        config_path = CONFIG_FILE
        # if not os.path.isfile(config_path):
        #     print('Error: can\'t find config file {0}.'.format(config_path))
        #     sys.exit(1)

        # noinspection PyBroadException
        try:
            # with open(config_path, 'r') as f:
            parameters = json.loads(config_path)

            self.__dict__ = parameters
        except Exception:
            print('Error: can\'t parse config file {0}.'.format(config_path))
            sys.exit(1)


config = Config()
