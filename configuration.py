# -*- coding: utf-8 -*-

import os

from six.moves import configparser

from pfe.services.image_grab.loggers import get_main_logger


# noinspection PyClassicStyleClass
class ImageGrabConfiguration(configparser.ConfigParser):
    DEFAULTS = {}

    def __init__(self, config_path):
        configparser.ConfigParser.__init__(self, ImageGrabConfiguration.DEFAULTS)
        self._config_path = config_path

    def load(self):
        if os.path.isfile(self._config_path):
            self.read(self._config_path)
        return self

    def _option(self, config):
        section, option = config.split('.')
        if self.has_option(section, option):
            return self.get(section, option)
        return None

    def __call__(self, config, default):
        v = self._option(config)
        return default if v is None else v

    def list(self, config, default):
        v = self._option(config)
        if v is not None:
            try:
                return v.split(",")
            except ValueError as e:
                get_main_logger().debug("Failed to get option '{}', conversion to list failed:".format(config))
                get_main_logger().exception(e)
                get_main_logger().debug("Falling back to default: {}".format(str(default)))
        return default

    def _get_type(self, config, default, method, type_str):
        try:
            section, option = config.split('.')
            if self.has_option(section, option):
                return method(section, option)
        except ValueError as e:
            get_main_logger().debug("Failed to get option '{}', conversion to {} failed:".format(config, type_str))
            get_main_logger().exception(e)
            get_main_logger().debug("Falling back to default.")
        return default

    def int(self, config, default):
        return self._get_type(config, default, self.getint, 'int')

    def float(self, config, default):
        return self._get_type(config, default, self.getfloat, 'float')

    def boolean(self, config, default):
        return self._get_type(config, default, self.getboolean, 'boolean')

    l = list
    i = int
    f = float
    b = boolean
