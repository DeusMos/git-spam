# -*- coding: utf-8 -*-

from pfe.services.image_grab.loggers import SERVICES_IMAGE_GRAB_LOGGER


def get_all_loggers():
    return [SERVICES_IMAGE_GRAB_LOGGER]


CONFIG = '/etc/fe/image_grab.conf'


def set_config(path):
    global CONFIG
    CONFIG = path
