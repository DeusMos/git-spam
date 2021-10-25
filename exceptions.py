# -*- coding: utf-8 -*-
from pfe.services.image_grab.definitions import ImageGrabData


class ImageGrabTimedOut(Exception):
    def __init__(self, seconds):
        super(ImageGrabTimedOut, self).__init__("Image grab timed out, timeout set to {} seconds.".format(
            seconds
        ))


class ImageGrabCancelled(Exception):
    def __init__(self):
        super(ImageGrabCancelled, self).__init__("Image grab cancelled.")


class ImageGrabFailed(Exception):
    def __int__(self, message):
        super(ImageGrabFailed, self).__init__(message)


class FMAlertMaxImagesReached(Exception):
    def __int__(self):
        super(FMAlertMaxImagesReached, self).__init__("FM Alert: Maximum number ({}) of images reached!".format(
            ImageGrabData.FM_ALERT_MAX_IMAGES
        ))


class SimulationNotActive(Exception):
    def __init__(self):
        super(SimulationNotActive, self).__init__("Simulation: simulation is not active.")
