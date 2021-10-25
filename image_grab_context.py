# -*- coding: utf-8 -*-

from enum import Enum

from pfe.services.image_grab.utils.data import Data


SIMULATION_DIRTY = "simulation_dirty"
NEW_DEFECT_SENSOR_GROUP = "new_defect_sensor_group"


class ImageGrabContext(Enum):
    NORMAL = 0, {}
    SIMULATION = 1, {
        SIMULATION_DIRTY: False
    }
    NEW_DEFECT = 2, {
        NEW_DEFECT_SENSOR_GROUP: None
    }

    def __init__(self, index, default_properties):
        self._value = index
        self._default_properties = dict(default_properties)

    def create_properties(self):
        return Data(dict(self._default_properties))
