# -*- coding: utf-8 -*-

import time

from pfe.common.ice.ice_manager import IceConfig, ice_manager
from pfe.services.image_grab.loggers import get_main_logger

IceConfig().load_slices_image_grab()
# noinspection PyUnresolvedReferences
import fe.services.imagegrab.slice as ice_image_grab
# noinspection PyUnresolvedReferences
import fe.services.recipes.slice as ice_recipes
# noinspection PyUnresolvedReferences
import fe.services.discovery.slice as ice_discovery
# noinspection PyUnresolvedReferences
import fe.services.plc.slice as ice_plc


#
# Log on Ice, allow multiple tries,
# to make sure all necessary services are up (e.g. permissions)
#
# Maximum timeout currently set at 15 seconds
#
def do_ice_logon():
    # timeout (seconds)
    timeout = 15
    while timeout > 0:
        ice_manager().log_on('service', 'l@s3rs0rt3r')
        if ice_manager().is_logged_on():
            break
        time.sleep(1)
        timeout -= 1

    if not ice_manager().is_logged_on():
        raise ice_image_grab.RunTimeError("Logging on failed!")
    get_main_logger().debug("Successfully logged on Ice.")
