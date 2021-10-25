import logging
from typing import List

from pfe.common.ice.loggers import MANAGER_LOGGER, SETTING_LOGGER, MODEL_LOGGER
from pfe.common.kif.loggers import KIF_LOGGER
from pfe.common.logging.file import LogFolder
from pfe.common.logging.logger import setup_console_handler
from pfe.common.utils.loggers import UTILS_DB_LOGGER, UTILS_IMAGE_GRAB_LOGGER
from pfe.interfaces.cltb.gui import CLTB_GUI_LOGGER
from pfe.interfaces.recipe_builder import RECIPE_BUILDER_LOGGER
from pfe.services.image_grab.fm_alert.loggers import FM_ALERT_LOGGER
from pfe.system.models.image_capture import IMAGE_CAPTURE_LOGGER

SERVICES_IMAGE_GRAB_LOGGER = "image_grab_service"

# Default log level to use
LOG_LEVEL = logging.INFO

LOGGERS = {
    KIF_LOGGER: False,
    UTILS_DB_LOGGER: False,
    SERVICES_IMAGE_GRAB_LOGGER: True,
    FM_ALERT_LOGGER: True,
    CLTB_GUI_LOGGER: False,
    RECIPE_BUILDER_LOGGER: True,
    IMAGE_CAPTURE_LOGGER: True,
    UTILS_IMAGE_GRAB_LOGGER: True,
    # Ice
    MANAGER_LOGGER: False,
    SETTING_LOGGER: False,
    MODEL_LOGGER: False
}


def get_main_logger(logger_name=SERVICES_IMAGE_GRAB_LOGGER):
    return logging.getLogger(logger_name)


def attach_file_handlers():
    log_folder = LogFolder('image_grab_service', use_pid=True, file_rotation_count=100)
    main_log = log_folder.all
    main_log.set_level(LOG_LEVEL)
    fm_alert_log = log_folder.f("fm_alert.log")
    fm_alert_log.set_level(LOG_LEVEL)

    for logger, enabled in LOGGERS.items():
        if not enabled:
            continue
        if logger == FM_ALERT_LOGGER:
            fm_alert_log.attach(logger)
        else:
            main_log.attach(logger)

    return log_folder.log_folder


def setup_loggers():
    for logger, enabled in LOGGERS.items():
        if enabled:
            setup_console_handler(logger, logging_level=LOG_LEVEL)


def set_log_level(log_level):
    for logger_name, enabled in LOGGERS.items():
        if enabled:
            get_main_logger().info("Setting level of logger '{}' to {}.".format(
                logger_name, logging.getLevelName(log_level)))
            logging.getLogger(logger_name).setLevel(log_level)
            handlers = logging.getLogger(logger_name).handlers  # type: List[logging.Handler]
            for handler in handlers:
                handler.setLevel(log_level)
