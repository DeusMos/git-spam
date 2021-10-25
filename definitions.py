# -*- coding: utf-8 -*-

import os
from grp import getgrnam
from pwd import getpwnam
from typing import Optional

from pfe.services.image_grab import CONFIG
from pfe.services.image_grab.configuration import ImageGrabConfiguration

# Some values can be overridden by the user ('/etc/fe/image_grab.conf'):
_ = ImageGrabConfiguration(CONFIG).load()


ALL_ATTRIBUTES = list(range(8))


class ImageGrabData(object):
    # Max length of items (history) for each project
    MAX_HISTORY = _.i("Images.max_save_count", 100)
    MAX_UNSAVED_SINGLE_GRAB_IMAGES_PER_SENSOR = _.i("Images.max_unsaved_single_grab_images_per_sensor", 1)
    MAX_UNSAVED_MULTI_GRAB_IMAGES_PER_SENSOR = _.i("Images.max_unsaved_multi_grab_images_per_sensor", 6)

    # Scan count
    SCAN_COUNT_DEFAULT = _.i("Grab.scan_count_default", 1024)
    SCAN_COUNT_CROPPED = _.i("Grab.scan_count_cropped", 128)

    # Db
    HISTORY_DB_PATH = _("Db.path", "/var/fe/image_grab/image_grab_history.db")

    # Images
    IMAGE_ROOT_PATH = _("Images.root_path", "/home/xp/ImagesFromUI/")
    IMAGE_EXTRA_PATHS = _.l("Images.extra_paths", ["/var/fe/image_grab/images/"])
    # IMAGE_LINKER_ROOT_PATH = "/home/xp/Images/FromUI/"

    # Recipes
    RECIPE_ROOT_PATH = _("Recipes.root_path", "/var/fe/image_grab/recipes/")
    RECIPE_FILENAME = "recipe.rp"
    RECIPE_INFO_FILENAME = "recipe.json"
    RECIPE_CURRENT = os.path.join(RECIPE_ROOT_PATH, 'current.rp')
    RECIPE_CHECKSUM = 'CHECKSUM'
    RECIPE_UUID = 'UUID'
    RECIPE_RP_UUID = 'RP_UUID'
    RECIPE_NAME = 'NAME'
    RECIPE_CURRENT_INFO = os.path.join(RECIPE_ROOT_PATH, 'current.json')

    USER_OWNER_IMAGES = getpwnam(_("Images.user", 'xp')).pw_uid
    GROUP_OWNER_IMAGES = getgrnam(_("Images.group", 'xp')).gr_gid

    # Image grab timeouts (seconds)
    GRAB_TIMEOUT = _.i("Grab.timeout", 120)
    TRIGGERED_GRAB_TIMEOUT = _.i("Grab.triggered_timeout", 300)

    # Publisher topic
    TOPIC = "READY"
    HISTORY_CHANGED_TOPIC = "history.changed"
    TRIGGERED_TOPIC = "grab_image_with_trigger.triggered"
    PROGRESS_TOPIC = "grab_image.progress"
    SIMULATION_TOPIC = "grab_image.simulation"
    SIMULATION_INVALIDATED_TOPIC = "grab_image.simulation_invalidated"

    # Image types
    IMAGE_TYPES = (

        RAW_IMAGE,
        SEGMENTATION_OVERLAY,
        OVERALL_SEGMENTATION_OVERLAY,
        EJECTS_OVERLAY,
        OVERALL_EJECTS_OVERLAY,
        CONTOURS_OVERLAY,
        OVERALL_CONTOURS_OVERLAY,
        ALGORITHMS_OVERLAY,
        OVERALL_ALGORITHMS_OVERLAY,
        BACKGROUND_OVERLAY,
        SELECTED_REGION_OVERLAY
    ) = (

        "RAW_IMAGE",
        "SEGMENTATION_OVERLAY",
        "OVERALL_SEGMENTATION_OVERLAY",
        "EJECTS_OVERLAY",
        "OVERALL_EJECTS_OVERLAY",
        "CONTOURS_OVERLAY",
        "OVERALL_CONTOURS_OVERLAY",
        "ALGORITHMS_OVERLAY",
        "OVERALL_ALGORITHMS_OVERLAY",
        "BACKGROUND_OVERLAY",
        "SELECTED_REGION_OVERLAY",
    )

    ALL_OVERLAYS = "ALL_OVERLAYS"

    KIF = "kif"
    SEGMENTATION_BASE = "segmentation_base"
    FILTERED_BASE = "filtered_base"

    # FM specific
    FM_SEGMENTED = "fm_segmented"
    FM_BOUNDING_BOX = "fm_bounding_box"

    IMAGE_ALL_TYPES = (KIF,
                       SEGMENTATION_BASE,
                       FILTERED_BASE,
                       FM_SEGMENTED,
                       FM_BOUNDING_BOX
                       ) + IMAGE_TYPES

    # Image names (used to store)

    IMAGE_NAMES = {
        KIF: ('raw_image', 'kif'),
        SEGMENTATION_BASE: ('segmentation_base', 'png'),
        FILTERED_BASE: ('filtered_base', 'png'),
        RAW_IMAGE: ('raw_image', 'png'),
        SEGMENTATION_OVERLAY: ('segmentation', 'png'),
        OVERALL_SEGMENTATION_OVERLAY: ('overall_segmentation', 'png'),
        EJECTS_OVERLAY: ('ejects', 'svg'),
        OVERALL_EJECTS_OVERLAY: ('overall_ejects', 'svg'),
        CONTOURS_OVERLAY: ('contours', 'svg'),
        OVERALL_CONTOURS_OVERLAY: ('overall_contours', 'svg'),
        ALGORITHMS_OVERLAY: ('algorithms', 'svg'),
        OVERALL_ALGORITHMS_OVERLAY: ('overall_algorithms', 'svg'),
        SELECTED_REGION_OVERLAY: ('selected_region_overlay', 'svg'),
        BACKGROUND_OVERLAY: ('background', 'png'),
        FM_SEGMENTED: ('fm_overall_segmentation_image', 'png'),
        FM_BOUNDING_BOX: ('fm_bounding_box_image', 'svg'),
    }

    # Test mode
    IMAGE_GRAB_ENABLED = _.b("General.enable_grab", True)
    PROJECT_SYNC_ENABLED = _.b("General.enable_project_sync", True)
    # If False TEST_RECIPE_* constants are used instead of contacting the recipe service
    RECIPE_SERVICE_ENABLED = _.b("General.enable_recipe_service", True)
    SEGMENT_IMAGE_MULTI_PROCESS_ENABLED = True
    TEST_IMAGE_ROOT_PATH = _("Test.kif_image_dir", "")
    TEST_IMAGES = {
        KIF: os.path.join(TEST_IMAGE_ROOT_PATH, _("Test.kif_image", ""))
    }

    TEST_IMAGE_GRAB_DELAY = _.i("Test.image_grab_delay", 3)
    TEST_IMAGE_GRAB_TRIGGER_DELAY = _.i("Test.image_grab_trigger_delay", 3)

    TEST_RECIPE_PATH = _("Test.recipe_path", "")
    TEST_RECIPE_NAME = _("Test.recipe_name", "")
    TEST_RECIPE_FILENAME = _("Test.recipe_filename", "")

    ########################################################
    #                       CLEANUP                        #
    ########################################################

    CLEANUP_CHECK_CURRENT_RECIPES = _.b("Cleanup.check_current_recipes", False)

    ########################################################
    #                       FM ALERT                       #
    ########################################################

    FM_ALERT_MAX_IMAGES = _.i("FMAlert.max_images", 2000)
    FM_ALERT_USE_CROP = _.b("FMAlert.use_crop", False)
    FM_ALERT_MAX_QUEUE_SIZE = _.i("FMAlert.max_queue_size", 5)
    FM_ALERT_ENABLED = _.b("FMAlert.enabled", True)

    ########################################################
    #                       IMAGES                         #
    ########################################################

    IMAGES_VALIDATION_ENABLED = _.b("Images.validation_enabled", True)

    ########################################################
    #                       SIMULATOR                      #
    ########################################################

    # identifiers
    LIVE_SIMULATOR = "LIVE_SIMULATOR"
    DEFAULT_SIMULATOR = "DEFAULT_SIMULATOR"
    FM_SIMULATOR = "FM_SIMULATOR"
    LOCAL_SIMULATOR = "LOCAL_SIMULATOR"
    NO_SIMULATOR = "NO_SIMULATOR"

    ########################################################
    #                       WATCH                          #
    ########################################################

    # noinspection PyPep8Naming
    @classmethod
    def IMAGES_WATCH_DB_OVERRIDE(cls, directory: str) -> Optional[str]:
        option = "override_db_{}".format(directory.replace("/", "_").strip("_"))
        return _("ImagesWatch.{}".format(option), None)

    ########################################################

    @classmethod
    def img_name(cls, img_type, suffix=None):
        name, extension = cls.IMAGE_NAMES[img_type]
        if suffix is not None:
            return "{}_{}.{}".format(name, suffix, extension)
        else:
            return "{}.{}".format(name, extension)
