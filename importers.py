# -*- coding: utf-8 -*-

import os
import shutil
from typing import Optional

from pfe.common.configuration.system import System
from pfe.common.kif.extended_image import ExtendedKifImage
from pfe.common.utils.db.history import History
from pfe.common.utils.image.image_grab.image_grab import ImageGrabConfiguration
from pfe.interfaces.common.exporter.convert import Convert
from pfe.services.image_grab.database import DB, Conversions, check_current_history_count
from pfe.services.image_grab.definitions import ImageGrabData
from pfe.services.image_grab.files.file import FileFactory, FF_FM_ALERT, FF_USER_IMPORTED
from pfe.services.image_grab.files.functions import update_file_permissions
from pfe.services.image_grab.files.images_watch import KIFInfo
from pfe.services.image_grab.fm_alert.data import Trigger, BoundingBox, FMAlertDBInfo
from pfe.services.image_grab.ice import ice_image_grab
from pfe.services.image_grab.loggers import get_main_logger
from pfe.services.image_grab.observers import publish_history_change
from pfe.services.image_grab.overlays.request import Files, FMRequest, Callbacks
from pfe.services.image_grab.overlays.simulator import Simulator
from pfe.services.image_grab.project_info import ProjectInfo
from pfe.services.image_grab.utils import cleanup


class Import(object):
    def __init__(self):
        super(Import, self).__init__()

    @staticmethod
    def __update_file_permissions(path):
        # noinspection PyBroadException
        try:
            update_file_permissions(path)
        except Exception:
            get_main_logger().warning("Failed updating file permissions of {}.".format(path))

    @staticmethod
    def fm_alert(kif_image: ExtendedKifImage, trigger: Trigger, bounding_box: BoundingBox, simulator: Optional[Simulator]) -> int:
        get_main_logger().debug("Importing FM alerted KIF image:")
        get_main_logger().debug("   Trigger:")
        get_main_logger().debug("      pipeline: {}".format(trigger.pipeline))
        get_main_logger().debug("      attribute: {}".format(trigger.attribute))
        get_main_logger().debug("      attribute name: {}".format(trigger.attribute_name))
        get_main_logger().debug("      algorithm: {}".format(trigger.algorithm))
        get_main_logger().debug("      algorithm name: {}".format(trigger.algorithm_name))
        get_main_logger().debug("      threshold_index: {}".format(trigger.threshold_index))
        get_main_logger().debug("   Bounding box:")
        get_main_logger().debug("      start_pixel: {}".format(bounding_box.start_pixel))
        get_main_logger().debug("      end_pixel: {}".format(bounding_box.end_pixel))
        get_main_logger().debug("      start_line: {}".format(bounding_box.start_line))
        get_main_logger().debug("      end_line: {}".format(bounding_box.end_line))

        local_simulator = simulator is None

        # Create simulator if needed, and reload to get current project info
        if local_simulator:
            simulator = Simulator(identifier=ImageGrabData.LOCAL_SIMULATOR)
            simulator.reload()
            with simulator.blocking_simulator():
                pass  # force reload

        sensor_id = trigger.pipeline
        sensor_name = ImageGrabConfiguration.sensor_by_id(sensor_id).name()
        with DB() as db:
            history_id = db.add(
                sensor_id=sensor_id,
                sensor_name=sensor_name,
                saved=1,
                triggered=1,
                fm_alert=1,
                triggered_by=ice_image_grab.TriggerType.FMALERT.value,
                trigger_thresholds=Conversions.trigger_threshold_tuples_to_db(
                    (trigger.attribute, trigger.algorithm, trigger.threshold_index)
                ),
                project_uuid="",
                project_vpf_uuid=simulator.vpf_uuid,
                project_rp_uuid=simulator.rp_uuid,
                project_name=simulator.recipe_name,
                data=dict()
            )

            #############################
            # CHECK MAX FM ALERT IMAGES #
            #############################

            num_fm_alerted_images = db.count(saved=1, triggered_by=ice_image_grab.TriggerType.FMALERT.value)
            if num_fm_alerted_images + 1 > ImageGrabData.FM_ALERT_MAX_IMAGES:
                get_main_logger().debug("Maximum number of FM alerted KIF images reached,"
                                        "going to purge oldest.")

                result = list(db.ordered_by([History.TIMESTAMP_COLUMN], asc=True, limit=1,
                                            saved=1, triggered_by=ice_image_grab.TriggerType.FMALERT.value))
                if result is not None:
                    item_id = getattr(result[0], History.ID_COLUMN)
                    get_main_logger("Purging item with id '{}'".format(item_id))
                    cleanup.remove(item_id, do_publish=True)

            get_main_logger().debug("Importing FM alerted KIF image: history id = {}".format(history_id))

        #############
        # KIF IMAGE #
        #############
        kif_image_to_save = bounding_box.extract_from_kif(kif_image) if ImageGrabData.FM_ALERT_USE_CROP else kif_image
        kif_file = FileFactory.create(ImageGrabData.KIF, history_id, flags=FF_FM_ALERT, project_name=simulator.recipe_name, sensor_name=sensor_name)
        kif_file_path = kif_file.path(create_directories=True)
        target_directory, img_name = os.path.split(kif_file_path)
        kif_image_to_save.name, _ = os.path.splitext(img_name)
        kif_image_to_save.save(target_directory)
        Import.__update_file_permissions(kif_file_path)

        ##############
        # SIMULATION #
        ##############

        files = Files()
        files.add(ImageGrabData.RAW_IMAGE,
                  FileFactory.create(ImageGrabData.RAW_IMAGE, history_id, project_name=simulator.recipe_name, sensor_name=sensor_name, flags=FF_FM_ALERT))
        files.add(ImageGrabData.FM_SEGMENTED,
                  FileFactory.create(ImageGrabData.FM_SEGMENTED, history_id, project_name=simulator.recipe_name, sensor_name=sensor_name))

        callbacks = Callbacks()

        def __raw_image_created(success: bool):
            if success:
                DB.update_img(
                    history_id,
                    ImageGrabData.RAW_IMAGE,
                    project_checksum="",
                    project_name=simulator.recipe_name
                )

        def __fm_segmented_created(success: bool):
            if not success:
                return

            with DB() as db__:
                db__.update_data(history_id, FMAlertDBInfo(bounding_box, trigger.attribute_name, trigger.algorithm_name).to_dict())

            ############################
            # BOUNDING BOX (SVG) IMAGE #
            ############################
            bb_file__ = FileFactory.create(ImageGrabData.FM_BOUNDING_BOX, history_id, project_name=simulator.recipe_name, sensor_name=sensor_name)
            bb_path__ = bb_file__.path(create_directories=True)
            bounding_box.create_svg(kif_image, bb_path__)
            Import.__update_file_permissions(bb_path__)

            # Finally push history changed event
            publish_history_change(history_id, ice_image_grab.TriggerType.FMALERT, ice_image_grab.HistoryChangedEvent.ADDED,
                                   simulator.project_info.get_compatible_sensor_group_uuids(kif_file_path))

        callbacks.add(ImageGrabData.RAW_IMAGE, __raw_image_created)
        callbacks.add(ImageGrabData.FM_SEGMENTED, __fm_segmented_created)

        simulation_request = FMRequest(kif_file_path, trigger, files, callbacks)

        if local_simulator:
            with simulator.blocking_simulator() as processor:
                processor(simulation_request)
        else:
            simulator.push(simulation_request)

        return history_id

    @staticmethod
    def from_file(kif_path: str, project_info: ProjectInfo, available_to_all: bool = True, save: bool = False) -> int:
        get_main_logger().debug("Importing KIF image from file:")
        get_main_logger().debug("   Path: {}".format(kif_path))
        check_current_history_count(project_info.project_uuid, None, True)

        with DB() as db:
            sensor_id = -1
            sensor_name = "UNKNOWN"
            # noinspection PyBroadException
            try:
                #
                # Try to resolve sensor id and name from meta data
                #
                image = ExtendedKifImage(image_path=kif_path)
                sensor_name = image.meta_data.sensor_name
                sensor = System().get_sensor_by_name(sensor_name)
                if sensor is not None:
                    sensor_id = sensor.pipeline()
            except Exception:
                get_main_logger().exception("Failed to resolve sensor id and/or name from KIF meta data.")

            ############
            # DATABASE #
            ############

            history_id = db.add(
                sensor_id=sensor_id,
                sensor_name=sensor_name,
                saved=1 if save else 0,
                triggered=0,
                fm_alert=0,
                triggered_by=ice_image_grab.TriggerType.FROMFILE.value,
                project_rp_uuid=project_info.project_uuid,
                project_name=project_info.recipe_name,
                data=dict(original_path=kif_path),
                available_to_all=1 if available_to_all else 0,
                user_imported=1
            )

            #######
            # KIF #
            #######

            destination = FileFactory.create(
                ImageGrabData.KIF, history_id,
                flags=FF_USER_IMPORTED,
                project_name=project_info.recipe_name,
                sensor_name=sensor_name,
            )

            shutil.copy(kif_path, destination.path(True))

            Import.__update_file_permissions(kif_path)

            #######
            # PNG #
            #######

            display_options = project_info.get_display_options(KIFInfo.create(kif_path))

            raw_image_file = FileFactory.create(
                ImageGrabData.RAW_IMAGE, history_id,
                flags=FF_USER_IMPORTED,
                project_name=project_info.recipe_name,
                sensor_name=sensor_name
            )
            full_raw_image_path = raw_image_file.path(create_directories=True)

            r, g, b = display_options.get_rgb()
            Convert.kif_to_png(kif_path, full_raw_image_path, display_options.display_method, r, g, b)

            Import.__update_file_permissions(full_raw_image_path)

            # update RAW_IMAGE fields too
            get_main_logger().debug("Updating img db info.")
            DB.update_img(
                history_id,
                ImageGrabData.RAW_IMAGE,
                project_name=project_info.recipe_name
            )

            publish_history_change(history_id, ice_image_grab.TriggerType.FROMFILE, ice_image_grab.HistoryChangedEvent.ADDED,
                                   project_info.get_compatible_sensor_group_uuids(kif_path))
            return history_id
