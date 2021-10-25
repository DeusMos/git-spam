# -*- coding: utf-8 -*-
import ctypes
import datetime
import json
import logging
import os
import shutil
import sys
import threading
import time
import uuid
from contextlib import contextmanager
from functools import partial
from io import BytesIO
from multiprocessing import Lock, Value
from tempfile import NamedTemporaryFile
from typing import List, Tuple, Optional, Type, Callable
from zipfile import ZipFile

import Ice

from pfe.common.configuration.system import System
from pfe.common.ice.ice_manager import ice_manager, cleanup_if_ice_manager, Subscriber
from pfe.common.kif.extended_image import ExtendedKifImage
from pfe.common.logging.unhandled_exceptions import UnhandledExceptionHandler
from pfe.common.utils.db.history import History
from pfe.common.utils.display_options import ImageDisplayOptionsBase, IImageDisplayOptions
from pfe.common.utils.image.image_grab.image_grab import ImageGrabConfiguration, TriggerThreshold
from pfe.common.utils.image.svg.svg import SVG
from pfe.common.utils.image.svg.svg_types import Metadata
from pfe.common.utils.image.svg.svg_utils import SVGUtils
from pfe.common.utils.numpy_coordinates_2d import NumpyCoordinates2D
from pfe.interfaces.common.exporter.convert import Convert
from pfe.services.image_grab.ai_addon.ai_addon_trigger_data import AIAddonBox
from pfe.services.image_grab.database import DB, Conversions, check_current_history_count, open_database
from pfe.services.image_grab.definitions import ALL_ATTRIBUTES
from pfe.services.image_grab.definitions import ImageGrabData
from pfe.services.image_grab.files import images_watch
from pfe.services.image_grab.files.definitions import IMAGES_LIBRARY_DB_DIR
from pfe.services.image_grab.files.file import FileFactory, is_non_attribute_format, reverse_lookup, File
from pfe.services.image_grab.files.image_id import ImageId
from pfe.services.image_grab.files.images import Images, ImageCompatibilityInfo
from pfe.services.image_grab.files.images_watch import KIFInfo
from pfe.services.image_grab.files.path import ImagePath
from pfe.services.image_grab.files.remove_files import RemoveFiles
from pfe.services.image_grab.fm_alert.data import FMAlertDBInfo
from pfe.services.image_grab.fm_alert.fm_alert import FMAlertProcess
from pfe.services.image_grab.fm_alert.fm_files import fm_files
from pfe.services.image_grab.grab.grabber import Grabber
from pfe.services.image_grab.grab.request import LiveGrabRequest, TriggeredGrabRequest, Request, \
    SurfaceTriggeredGrabRequest, SensorGrabRequest, AIAddonTriggeredGrabRequest
from pfe.services.image_grab.ice import ice_image_grab, do_ice_logon
from pfe.services.image_grab.image_grab_context import ImageGrabContext, SIMULATION_DIRTY, NEW_DEFECT_SENSOR_GROUP
from pfe.services.image_grab.importers import Import
from pfe.services.image_grab.loggers import get_main_logger, attach_file_handlers, set_log_level, \
    LOG_LEVEL
from pfe.services.image_grab.messages import MSG_OVERLAY_TYPE
from pfe.services.image_grab.observers import RecipeLoaderObserverImpl, RecipeUnloaderObserverImpl, \
    RecipeModifiedObserverImpl, publish_history_change, publish_history_changes, PublishHistoryChanges
from pfe.services.image_grab.overlays.request import OverlayRequest, Files, Callbacks, DisableUpdatesRequest, \
    EnableUpdatesRequest, ColorThresholdUpdate, ShapeThresholdUpdate, ShapeComboConditionUpdate, \
    ShapeThresholdActiveUpdate, RegionInfo, ApplyCustomDefectRequest, UpdateCustomDefectRequest, \
    DeleteCustomDefectRequest
from pfe.services.image_grab.overlays.simulator import Simulator
from pfe.services.image_grab.project_info import ProjectInfo
from pfe.services.image_grab.utils import cleanup
from pfe.services.image_grab.utils.cleanup import clean_history, check_current_recipes
from pfe.services.image_grab.utils.data import Data
from pfe.services.image_grab.utils.debug import dump_all_threads
from pfe.services.image_grab.utils.debug import enable_remote_debug
from pfe.services.image_grab.utils.decorators import log_call, lock_within_context
from pfe.services.image_grab.utils.functions import get_utc_seconds, noop, db_item_to_history_item
from pfe.services.image_grab.utils.progress import ProgressManager, LoadImagesManager, LoadHistoryManager
from pfe.services.image_grab.utils.thread_base import set_thread_name
from pfe.services.image_grab.utils.timing import Timings
from pfe.services.image_grab.utils.usb import USB, make_timestamp_dir
from pfe.services.image_grab.utils.worker_thread import WorkerThread
from pfe.system.models.image_capture.image_capture_model import ImageCaptureHardwareDescription


class SensorInfo(object):
    def __init__(self, sensor_id: int, sensor_group_id: str):
        super(SensorInfo, self).__init__()
        self._sensor_id = sensor_id
        self._sensor_group_id = sensor_group_id

    @property
    def sensor_id(self) -> int:
        return self._sensor_id

    @property
    def sensor_group_id(self) -> str:
        return self._sensor_group_id


class AbortGetHistory(Exception):
    def __init__(self):
        super(AbortGetHistory, self).__init__()


# noinspection PyPep8Naming
class ImageGrabServant(ice_image_grab.ImageGrabInterface):
    STOP_WORKER = "STOP"
    LIVE_SYNC_ID = "live_sync"

    def __init__(self):
        set_thread_name("Servant")

        self._recipe_update_timestamp = Value(ctypes.c_double, 0.)

        get_main_logger().info("[init] Logging on to Ice.")
        do_ice_logon()
        get_main_logger().info("[init] Done logging on to Ice.")

        # Start with a cleanup
        self.__clean_up()

        # Image grabber
        self._grabber = Grabber()
        self._grabber.start()
        #

        #
        # Start up of our images watch threads
        #
        self._images_watchers = dict()
        # noinspection PyTypeChecker
        for image_path in ImagePath:
            watch = images_watch.ImagesWatch(image_path.path)
            watch.watch()
            self._images_watchers[image_path] = watch
            get_main_logger().info("[init] Started watch of '{}'.".format(image_path))

        self._history_worker = WorkerThread("History worker", on_empty_callback=PublishHistoryChanges().allow_publishing)
        self._history_worker.start()

        self._live_simulator = Simulator(identifier=ImageGrabData.LIVE_SIMULATOR)
        self._live_simulator.start()
        self._live_simulator.reload()

        self._simulator = Simulator(identifier=ImageGrabData.DEFAULT_SIMULATOR)
        self._simulator.start()
        self._simulator.reload()

        self._image_grab_context_stack_lock = Lock()
        self._image_grab_context_stack = list()
        self._image_grab_context_stack.append((ImageGrabContext.NORMAL, ImageGrabContext.NORMAL.create_properties()))  # type: List[Tuple[ImageGrabContext, Data]]

        #
        # FM Alert
        # (As stated above, this needs to be done before the service's log on ice.)
        #
        if ImageGrabData.FM_ALERT_ENABLED:
            get_main_logger().info("[init] FM Alert: creating process (Service PID: {}).".format(os.getpid()))
            self._fm_alert_process = FMAlertProcess()
            self._fm_alert_process.start()
            get_main_logger().info("[init] FM Alert: done creating and starting process.")
        else:
            get_main_logger().info("[init] FM Alert not enabled.")
            self._fm_alert_process = None

        # project info
        self._project_info = ProjectInfo()
        self.rebuild_project_info()

        # init observers
        self._recipe_loader_observer = RecipeLoaderObserverImpl(self)
        self._recipe_unloader_observer = RecipeUnloaderObserverImpl(self)
        self._recipe_modified_observer = RecipeModifiedObserverImpl(self)
        self._recipe_loader_observer_subscriber = Subscriber(self._recipe_loader_observer, "recipe_loader_topic")
        self._recipe_unloader_observer_subscriber = Subscriber(self._recipe_unloader_observer, "recipe_unloader_topic")
        self._recipe_modified_observer_subscriber = Subscriber(self._recipe_modified_observer, "recipe_modified_topic")

        enable_remote_debug()

        get_main_logger().info("[init] Image grab service start up complete.")

    def finish(self):
        get_main_logger().info("[finish] Image grab service finish.")
        if self._fm_alert_process is not None:
            get_main_logger().info("[finish] Stopping FM Alert process.")
            self._fm_alert_process.stop()
            self._fm_alert_process.join()

        get_main_logger().info("[finish] Cancel image grabs (if needed).")
        self.cancelGrabImage()

        workers = [
            ("History worker", self._history_worker),
            ("Overlay simulator (live)", self._live_simulator),
            ("Overlay simulator", self._simulator),
            ("Grabber", self._grabber)
        ]
        for worker_name, worker in workers:
            # noinspection PyBroadException
            try:
                get_main_logger().info("[finish] Stopping '{}'.".format(worker_name))
                worker.join()
            except Exception:
                get_main_logger().info("[finish] Failed stopping '{}'.".format(worker_name))

        for image_path, watch in self._images_watchers.items():
            get_main_logger().info("[finish] Stopping watch of '{}'.".format(image_path))
            # noinspection PyBroadException
            try:
                watch.stop()
            except Exception:
                get_main_logger().exception("[finish] Failed stopping watch of '{}'.".format(image_path))

        if os.path.isdir(IMAGES_LIBRARY_DB_DIR):
            get_main_logger().info("[finish] Removing '{}'.".format(IMAGES_LIBRARY_DB_DIR))
            shutil.rmtree(IMAGES_LIBRARY_DB_DIR)
        get_main_logger().info("[finish] Image grab service finish complete.")

    @staticmethod
    def _check_flags(history, flags):
        return list(history.find_where("flags & {} = {}".format(flags, flags)))

    @staticmethod
    def _check_non_cache(history):
        max_ = 1
        #
        get_main_logger().debug("Checking non cache items.")
        non_cache_items = history.where_flags(DB.HISTORY_ITEM_NOCACHE)
        if len(non_cache_items) > max_:
            get_main_logger().debug("Going to delete old non cache items ({}).".format(len(non_cache_items) - max_))
            sorted_items = sorted(non_cache_items, key=lambda item_: getattr(item_, History.TIMESTAMP_COLUMN))
            for item in sorted_items[:-1]:
                id__ = getattr(item, History.ID_COLUMN)
                get_main_logger().debug("Deleting item with history id: {}.".format(id__))
                cleanup.remove(id__, do_publish=False)
        else:
            get_main_logger().debug("No need to delete non cache items.")

    @staticmethod
    def __build_request(sensor_grab_requests: List[SensorGrabRequest], request_type: Type[Request],
                        extra_request_arguments: Optional[List] = None,
                        crop: bool = False,
                        sensor_grab_requests_offset: int = 0,
                        trigger_type: ice_image_grab.TriggerType = ice_image_grab.TriggerType.UNKNOWN) -> (Type[Request], int):
        system = System()
        total_nb_of_frame_buffers_needed = 0
        for sensor_grab_request_index, sensor_grab_request in enumerate(sensor_grab_requests[sensor_grab_requests_offset:]):
            sensor = system.get_sensors_by_pipeline(sensor_grab_request.sensor_id)[0]
            total_nb_of_frame_buffers_needed += ImageCaptureHardwareDescription.channels_to_nb_of_frame_buffers(
                sensor.nb_of_channels())
            if total_nb_of_frame_buffers_needed > ImageCaptureHardwareDescription.NB_OF_FRAME_BUFFERS:
                end = sensor_grab_requests_offset + sensor_grab_request_index
                return request_type(sensor_grab_requests[sensor_grab_requests_offset:end], *extra_request_arguments, crop=crop, trigger_type=trigger_type), end
        return request_type(sensor_grab_requests[sensor_grab_requests_offset:], *extra_request_arguments, crop=crop, trigger_type=trigger_type), len(sensor_grab_requests)

    @classmethod
    def _get_history_ids_to_remove(cls, open_db: DB, sensor_id: int, trigger_type: ice_image_grab.TriggerType, extra_max_shift: int = 0):
        non_saved_items = list(open_db.ordered_by([History.TIMESTAMP_COLUMN], asc=False, sensor_id=sensor_id, saved=0, triggered_by=trigger_type.value))

        if trigger_type in (
                ice_image_grab.TriggerType.MULTIMANUAL,
                ice_image_grab.TriggerType.MULTISCHEDULED,
                ice_image_grab.TriggerType.MULTITRIGGERED,
                ice_image_grab.TriggerType.AIADDON,
        ):
            max_unsaved = ImageGrabData.MAX_UNSAVED_MULTI_GRAB_IMAGES_PER_SENSOR + extra_max_shift
        else:
            max_unsaved = ImageGrabData.MAX_UNSAVED_SINGLE_GRAB_IMAGES_PER_SENSOR + extra_max_shift
        return [getattr(non_saved_item, History.ID_COLUMN) for non_saved_item in non_saved_items[max_unsaved:]]

    def __push_grab_request(self, sensor_info_list: List[SensorInfo], trigger_type: ice_image_grab.TriggerType, request_type: Type[Request],
                            extra_request_arguments: Optional[List] = None, crop: bool = False) -> List[int]:
        num_sensors = len(sensor_info_list)
        max_allowed_sensors = {
            ice_image_grab.TriggerType.UNKNOWN: 0,
            ice_image_grab.TriggerType.MANUAL: 1,
            ice_image_grab.TriggerType.TRIGGERED: 1,
            ice_image_grab.TriggerType.FMALERT: 1,
            ice_image_grab.TriggerType.FROMFILE: 1,
            ice_image_grab.TriggerType.MULTIMANUAL: 6,
            ice_image_grab.TriggerType.MULTISCHEDULED: 6,
            ice_image_grab.TriggerType.MULTITRIGGERED: 6,
            ice_image_grab.TriggerType.AIADDON: 6,
        }[trigger_type]

        assert (num_sensors <= max_allowed_sensors)

        sensor_grab_requests = list()
        for sensor_info in sensor_info_list:

            def __sensor_group_info(sensor_group_uuid__: str) -> Tuple[str, str, IImageDisplayOptions]:
                if sensor_group_uuid__:
                    # (tile_name, sensor_group_name, display_options)
                    return self._project_info.get_tile_name(sensor_group_uuid__), \
                           self._project_info.get_sensor_group_name(sensor_group_uuid__), \
                           self._project_info.get_display_options(sensor_group_uuid__)
                elif trigger_type == ice_image_grab.TriggerType.AIADDON:
                    return 'AI Addon', '', self._project_info.get_display_options(sensor_info.sensor_id)
                else:
                    # (tile_name, sensor_group_name, display_options)
                    return "tile", "sensor_group", self._project_info.get_display_options(None)

            sensor_name = self.getSensorName(sensor_info.sensor_id)
            tile_name, sensor_group_name, display_options = __sensor_group_info(sensor_info.sensor_group_id)

            with DB() as history:
                ids_to_remove = self._get_history_ids_to_remove(history, sensor_info.sensor_id, trigger_type, extra_max_shift=-1)

                db_info = dict(
                    sensor_id=sensor_info.sensor_id,
                    sensor_name=sensor_name,
                    triggered_by=trigger_type.value,
                    flags=0,
                    project_name=self._project_info.recipe_name,
                    tile_name=tile_name,
                    project_rp_uuid=self._project_info.project_uuid,
                    trigger_sensor_group_id=sensor_info.sensor_group_id,
                    trigger_sensor_group_name=sensor_group_name
                )
                db_info[DB.project_name_field(ImageGrabData.RAW_IMAGE)] = self._project_info.recipe_name
                db_info[DB.sensor_group_id_field(ImageGrabData.RAW_IMAGE)] = sensor_info.sensor_group_id
                db_info[DB.sensor_group_name_field(ImageGrabData.RAW_IMAGE)] = sensor_group_name
                #
                history_id = history.add(**db_info)
                history_item = history.get(history_id)

                #
                # We are deleting old unsaved images *after* getting a new history id to avoid old history ids to be used again.
                # So this needs be to *after*
                #     history_id = history.add(**db_info)
                #
                for id_to_remove in ids_to_remove:
                    get_main_logger().info("Removing old unsaved grab with history id {} for sensor id {} (and trigger type value: {}).".format(
                        id_to_remove, sensor_info.sensor_id, trigger_type.value
                    ))
                    cleanup.remove(id_to_remove, do_publish=True)

            common_kwargs = dict(project_name=self._project_info.recipe_name, tile_name=tile_name, sensor_group_name=sensor_group_name, sensor_name=sensor_name)
            kif = FileFactory.create(ImageGrabData.KIF, history_id, **common_kwargs).path(create_directories=True)
            png = FileFactory.create(ImageGrabData.RAW_IMAGE, history_id, **common_kwargs).path(create_directories=True)

            sensor_grab_requests.append(SensorGrabRequest(
                history_id, sensor_info.sensor_group_id, sensor_info.sensor_id, get_utc_seconds(history_item), kif, png, display_options))

        with self._grabber as push:
            extra_request_arguments = list() if extra_request_arguments is None else extra_request_arguments
            # check if we need to split up in multiple requests
            offset = 0
            while offset < len(sensor_grab_requests):
                request, offset = self.__build_request(sensor_grab_requests, request_type, extra_request_arguments,
                                                       crop=crop, sensor_grab_requests_offset=offset, trigger_type=trigger_type)
                push(request)
        return [sensor_grab_request.history_id for sensor_grab_request in sensor_grab_requests]

    @log_call()
    def grabImageFromAllSensors(self, crop: bool, _=None) -> List[int]:
        system = System()
        sensors_to_grab = list()
        for sensor in system.get_sensors():
            sensor_id = sensor.pipeline()
            sensor_group_uuids = self._project_info.get_sensor_groups_for_sensor_id(sensor.pipeline())
            if len(sensor_group_uuids) == 0:
                get_main_logger().info("grabImageFromAllSensors: Could not find any sensor group uuids for sensor id {}, so going to skip this sensor.".format(
                    sensor_id
                ))
                continue
            sensors_to_grab.append(SensorInfo(sensor_id, sensor_group_uuids[0]))

        if len(sensors_to_grab) == 0:
            get_main_logger().info("grabImageFromAllSensors: No sensors to grab from.")
        else:
            return self.__push_grab_request(sensors_to_grab, ice_image_grab.TriggerType.MULTIMANUAL, LiveGrabRequest, crop=crop)

    @log_call()
    def grabImage(self, sensor_id: int, sensor_group_uuid: str, _, __=None) -> int:
        return self.__push_grab_request([SensorInfo(sensor_id, sensor_group_uuid), ], ice_image_grab.TriggerType.MANUAL, LiveGrabRequest)[0]

    @log_call()
    def grabImageWithCustomTrigger(self, sensor_id: int, sensor_group_uuid: str, thresholds: List[ice_image_grab.Threshold], cache: bool, _=None):
        return self.__grab_image_with_trigger(sensor_id, sensor_group_uuid, thresholds, crop=False, cache=cache)

    @log_call()
    def grabImageWithTrigger(self, sensor_id: int, sensor_group_uuid: str, cache: bool, _=None):
        thresholds = self._project_info.get_thresholds(sensor_group_uuid)
        return self.__grab_image_with_trigger(sensor_id, sensor_group_uuid, thresholds, crop=False, cache=cache)

    @log_call()
    def grabFrontImagesWithAIAddonTrigger(self, _=None) -> List[int]:
        def get_sensors_with_view(predicate: Callable[[str], bool]) -> List[SensorInfo]:
            return [
                SensorInfo(
                    sensor.pipeline(),
                    self._project_info.get_sensor_groups_for_sensor_id(sensor.pipeline())[0]
                )
                for sensor in System().get_sensors()
                if predicate(sensor.view())
            ]
        sensors = get_sensors_with_view(lambda view: 'front' in view.lower())
        if not sensors:
            get_main_logger().info('No front sensors detected, using top sensors instead')
            sensors = get_sensors_with_view(lambda view: 'top' in view.lower())
        if not sensors:
            raise RuntimeError('Unable to determine front sensors! It is very unlikely that you have an instance without front and top sensors!')

        return self.__push_grab_request(
            sensor_info_list=sensors,
            trigger_type=ice_image_grab.TriggerType.AIADDON,
            request_type=AIAddonTriggeredGrabRequest,
        )

    @log_call()
    def grabCroppedImageWithCustomTrigger(self, sensor_id: int, sensor_group_uuid: str, thresholds: List[ice_image_grab.Threshold], cache: bool, _=None):
        return self.__grab_image_with_trigger(sensor_id, sensor_group_uuid, thresholds, crop=True, cache=cache)

    @log_call()
    def grabCroppedImageWithTrigger(self, sensor_id: int, sensor_group_uuid: str, cache: bool, _=None):
        thresholds = self._project_info.get_thresholds(sensor_group_uuid)
        return self.__grab_image_with_trigger(sensor_id, sensor_group_uuid, thresholds, crop=True, cache=cache)

    @log_call()
    def grabImageWithSurfaceTrigger(self, sensor_id: int, sensor_group_uuid: str, surface_area: int, _=None):
        return self.__push_grab_request([SensorInfo(sensor_id, sensor_group_uuid), ], ice_image_grab.TriggerType.TRIGGERED, SurfaceTriggeredGrabRequest, [surface_area])[0]

    def __grab_image_with_trigger(self,
                                  sensor_id: int, sensor_group_uuid: Optional[str], thresholds: List[ice_image_grab.Threshold], crop: bool = False, cache: bool = False) -> int:
        trigger_thresholds = list()
        for threshold in thresholds:
            trigger_threshold = TriggerThreshold()
            trigger_threshold.attribute_id = threshold.attributeId
            trigger_threshold.algorithm = threshold.algorithmIndex
            trigger_threshold.threshold_index = threshold.thresholdIndex
            trigger_thresholds.append(trigger_threshold)

        return self.__push_grab_request(
            [SensorInfo(sensor_id, sensor_group_uuid), ],
            ice_image_grab.TriggerType.TRIGGERED,
            TriggeredGrabRequest,
            [trigger_thresholds],
            crop=crop
        )[0]

    # noinspection PyMethodMayBeStatic
    def getSensors(self, _=None) -> List[ice_image_grab.Sensor]:
        try:
            system = System()
            sensors = list()
            for sensor in system.get_sensors():
                s = ice_image_grab.Sensor()
                s.sensorId = sensor.pipeline()
                s.sensorName = sensor.name()
                sensors.append(s)
            return sensors
        except ValueError as e:
            raise ice_image_grab.RunTimeError(str(e))

    # noinspection PyMethodMayBeStatic
    def getSensorName(self, sensorId, _=None):
        try:
            sensor = ImageGrabConfiguration.sensor_by_id(sensorId)
            return sensor.name()
        except ValueError as e:
            raise ice_image_grab.RunTimeError(str(e))

    def cancelGrabImage(self, _=None):
        self._grabber.cancel()

    def grabImageActive(self, _=None):
        return self._grabber.is_active()

    @log_call()
    def getOverlays(self, historyId, sensorGroupId, live, _=None):
        self._get_all_overlays(historyId, sensorGroupId, live)

    class RequestCallback(object):
        def __init__(self,
                     history_item: object,
                     history_id: int,
                     sensor_group_id: str,
                     image_type: str,
                     simulator_id: str):
            #
            self._sensor_id = history_item.sensor_id
            self._saved = history_item.saved
            self._history_id = history_id
            self._sensor_group_id = sensor_group_id
            self._image_type = image_type
            self._simulator_id = simulator_id
            # ice_image_grab.JobFinishedObserverPrx
            self._publisher = ice_manager().get_publisher_for_topic(
                ice_image_grab.JobFinishedObserverPrx, ImageGrabData.TOPIC)

        def __call__(self, success: bool, errors: Optional[List] = None):
            get_main_logger().info("Job finished for '{}'.".format(self._image_type))
            # we need this to retrieve data
            DB.update_img(self._history_id, self._image_type, sensor_group_id=self._sensor_group_id)
            #
            job_finished = ice_image_grab.JobFinished()
            job_finished.historyId = self._history_id
            job_finished.sensorId = self._sensor_id
            job_finished.sensorGroupId = self._sensor_group_id
            job_finished.imageType = self._image_type
            job_finished.projectChecksum = ImageGrabServant.LIVE_SYNC_ID
            job_finished.projectUuid = ""
            job_finished.saved = self._saved
            job_finished.success = success
            if errors is not None:
                job_finished.errors = [MSG_OVERLAY_TYPE[self._image_type], ] + errors
            else:
                job_finished.errors = list()
            job_finished.jobFlags = 0
            job_finished.simulatorId = self._simulator_id
            job_finished.trigger = ice_image_grab.TriggerType.UNKNOWN
            self._publisher.finished(job_finished)

    OVERLAY_TYPES = [
        ImageGrabData.OVERALL_SEGMENTATION_OVERLAY,
        ImageGrabData.OVERALL_EJECTS_OVERLAY,
        ImageGrabData.OVERALL_CONTOURS_OVERLAY,
        ImageGrabData.OVERALL_ALGORITHMS_OVERLAY,
        ImageGrabData.BACKGROUND_OVERLAY
    ]

    def _get_all_overlays(self, history_id, sensor_group_id, live):
        # type: (int, str, bool) -> None
        if sensor_group_id == "":
            raise ice_image_grab.RunTimeError("Sensor group id cannot be empty.")

        history_item = DB.get(history_id)

        f = FileFactory.get_from_db(ImageGrabData.KIF, history_id, history_item=history_item)
        kif_path = f.path(create_directories=False)

        files = Files()
        callbacks = Callbacks()
        ff_kwargs = dict(project_checksum=self.LIVE_SYNC_ID, history_item=history_item)

        ff_kwargs["sensor_group_id"] = sensor_group_id
        attribute_info_list = self._project_info.get_attribute_info(sensor_group_id)
        attributes = [True for _ in attribute_info_list]

        simulator_id = self._live_simulator.identifier if live else self._simulator.identifier
        ff_kwargs["simulator_id"] = simulator_id

        for overlay_type in self.OVERLAY_TYPES:
            files.add(overlay_type, FileFactory.get_from_db(overlay_type, history_id, **ff_kwargs))
            callbacks.add(overlay_type,
                          ImageGrabServant.RequestCallback(
                              history_item, history_id, sensor_group_id, overlay_type, simulator_id))

        assignment_uuid = self._project_info.get_assignment_uuid_for_sensor_group(sensor_group_id)
        pipeline = history_item.sensor_id if history_item.sensor_id is not None and history_item.sensor_id > 0 else 0
        overlay_request = OverlayRequest(kif_path, pipeline, assignment_uuid, attributes, None, files, callbacks)

        if live:
            self._live_simulator.push(overlay_request)
        else:
            self._simulator.push(overlay_request)

        for overlay_type in self.OVERLAY_TYPES:
            self.publish_start(history_id, history_item.sensor_id, sensor_group_id, overlay_type)

    @log_call()
    def getRegionOverlay(self, historyId: int, sensorGroupId: str, x: int, y: int, live: bool, _=None):
        history_item = DB.get(historyId)
        f = FileFactory.get_from_db(ImageGrabData.KIF, historyId, history_item=history_item)
        kif_path = f.path(create_directories=False)
        pipeline = history_item.sensor_id if history_item.sensor_id is not None and history_item.sensor_id > 0 else 0
        assignment_uuid = self._project_info.get_assignment_uuid_for_sensor_group(sensorGroupId)
        attribute_info_list = self._project_info.get_attribute_info(sensorGroupId)
        attributes = [attribute_info.inSensorGroup for attribute_info in attribute_info_list]

        simulator = self._live_simulator if live else self._simulator

        files = Files()
        callbacks = Callbacks()
        files.add(ImageGrabData.SELECTED_REGION_OVERLAY,
                  FileFactory.get_from_db(
                      ImageGrabData.SELECTED_REGION_OVERLAY, historyId, sensor_group_id=sensorGroupId, project_checksum=self.LIVE_SYNC_ID))
        callbacks.add(ImageGrabData.SELECTED_REGION_OVERLAY, ImageGrabServant.RequestCallback(
            history_item, historyId, sensorGroupId, ImageGrabData.SELECTED_REGION_OVERLAY, simulator.identifier))

        overlay_request = OverlayRequest(
            kif_path, pipeline, assignment_uuid, attributes, RegionInfo(x, y), files, callbacks)
        simulator.push(overlay_request)

        self.publish_start(historyId, history_item.sensor_id, sensorGroupId, ImageGrabData.SELECTED_REGION_OVERLAY)

    @log_call()
    def getAIAddonSVGOverlay(self, history_id: int, _=None) -> str:
        history_item = DB.get(history_id)

        file = FileFactory.get_from_db(ImageGrabData.KIF, history_id, history_item=history_item)  # type: File
        kif_path = file.path(create_directories=False)
        extended_kif_image = ExtendedKifImage(kif_path)

        bounding_box = AIAddonBox.from_dict(history_item.data)
        assert(bounding_box.is_initialized())

        sensor_id = history_item.sensor_id
        sensor = next(sensor for sensor in System().get_sensors() if sensor.pipeline() == sensor_id)

        x_range = bounding_box.get_x_range_in_image(sensor)
        y_range = bounding_box.get_y_range_in_image(extended_kif_image)

        return SVGUtils.create_svg_rect(
            x=x_range[0],
            y=y_range[0],
            width=x_range[1] - x_range[0],
            height=y_range[1] - y_range[0],
            image_dimensions=(extended_kif_image.shape[1], extended_kif_image.shape[0])
        )

    @staticmethod
    def _convert_tuple_color_to_hex(color_as_tuple, alpha=False):
        r, g, b, a = color_as_tuple
        if alpha:
            return "#{:02x}{:02x}{:02x}{:02x}".format(int(r), int(g), int(b), int(a))
        else:
            return "#{:02x}{:02x}{:02x}".format(int(r), int(g), int(b))

    # noinspection PyMethodMayBeStatic
    @log_call()
    def getBackgroundColor(self, _=None):
        return self._convert_tuple_color_to_hex(self._project_info.background_color)

    # noinspection PyMethodMayBeStatic
    @log_call()
    def getDefaultAttributeColors(self, _=None):
        return [self._convert_tuple_color_to_hex(color) for color in self._project_info.default_attribute_colors]

    # noinspection PyMethodMayBeStatic
    @log_call()
    def getEjectAttributeColors(self, _=None):
        return [self._convert_tuple_color_to_hex(color) for color in self._project_info.eject_attribute_colors]

    # noinspection PyMethodMayBeStatic
    @log_call()
    def getAttributeInfo(self, sensorGroupId, _=None):
        return self._project_info.get_attribute_info(sensorGroupId)

    # noinspection PyMethodMayBeStatic
    @log_call()
    def save(self, historyId, _=None):
        self._save(historyId, self._project_info.project_uuid)

    # noinspection PyMethodMayBeStatic
    @log_call()
    def saveWithDescription(self, historyId, description, _=None):
        self._save(historyId, self._project_info.project_uuid, description)

    # noinspection PyMethodMayBeStatic
    @log_call()
    def remove(self, historyId, _=None):
        cleanup.remove(historyId)

    @staticmethod
    def _has_overall_image_type(image_type):
        try:
            return {
                ImageGrabData.SEGMENTATION_OVERLAY: ImageGrabData.OVERALL_SEGMENTATION_OVERLAY,
                ImageGrabData.EJECTS_OVERLAY: ImageGrabData.OVERALL_EJECTS_OVERLAY,
                ImageGrabData.CONTOURS_OVERLAY: ImageGrabData.OVERALL_CONTOURS_OVERLAY,
                ImageGrabData.ALGORITHMS_OVERLAY: ImageGrabData.OVERALL_ALGORITHMS_OVERLAY,
            }[image_type]
        except KeyError:
            return None

    @log_call()
    def getData(self, historyId, imageType, sensorGroupId, simulatorId, _=None):
        history_item = DB.get(historyId)
        #
        if imageType not in ImageGrabData.IMAGE_TYPES + (ImageGrabData.FM_SEGMENTED, ImageGrabData.FM_BOUNDING_BOX):
            raise ice_image_grab.RunTimeError(
                "Given image type ({}) is not valid! Supported image types are: {}.".format(
                    imageType, ",".join(ImageGrabData.IMAGE_TYPES)
                )
            )

        if imageType in self.OVERLAY_TYPES:
            simulator_id = simulatorId
        else:
            simulator_id = ""

        get_main_logger().debug("Request for data, history id: {}, image type: '{}'.".format(
            historyId, imageType
        ))

        if imageType in (ImageGrabData.FM_SEGMENTED, ImageGrabData.FM_BOUNDING_BOX):
            # for these specific image types, we rely on RAW_IMAGE data.
            db_image_type = ImageGrabData.RAW_IMAGE
        else:
            db_image_type = imageType

        # Get the latest project
        project_checksum = getattr(history_item, DB.project_checksum_field(db_image_type))
        sensor_group_id = getattr(history_item, DB.sensor_group_id_field(db_image_type))

        if sensorGroupId != "" and sensorGroupId != sensor_group_id:
            get_main_logger().warning("Given sensor group id ({}) doesn't match the one found in the DB: {}.".format(
                sensorGroupId, sensor_group_id
            ))

        result = list()

        #
        def __create_grabbed_image(_file_path, _attribute_index=-1, _is_fallback_path=False):
            res = ice_image_grab.GrabbedImage()
            res.fileName = os.path.basename(_file_path)
            res.name = imageType
            res.description = history_item.description
            res.sensorId = history_item.sensor_id
            res.sensorName = history_item.sensor_name
            if _is_fallback_path and sensor_group_id != "":
                res.sensorGroupId = sensor_group_id
            else:
                res.sensorGroupId = sensorGroupId
            res.attributeId = _attribute_index
            res.saved = history_item.saved == 1
            res.triggered = history_item.triggered == 1
            res.fmAlert = history_item.fm_alert == 1
            res.trigger = Conversions.trigger_type_value_to_trigger_type(history_item.triggered_by)
            res.timestamp = get_utc_seconds(history_item)
            meta_data = dict()
            # noinspection PyBroadException
            try:
                if imageType == ImageGrabData.SELECTED_REGION_OVERLAY:
                    svg = SVG(_file_path)
                    svg.load()
                    metadata_obj = svg.metadata[0]  # type: Metadata
                    meta_data = metadata_obj.data
            except Exception:
                get_main_logger().exception(
                    "Exception while trying to extract meta data from select region overlay SVG: {}.".format(
                        _file_path
                    )
                )
            res.metaData = json.dumps(meta_data)

            with open(_file_path, 'rb') as file_:
                res.data = file_.read()
            return res

        def __check_and_add_path(_path, _path2, _fallback_path=None, _attribute=-1):
            if os.path.isfile(_path):
                get_main_logger().debug("File '{}' exists!.".format(_path))
                result.append(__create_grabbed_image(_path, _attribute))
            elif os.path.isfile(_path2):
                get_main_logger().debug("File '{}' exists!.".format(_path2))
                result.append(__create_grabbed_image(_path2, _attribute))
            elif _fallback_path is not None and os.path.isfile(_fallback_path):
                get_main_logger().debug("Fallback file '{}' exists!.".format(_path))
                result.append(__create_grabbed_image(_fallback_path, _attribute, _is_fallback_path=True))
            else:
                get_main_logger().debug("File '{}' does NOT exist!.".format(_path))
                get_main_logger().debug("File '{}' does NOT exist!.".format(_path2))

        ##########################
        # fallback image type ####
        ##########################
        fallback_image_type = self._has_overall_image_type(imageType)
        if fallback_image_type is not None:
            fallback_data_file = FileFactory.get_from_db(fallback_image_type, historyId,
                                                         project_checksum=project_checksum,
                                                         sensor_group_id=sensor_group_id,
                                                         simulator_id=simulator_id)
            fallback_full_path = partial(fallback_data_file.path, create_directories=False)
        else:
            def NOOP(*_, **__):
                return None

            fallback_full_path = NOOP
        ##########################

        data_file = FileFactory.get_from_db(imageType, historyId,
                                            project_checksum=project_checksum,
                                            sensor_group_id=sensor_group_id,
                                            simulator_id=simulator_id)
        full_path = partial(data_file.path, create_directories=False)

        data_file2 = FileFactory.get_from_db(imageType, historyId,
                                             project_checksum=self.LIVE_SYNC_ID,
                                             sensor_group_id=sensor_group_id,
                                             simulator_id=simulator_id)
        full_path2 = partial(data_file2.path, create_directories=False)

        if is_non_attribute_format(imageType):
            __check_and_add_path(full_path(), full_path2())
        else:
            for attribute in ALL_ATTRIBUTES:
                # noinspection PyArgumentList
                __check_and_add_path(
                    full_path(suffix=str(attribute)),
                    full_path2(suffix=str(attribute)),
                    fallback_full_path(suffix=str(attribute)),
                    attribute)

        return result

    # noinspection PyMethodMayBeStatic
    @log_call()
    def getRawData(self, historyId, _=None):
        history_data = DB.get(historyId)

        kif_file = FileFactory.get_from_db(ImageGrabData.KIF, historyId)
        full_path = kif_file.path(create_directories=False)
        if not os.path.exists(full_path):
            raise ice_image_grab.RunTimeError("Raw image data for history id {} does not exist!".format(historyId))

        result = ice_image_grab.GrabbedImage()
        result.fileName = os.path.basename(full_path)
        result.name = ImageGrabData.KIF
        result.description = history_data.description
        result.sensorId = history_data.sensor_id
        result.sensorName = history_data.sensor_name
        result.sensorGroupId = history_data.trigger_sensor_group_id
        result.attributeId = -1
        result.saved = result.saved == 1
        result.triggered = history_data.triggered == 1
        result.fmAlert = history_data.fm_alert == 1
        result.trigger = Conversions.trigger_type_value_to_trigger_type(history_data.triggered_by)
        result.timestamp = get_utc_seconds(history_data)
        with open(full_path, 'rb') as raw_data:
            result.data = raw_data.read()
        return result

    # noinspection PyMethodMayBeStatic
    @log_call()
    def getHistory(self, _=None):
        return ImageGrabServant._get_history(self._project_info.project_uuid, None)

    def __verify_recipe_timestamp(self, timestamp: float) -> None:
        with self._recipe_update_timestamp.get_lock():
            if self._recipe_update_timestamp.value != 0. and timestamp < self._recipe_update_timestamp.value:
                raise AbortGetHistory()

    def _get_compatible_history_for_sensor_id(self, identifier: str, timestamp: float, sensor_id: int, with_thumbnail: bool) -> None:
        # noinspection PyBroadException
        try:
            self.__verify_recipe_timestamp(timestamp)
            sensor_group_id = self._project_info.get_sensor_groups_for_sensor_id(sensor_id)[0]
        except IndexError:
            sensor_group_id = ""
        except AbortGetHistory:
            self._send_load_history_expired(identifier, "")
            return
        except Exception:
            get_main_logger().error("Get compatible history failed while getting sensor group id for sensor with id '{}'.".format(sensor_id))
            LoadHistoryManager.load_history_finished(identifier, "", ice_image_grab.LoadHistoryState.FAILED, "Get compatible history failed.", list())
            #
            return
        self._get_compatible_history_for_sensor_group_id(identifier, timestamp, sensor_group_id, with_thumbnail)

    def _get_compatible_history_for_sensor_group_id(self, identifier: str, timestamp: float, sensor_group_id: str, with_thumbnail: bool) -> None:
        get_main_logger().info("Get compatible history: sensor_group_id={} -- internal id={} -- qsize~={}".format(
            sensor_group_id, identifier, self._history_worker.approximate_size))

        # noinspection PyBroadException
        try:
            self.__verify_recipe_timestamp(timestamp)
            if sensor_group_id not in [sensor_group_info[0] for sensor_group_info in self._project_info.sensor_groups]:
                get_main_logger().error("getCompatibleHistoryForSensorGroupId: invalid sensor group uuid given: {}.".format(sensor_group_id))
                LoadHistoryManager.load_history_finished(
                    identifier, sensor_group_id, ice_image_grab.LoadHistoryState.EXPIRED, "Get compatible history failed: invalid sensor group id ({}) given.".format(sensor_group_id), list())
                #
                return
            image_compatibility_info = self._project_info.get_image_compatibility_info(sensor_group_id)
            project_uuid = self._project_info.project_uuid
        except AbortGetHistory:
            self._send_load_history_expired(identifier, sensor_group_id)
            return
        except KeyError:
            # Sensor group id is not valid anymore for current project, hence expired.
            self._send_load_history_expired(identifier, sensor_group_id)
            return
        except Exception:
            get_main_logger().exception(
                "Sending failed event for 'Get compatible history': Id of request: {}, id of sensor group: {}.".format(
                    identifier, sensor_group_id))
            LoadHistoryManager.load_history_finished(
                identifier, sensor_group_id, ice_image_grab.LoadHistoryState.FAILED, "Get compatible history failed: failed to get sensor group info.".format(sensor_group_id), list())
            #
            return

        self._get_compatible_history(
            identifier,
            timestamp,
            [
                self._images_watchers[ImagePath.FMALERT],
                self._images_watchers[ImagePath.UI],
                self._images_watchers[ImagePath.USER]
            ],
            image_compatibility_info,
            with_thumbnail,
            project_uuid,
            sensor_group_id
        )

    @staticmethod
    def _send_load_history_expired(identifier: str, sensor_group_id: str):
        get_main_logger().info("Sending abort event for 'Get compatible history' since request is outdated:"
                               "Id of request: {}, id of sensor group: {}.".format(identifier, sensor_group_id))
        LoadHistoryManager.load_history_finished(
            identifier,
            sensor_group_id,
            ice_image_grab.LoadHistoryState.EXPIRED,
            "Get compatible history was aborted.",
            list()
        )

    def _get_compatible_history(self, identifier: str, timestamp: float,
                                images_watches: List[images_watch.ImagesWatch],
                                image_compatibility_info: ImageCompatibilityInfo,
                                with_thumbnail: bool, project_uuid: str,
                                reference_sensor_group_id: str) -> None:
        # noinspection PyBroadException
        try:
            self.__verify_recipe_timestamp(timestamp)
            PublishHistoryChanges().block_publishing()
            #
            skip_validation = not ImageGrabData.IMAGES_VALIDATION_ENABLED
            timings = Timings()

            temp_list = list()

            with timings.time("Get compatible history ({}))".format(identifier)):
                result = list()
                self.__verify_recipe_timestamp(timestamp)
                with timings.time("Get data from db ({})".format(identifier)):
                    with DB() as open_history_database:
                        query = ImageGrabServant._get_history_query(open_history_database, project_uuid)
                        # noinspection PyProtectedMember
                        query._expression.append(" ORDER BY {} DESC".format(History.TIMESTAMP_COLUMN))
                        q_end = query.end()

                        self.__verify_recipe_timestamp(timestamp)
                        for item in q_end:
                            f = FileFactory.get_from_db(ImageGrabData.KIF, item.history_item_id, history_item=item)
                            kif_path = f.path(create_directories=False)
                            if not os.path.isfile(kif_path):
                                get_main_logger().debug("Path does not exist: {}".format(kif_path))
                                continue
                            temp_list.append((kif_path, db_item_to_history_item(item)))
                timings.log("Get data from db ({})".format(identifier))

                self.__verify_recipe_timestamp(timestamp)
                for kif_path, ice_item in temp_list:
                    self.__verify_recipe_timestamp(timestamp)
                    for watch in images_watches:
                        kif_info = watch.get_image_info(kif_path)
                        if kif_info is None:
                            continue
                        #
                        compatibility_validation = Images.check_compatibility(
                            kif_info.bit_depth, kif_info.number_of_channels, kif_info.is_signed,
                            kif_info.sensor_name, image_compatibility_info)

                        if skip_validation or compatibility_validation:
                            ##
                            # noinspection PyBroadException
                            try:
                                if with_thumbnail:
                                    ice_item.thumbnail, _, _ = kif_info.get_png_data(
                                        image_compatibility_info.display_options)
                                result.append(ice_item)
                            except Exception:
                                get_main_logger().exception("Failed to retrieve thumbnail from KIF info: '{}'.".format(
                                    kif_path
                                ))
                        break  # no need to check other watches

            timings.log_all()
            LoadHistoryManager.load_history_finished(
                identifier,
                reference_sensor_group_id,
                ice_image_grab.LoadHistoryState.SUCCESS,
                "Get compatible history was successful.",
                result
            )
        except AbortGetHistory:
            self._send_load_history_expired(identifier, reference_sensor_group_id)
        except Exception:
            get_main_logger().exception(
                "Sending failed event for 'Get compatible history': Id of request: {}, id of sensor group: {}.".format(
                    identifier, reference_sensor_group_id))
            LoadHistoryManager.load_history_finished(
                identifier,
                reference_sensor_group_id,
                ice_image_grab.LoadHistoryState.FAILED,
                "Get compatible history failed.",
                list()
            )

    # noinspection PyMethodMayBeStatic
    @log_call()
    def getCompatibleHistoryForSensorId(self, sensorId, withThumbnail, _=None):
        identifier = str(uuid.uuid4())
        timestamp = time.time()
        job = partial(self._get_compatible_history_for_sensor_id, identifier, timestamp, sensorId, withThumbnail)
        self._history_worker.add(job)
        return identifier

    # noinspection PyMethodMayBeStatic
    @log_call()
    def getCompatibleHistoryForSensorGroupId(self, sensorGroupId, withThumbnail, _=None):
        identifier = str(uuid.uuid4())
        timestamp = time.time()
        job = partial(self._get_compatible_history_for_sensor_group_id, identifier, timestamp, sensorGroupId, withThumbnail)
        self._history_worker.add(job)
        return identifier

    # noinspection PyMethodMayBeStatic
    @log_call()
    def getHistoryByTrigger(self, triggerTypes, _=None):
        return ImageGrabServant._get_history(self._project_info.project_uuid, None, *triggerTypes)

    # noinspection PyMethodMayBeStatic
    @log_call()
    def getHistoryItem(self, historyId, _=None):
        db_item = DB.get(history_id=historyId)
        if db_item.saved == 1:
            return db_item_to_history_item(db_item)
        else:
            raise ice_image_grab.RunTimeError("Invalid history id: {}.".format(historyId))

    # noinspection PyMethodMayBeStatic
    @log_call()
    def getFMHistory(self, _=None):
        return ImageGrabServant._get_history(None, None, ice_image_grab.TriggerType.FMALERT)

    # noinspection PyMethodMayBeStatic
    @log_call()
    def getFMAlertData(self, historyId, _=None):
        db_item = DB.get(history_id=historyId)
        if db_item.saved == 1 and db_item.fm_alert == 1:
            fm_alert_info = FMAlertDBInfo.from_dict(db_item.data)

            data = ice_image_grab.FMAlertData()
            get_main_logger().debug(db_item.trigger_thresholds)
            data.triggeredBy = Conversions.trigger_thresholds_from_db(db_item.trigger_thresholds)[0]
            box = ice_image_grab.BoundingBox()
            box.startPixel = fm_alert_info.bounding_box.start_pixel
            box.endPixel = fm_alert_info.bounding_box.end_pixel
            box.startLine = fm_alert_info.bounding_box.start_line
            box.endLine = fm_alert_info.bounding_box.end_line
            data.box = box
            data.attributeName = fm_alert_info.attribute_name
            data.algorithmName = fm_alert_info.algorithm_name
            return data
        else:
            raise ice_image_grab.RunTimeError("Invalid history id: {}.".format(historyId))

    # noinspection PyMethodMayBeStatic
    @log_call()
    def deleteCompleteFMHistory(self, _=None):
        with open_database() as db:
            history_changes = list()

            all_history = ImageGrabServant._get_history(None, db, ice_image_grab.TriggerType.FMALERT)
            if len(all_history) == 0:
                return
            #
            files_to_remove = list()
            for history_item in all_history:
                for image_type in ImageGrabData.IMAGE_ALL_TYPES:
                    # noinspection PyBroadException
                    try:
                        f = FileFactory.get_from_db(image_type, history_item.historyId, open_history_database=db)
                        files_to_remove.append(f.path(create_directories=False))
                    except Exception:
                        continue

                history_change = ice_image_grab.HistoryChange()
                # Item is going to be removed so we are just filling in the history id
                stripped_history_item = ice_image_grab.HistoryItem()
                stripped_history_item.historyId = history_item.historyId
                history_change.historyItem = stripped_history_item
                history_change.event = ice_image_grab.HistoryChangedEvent.REMOVED
                history_change.sensorGroupIds = list()
                history_changes.append(history_change)
                db.remove(history_item.historyId, commit=False)
            db.commit()
            publish_history_changes(*history_changes)

        def __delete_files(files_to_remove_):
            for file_to_remove_ in files_to_remove_:
                # noinspection PyBroadException
                try:
                    if os.path.isfile(file_to_remove_):
                        os.remove(file_to_remove_)
                except Exception:
                    get_main_logger().exception("Failed to remove: {}.".format(file_to_remove_))

        threading.Thread(target=__delete_files, args=(files_to_remove,)).start()

    # noinspection PyMethodMayBeStatic
    @log_call()
    def downloadFMZipFile(self, historyId, _=None):
        buff = BytesIO()
        self.fm_history_to_zip(buff, (historyId,))
        f = ice_image_grab.FMZipFile()
        f.data = buff.getvalue()
        return f

    # noinspection PyMethodMayBeStatic
    @log_call()
    def downloadCompleteFMHistory(self, _=None):
        buff = BytesIO()
        self.fm_history_to_zip(buff, self.fm_history_ids())
        f = ice_image_grab.FMZipFile()
        f.data = buff.getvalue()
        return f

    @staticmethod
    def __result(success, message):
        result = ice_image_grab.Result()
        result.identifier = ""
        result.success = success
        result.message = message
        return result

    def __cp_to_usb(self, prefix, history_ids, progress_identifier=None):
        usb = USB()
        usb.load()
        if usb.num_devices == 0:
            return self.__result(False, "No USB drive detected and/or available.")

        with NamedTemporaryFile() as f:
            self.fm_history_to_zip(f.name, history_ids, progress_identifier=progress_identifier)
            f.flush()
            # noinspection PyBroadException
            try:
                with usb.mounted() as mount_path:
                    d = make_timestamp_dir(mount_path, "FM_ALERT")
                    file_name = '{prefix}{timestamp:%Y%m%d_%H%M%S}.zip'.format(
                        prefix=prefix,
                        timestamp=datetime.datetime.now()
                    )
                    file_path = os.path.join(d, file_name)
                    get_main_logger().debug("Starting copy to USB: from '{}' to '{}'.".format(
                        f.name,
                        file_path
                    ))
                    shutil.copy(f.name, file_path)
                    return self.__result(True, "FM zip file successfully copied to: {}.".format(file_path))
            except Exception:
                get_main_logger().exception("Failed to create directory on USB.")
                return self.__result(False, "Failed to create directory on USB.")

    # noinspection PyMethodMayBeStatic
    @log_call()
    def copyFMZipFileToUSB(self, historyId, _=None):
        return self.__cp_to_usb("fm_{}_".format(historyId), (historyId,))

    # noinspection PyMethodMayBeStatic
    @log_call()
    def copyCompleteFMHistoryToUSB(self, _=None):
        identifier = str(uuid.uuid4())

        def __wrapper():
            # noinspection PyBroadException
            try:
                result = self.__cp_to_usb("fm_all_", self.fm_history_ids())
                ProgressManager.finished(
                    identifier,
                    result.success,
                    result.message,
                    "copyCompleteFMHistoryToUSB"
                )
            except Exception:
                get_main_logger().exception("Failed to copy complete FM history to USB.")
                ProgressManager.finished(
                    identifier,
                    False,
                    "Failed to copy complete FM history to USB.",
                    "copyCompleteFMHistoryToUSB"
                )

        threading.Thread(target=__wrapper).start()
        return identifier

    @classmethod
    def fm_history_ids(cls):
        all_history = ImageGrabServant._get_history(None, None, ice_image_grab.TriggerType.FMALERT)
        return (history_item.historyId for history_item in all_history)

    # noinspection PyUnusedLocal
    @classmethod
    def fm_history_to_zip(cls, destination, history_ids, progress_identifier=None):
        with ZipFile(destination, 'w', allowZip64=True) as zip_file:
            for history_id in history_ids:
                cls.add_fm_to_zip(history_id, zip_file)

    @classmethod
    def add_fm_to_zip(cls, history_id, zip_file):
        with fm_files(history_id) as files:
            for (path, arcname) in files:
                if not os.path.exists(path):
                    get_main_logger().warning("'{}' ('{}') does not exist, skipping!".format(
                        path, arcname
                    ))
                    continue
                zip_file.write(path, arcname=arcname)

    # noinspection PyMethodMayBeStatic
    @log_call()
    def cleanUp(self, _=None):
        get_main_logger().info("This is a NOOP method.")

    @staticmethod
    def __clean_up():
        RemoveFiles.remove_all_generated()
        clean_history()
        check_current_recipes()

    # noinspection PyMethodMayBeStatic
    @log_call()
    def setDescription(self, historyId, description, _=None):
        with DB() as history:
            history.update(historyId, description=description)

        publish_history_change(historyId, ice_image_grab.TriggerType.UNKNOWN, ice_image_grab.HistoryChangedEvent.UPDATED)

    # noinspection PyMethodMayBeStatic
    @log_call()
    def getSensorGroupIdsForHistoryId(self, historyId, _=None):
        history_item = DB.get(historyId)
        sensor_id = history_item.sensor_id
        return self._project_info.get_sensor_groups_for_sensor_id(sensor_id)

    ####################
    # IMAGES           #
    ####################

    def __get_images(self, identifier, category_images_watch, sensorGroupId, category):
        # type: (str, images_watch.ImagesWatch, str, ImagePath) -> None
        # noinspection PyBroadException
        try:
            image_compatibility_info = self._project_info.get_image_compatibility_info(sensorGroupId)
            image_path = ImagePath.from_ice(category)

            if category_images_watch is not None:
                result = Images()
                skip_compatibility_check = not Images.validate_image_compatibility_info(image_compatibility_info)
                for kif_full_path, kif_info in category_images_watch.snapshot():
                    if skip_compatibility_check or Images.check_compatibility(
                            kif_info.bit_depth,
                            kif_info.number_of_channels,
                            kif_info.is_signed,
                            kif_info.sensor_name,
                            image_compatibility_info):
                        #
                        result.add_image_with_kif_info(
                            image_path, kif_full_path, kif_info, image_compatibility_info.display_options)
            else:
                result = Images.search(image_compatibility_info, image_path)

            LoadImagesManager.load_images_finished(
                identifier,
                True,
                "Load images was successful.",
                category,
                result.get(image_path)
            )

        except Exception:
            get_main_logger().exception("Load images failed.")
            LoadImagesManager.load_images_finished(
                identifier,
                False,
                "Load images has failed.",
                category,
                list()
            )

    # noinspection PyMethodMayBeStatic
    @log_call()
    def getImages(self, sensorGroupId, category, _=None):
        identifier = str(uuid.uuid4())
        threading.Thread(target=self.__get_images,
                         args=(
                             identifier,
                             self._images_watchers[ImagePath.from_ice(category)],
                             sensorGroupId,
                             category
                         )).start()
        return identifier

    # noinspection PyMethodMayBeStatic
    @log_call()
    def importImage(self, imageId, save, _=None):
        image_id = ImageId.decode(imageId)
        get_main_logger().info("Request for import of image: path ({}), image_path ({}).".format(
            image_id.path,
            image_id.image_path
        ))

        lookup_result = reverse_lookup(image_id.path, ImageGrabData.KIF)
        if lookup_result is not None:
            image_type, history_item = lookup_result
            if image_type == ImageGrabData.KIF:
                # Let's check if it's available for current project
                if history_item.project_rp_uuid == self._project_info.project_uuid:
                    return getattr(history_item, History.ID_COLUMN)

        return Import.from_file(image_id.path, self._project_info, available_to_all=True, save=save)

    @log_call()
    def getFullSizeImage(self, imageId, _=None):
        path = ImageId.decode(imageId).path
        kif_info = KIFInfo.create(path)
        display_options = self._project_info.get_display_options(kif_info)
        r, g, b = display_options.get_rgb()

        buff = BytesIO()
        setattr(buff, "name", "full_size.png")
        Convert.kif_to_png(
            path,
            buff,
            display_method=display_options.display_method,
            red_channel=r,
            green_channel=g,
            blue_channel=b
        )
        return buff.getvalue()

    # noinspection PyMethodMayBeStatic
    @log_call()
    def getFullSizeImageByHistoryId(self, historyId, _=None):
        f = FileFactory.get_from_db(ImageGrabData.RAW_IMAGE, historyId)
        path = f.path(create_directories=False)
        if not os.path.isfile(path):
            raise ValueError("{} does not exist!".format(path))
        with open(path, 'rb') as png_file:
            return png_file.read()

    @log_call()
    def getThumbnailImage(self, imageId, sensorGroupId, _=None):
        decoded_image_id = ImageId.decode(imageId)
        image_info = self._images_watchers[decoded_image_id.image_path].get_image_info(decoded_image_id.path)

        display_options = None
        if sensorGroupId is not None and sensorGroupId != "":
            display_options = self._project_info.get_display_options(sensorGroupId)
        else:
            for sensor_group_uuid, _, _ in self._project_info.sensor_groups:
                image_compatibility_info = self._project_info.get_image_compatibility_info(sensor_group_uuid)
                if image_compatibility_info.is_valid_sensor_name(image_info.sensor_name):
                    display_options = image_compatibility_info.display_options
                    break

        if display_options is None:
            display_options = ImageDisplayOptionsBase()  # just use default settings..

        data, _, _ = image_info.get_png_data(display_options)
        return data

    @log_call()
    def getThumbnailImageByHistoryId(self, historyId, _=None):
        history_item = DB.get(historyId)

        f = FileFactory.get_from_db(ImageGrabData.KIF, historyId, history_item=history_item)
        path = f.path(create_directories=False)
        if not os.path.isfile(path):
            raise ValueError("{} does not exist!".format(path))

        display_options = self._project_info.get_display_options(history_item.sensor_id)

        thumbnail = None
        for _, watch in self._images_watchers.items():
            kif_info = watch.get_image_info(path)
            if kif_info is None:
                continue
            thumbnail, _, _ = kif_info.get_png_data(display_options)

        if thumbnail is None:
            raise ice_image_grab.RunTimeError("Unable to provide thumbnail for history id '{}' ('{}').".format(
                historyId, path
            ))
        return thumbnail

    ####################
    ####################

    @contextmanager
    def _context(self, context: ImageGrabContext):
        self._enter_context(context)
        try:
            yield
        finally:
            self._leave_context(context)

    def _enter_context(self, context: ImageGrabContext):
        with self._image_grab_context_stack_lock:
            self._image_grab_context_stack.append(
                (context, context.create_properties())
            )

    def _leave_context(self, context: ImageGrabContext) -> ImageGrabContext:
        with self._image_grab_context_stack_lock:
            popped_context = self._image_grab_context_stack.pop()
            assert context is popped_context[0]
            return self._image_grab_context_stack[-1][0]

    @property
    def current_context(self) -> ImageGrabContext:
        with self._image_grab_context_stack_lock:
            return self._image_grab_context_stack[-1][0]

    @property
    def current_context_properties(self) -> Data:
        with self._image_grab_context_stack_lock:
            get_main_logger().debug("Returning current properties: {}".format(
                self._image_grab_context_stack[-1][1]
            ))
            return self._image_grab_context_stack[-1][1]

    ####################
    # SIMULATION       #
    ####################

    def __set_simulation_dirty(self, is_dirty):
        #
        # Assumes to be called from method wrapped in
        # @lock_within_context(ImageGrabContext.SIMULATION)
        #
        assert lock_within_context.is_locked(ImageGrabContext.SIMULATION)
        self.current_context_properties[SIMULATION_DIRTY] = is_dirty
        ice_manager().get_publisher_for_topic(
            ice_image_grab.SimulationChangedObserverPrx, ImageGrabData.SIMULATION_TOPIC).simulationChanged(
            is_dirty)

    @log_call()
    def enterSimulation(self, _=None):
        # step 1: enter simulation context
        self._enter_context(ImageGrabContext.SIMULATION)

        # step 2: prepare simulation context
        @lock_within_context(ImageGrabContext.SIMULATION)
        def _enter(_: ImageGrabServant) -> None:
            self._simulator.push(DisableUpdatesRequest())
            self.__set_simulation_dirty(False)

        _enter(self)
        return True

    @log_call()
    @lock_within_context(ImageGrabContext.SIMULATION)
    def leaveSimulation(self, _=None):
        new_context = self._leave_context(ImageGrabContext.SIMULATION)
        if new_context is not ImageGrabContext.NEW_DEFECT:
            # In the special case of TND, we want to wait to enabled updates until
            # TND is cancelled or applied.
            self._simulator.push(EnableUpdatesRequest())
        self.__set_simulation_dirty(False)
        return True

    @log_call()
    def simulationActive(self, _=None):
        return self.current_context is ImageGrabContext.SIMULATION

    @log_call()
    @lock_within_context(ImageGrabContext.SIMULATION, no_context=noop(False))
    def simulationHasChanged(self, _=None):
        return self.current_context_properties[SIMULATION_DIRTY]

    @log_call()
    @lock_within_context(ImageGrabContext.SIMULATION, no_context=noop())
    def setSimulationHasChanged(self, has_changed, _=None):
        self.__set_simulation_dirty(has_changed)

    @log_call()
    @lock_within_context(ImageGrabContext.SIMULATION)
    def updateSimulatedColorThreshold(self, threshold, value, _=None):
        self._simulator.push(ColorThresholdUpdate(
            threshold.pipelines, threshold.b1Bank, threshold.thresholdIndex, value
        ))
        self.__set_simulation_dirty(True)

    @log_call()
    @lock_within_context(ImageGrabContext.SIMULATION)
    def updateSimulatedShapeThreshold(self, threshold, value, _=None):
        self._simulator.push(ShapeThresholdUpdate(
            threshold.pipelines, threshold.attributeId, threshold.algorithmIndex, threshold.thresholdIndex,
            value
        ))
        self.__set_simulation_dirty(True)

    @log_call()
    @lock_within_context(ImageGrabContext.SIMULATION)
    def updateSimulatedShapeConditionThreshold(self, threshold, value, _=None):
        self._simulator.push(ShapeComboConditionUpdate(
            threshold.pipelines, threshold.attributeId, threshold.algorithmIndex, threshold.conditionIndex,
            value
        ))
        self.__set_simulation_dirty(True)

    @log_call()
    @lock_within_context(ImageGrabContext.SIMULATION)
    def updateSimulatedShapeActive(self, threshold, value, _=None):
        self._simulator.push(ShapeThresholdActiveUpdate(
            threshold.pipelines, threshold.attributeId, threshold.algorithmIndex, threshold.thresholdIndex,
            value
        ))
        self.__set_simulation_dirty(True)

    ####################
    ####################

    ####################
    # NEW DEFECT       #
    ####################

    @log_call()
    @lock_within_context(ImageGrabContext.NORMAL)
    def enterNewDefectContext(self, _=None) -> bool:
        self._enter_context(ImageGrabContext.NEW_DEFECT)
        return True

    @log_call()
    @lock_within_context(ImageGrabContext.NEW_DEFECT)
    def leaveNewDefectContext(self, _=None) -> bool:
        # discard all changes -- reset the simulator
        self._simulator.reload()
        self._leave_context(ImageGrabContext.NEW_DEFECT)
        return True

    @log_call()
    def newDefectContextActive(self, _=None):
        return self.current_context is ImageGrabContext.NEW_DEFECT

    @log_call()
    @lock_within_context(ImageGrabContext.NEW_DEFECT)
    def applyNewDefectContext(self, recipe_name: str, tile_name: str, _=None) -> str:
        sensor_group_id = self.current_context_properties[NEW_DEFECT_SENSOR_GROUP]
        if sensor_group_id is None:
            raise ice_image_grab.RunTimeError("No sensor group available, did you forget to update the defect?")
        request = ApplyCustomDefectRequest(sensor_group_id, recipe_name, tile_name, caller="applyNewDefectContext")
        self._simulator.push(request)
        self._leave_context(ImageGrabContext.NEW_DEFECT)
        return request.identifier

    @log_call()
    @lock_within_context(ImageGrabContext.NORMAL)
    def deleteCustomDefect(self, sensorGroupId: str, recipeName: str, _=None) -> str:
        request = DeleteCustomDefectRequest(sensorGroupId, recipeName, "", caller="deleteCustomDefect")
        self._live_simulator.push(request)
        # discard all changes -- reset the simulator
        self._live_simulator.reload()
        return request.identifier

    @log_call()
    @lock_within_context(ImageGrabContext.NEW_DEFECT)
    def updateDefect(self, sensor_group_id: str, image_coordinates_sequence: List, _=None):
        new_defect_sensor_group = self.current_context_properties[NEW_DEFECT_SENSOR_GROUP]
        if new_defect_sensor_group is not None and new_defect_sensor_group != sensor_group_id:
            get_main_logger().error("New defect context only allows editing of a single sensor group: "
                                    "given sensor group '{}' differs from expected sensor group '{}'".format(
                                        sensor_group_id, new_defect_sensor_group
                                    ))
            raise ice_image_grab.RunTimeError("New defect context only allows editing of a single sensor group.")
        self.current_context_properties[NEW_DEFECT_SENSOR_GROUP] = sensor_group_id
        #
        request = UpdateCustomDefectRequest(sensor_group_id, caller="updateDefect")
        for image_coordinates in image_coordinates_sequence:
            history_id = image_coordinates.historyId
            image_id = image_coordinates.imageId
            if history_id != -1:
                f = FileFactory.get_from_db(ImageGrabData.KIF, history_id)
                path = f.path(create_directories=False)
            else:
                image_id_decoded = ImageId.decode(image_id)
                path = image_id_decoded.path

            good_x = list()
            good_y = list()
            for coordinate in image_coordinates.good:
                good_x.append(coordinate.x)
                good_y.append(coordinate.y)
            bad_x = list()
            bad_y = list()
            for coordinate in image_coordinates.bad:
                bad_x.append(coordinate.x)
                bad_y.append(coordinate.y)

            request.add_image_coordinates_info(
                path,
                NumpyCoordinates2D(good_x, good_y),
                NumpyCoordinates2D(bad_x, bad_y)
            )

        self._simulator.push(request)
        return request.identifier

    ####################
    ####################

    ####################
    # LOGGING          #
    ####################

    # noinspection PyMethodMayBeStatic
    @log_call()
    def setLogLevel(self, log_level, _=None):
        try:
            py_log_level = {
                ice_image_grab.LogLevel.NOTSET: logging.NOTSET,
                ice_image_grab.LogLevel.DEBUG: logging.DEBUG,
                ice_image_grab.LogLevel.INFO: logging.INFO,
                ice_image_grab.LogLevel.WARN: logging.WARN,
                ice_image_grab.LogLevel.ERROR: logging.ERROR,
                ice_image_grab.LogLevel.FATAL: logging.FATAL,
            }[log_level]
        except KeyError:
            py_log_level = LOG_LEVEL

        set_log_level(py_log_level)

    ####################
    ####################

    def _save(self, history_id, project_uuid, description=None, do_publish=True):
        with DB() as history:
            count = check_current_history_count(project_uuid, history, True)
            get_main_logger().debug("Saving items linked with history id {}, history count before saving: {}.".format(
                history_id, count
            ))
            if history.has_flags(DB.HISTORY_ITEM_NOCACHE, history_id):
                raise ice_image_grab.RunTimeError("Can not save a non-cached grabbed image!")

            if history_id in history:
                if description is not None:
                    history.update(history_id, saved=1, description=description)
                else:
                    history.update(history_id, saved=1)
            if do_publish:
                kif_file_path = FileFactory.get_from_db(ImageGrabData.KIF, history_id).path(create_directories=False)
                publish_history_change(history_id, ice_image_grab.TriggerType.UNKNOWN, ice_image_grab.HistoryChangedEvent.ADDED,
                                       self._project_info.get_compatible_sensor_group_uuids(kif_file_path))

    @staticmethod
    def _get_history_query(db, project_uuid, *trigger_types):
        db_filter = dict()
        if project_uuid is not None:
            db_filter["project_rp_uuid"] = project_uuid

        result = None
        for trigger_type in trigger_types:
            result = (db.filter if result is None else result.OR).ALL(triggered_by=trigger_type.value, **db_filter)

        if result is None:
            result = db.filter.ALL(**db_filter)

        return result

    @staticmethod
    def _get_history(project_uuid, open_history_database, *trigger_types):
        history_sequence = list()
        with open_database(open_history_database) as history:
            result = ImageGrabServant._get_history_query(history, project_uuid, *trigger_types)

            for item in result.end():
                history_sequence.append(db_item_to_history_item(item))

            return sorted(history_sequence, key=lambda item__: item__.timestamp, reverse=False)

    @staticmethod
    def publish_start(history_id, sensor_id, sensor_group_id, image_type, triggered=False):
        job_started = ice_image_grab.JobStarted()
        job_started.historyId = history_id
        job_started.sensorId = sensor_id
        job_started.sensorGroupId = "" if sensor_group_id is None else sensor_group_id
        job_started.imageType = image_type
        job_started.triggered = triggered
        ice_manager().get_publisher_for_topic(
            ice_image_grab.JobStartedObserverPrx, ImageGrabData.TOPIC).started(job_started)

    def rebuild_project_info(self):
        get_main_logger().info("Rebuild project info triggered.")
        new_timestamp = time.time()
        get_main_logger().info("Updating recipe timestamp from {} to {}.".format(
            self._recipe_update_timestamp.value, new_timestamp
        ))
        self._recipe_update_timestamp.value = new_timestamp
        self._project_info.rebuild()
        self._simulator.reload()
        self._live_simulator.reload()

    def reset_project_info(self):
        get_main_logger().info("Reset project info triggered.")
        new_timestamp = time.time()
        get_main_logger().info("Updating recipe timestamp from {} to {}.".format(
            self._recipe_update_timestamp.value, new_timestamp
        ))
        self._recipe_update_timestamp.value = new_timestamp
        self._project_info.reset()
        self._simulator.discard()
        self._live_simulator.discard()


class ImageGrabService(Ice.Application):
    def run(self, args):
        ic = self.communicator()
        properties = ic.getProperties()
        adapter = ic.createObjectAdapter("fe.services.ImageGrab")
        identity = ic.stringToIdentity(properties.getProperty("Identity"))
        try:
            servant = ImageGrabServant()
        except Exception as e:
            get_main_logger().exception(e)
            raise e
        adapter.add(servant, identity)
        adapter.activate()
        self.shutdownOnInterrupt()
        ic.waitForShutdown()

        get_main_logger().info('ice image grab service stopped, cleaning up')
        cleanup_if_ice_manager()
        servant.finish()
        get_main_logger().info('ice image grab service ended')
        return 0


def main():
    set_thread_name("Main")
    log_folder = attach_file_handlers()

    # setup exception handling
    exception_handler = UnhandledExceptionHandler(main_window=None, logger=get_main_logger())
    exception_handler.setup_fault_handler(log_folder)
    sys.excepthook = exception_handler.handle_exception

    app = ImageGrabService()
    app_result = app.main(sys.argv)
    # noinspection PyProtectedMember
    num_current_frames = len(sys._current_frames())
    if num_current_frames > 1:
        get_main_logger().info(
            "Something went wrong, "
            "number of frames is {} while expected 1, dumping threads and self({})-destructing..".format(
                num_current_frames, os.getpid()
            ))
        dump_all_threads()
        sys.stderr.flush()
        os.system('kill {}'.format(os.getpid()))
    return app_result


if __name__ == '__main__':
    main()
