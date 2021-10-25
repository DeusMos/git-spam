# -*- coding: utf-8 -*-

import weakref
from multiprocessing import Event
from queue import Empty, Queue
from typing import Optional, List

from pfe.common.ice.converters import to_type
from pfe.common.ice.ice_manager import ice_manager
from pfe.common.utils.singleton import Singleton
from pfe.services.image_grab.database import DB
from pfe.services.image_grab.definitions import ImageGrabData
from pfe.services.image_grab.ice import ice_recipes, ice_image_grab, ice_discovery
from pfe.services.image_grab.loggers import get_main_logger
from pfe.services.image_grab.utils.functions import db_item_to_history_item


class RecipeLoaderObserverImpl(ice_recipes.RecipeLoaderObserver):
    def __init__(self, image_grab_service):
        super(RecipeLoaderObserverImpl, self).__init__()
        #
        self._image_grab_service = image_grab_service

    # noinspection PyPep8Naming,PyUnusedLocal
    def recipeLoaded(self, data, current=None):
        get_main_logger().debug("Recipe loaded triggered!")
        self._image_grab_service.rebuild_project_info()


class RecipeUnloaderObserverImpl(ice_recipes.RecipeUnloaderObserver):
    def __init__(self, image_grab_service):
        super(RecipeUnloaderObserverImpl, self).__init__()
        #
        self._image_grab_service = image_grab_service

    # noinspection PyPep8Naming,PyUnusedLocal
    def recipeUnloaded(self, data, current=None):
        get_main_logger().debug("Recipe unloaded triggered!")
        self._image_grab_service.reset_project_info()


class RecipeModifiedObserverImpl(ice_recipes.RecipeModifiedObserver):
    def __init__(self, image_grab_service):
        super(RecipeModifiedObserverImpl, self).__init__()
        #
        self._image_grab_service = image_grab_service

    # noinspection PyPep8Naming,PyUnusedLocal,PyMethodMayBeStatic
    def recipeModified(self, modified, current=None):
        get_main_logger().debug("Recipe modified!")


class SettingObserverI(ice_discovery.SettingObserver):
    def __init__(self, setting, callback):
        super(SettingObserverI, self).__init__()
        self._setting_id = setting.setting_id
        self._setting = weakref.ref(setting)
        self._callback = weakref.ref(callback)

    # noinspection PyPep8Naming,PyUnusedLocal
    def settingUpdateAvailable(self, byte_array, current=None):
        setting = self._setting()
        callback = self._callback()
        if setting is not None and callback is not None:
            value = to_type(setting.value_type, byte_array, setting.bit_size)
            # noinspection PyProtectedMember
            if setting._old_value is None or value != setting._old_value:
                callback(value)


class PublishHistoryChanges(metaclass=Singleton):
    def __init__(self):
        super(PublishHistoryChanges, self).__init__()
        self._allow_history_publish = Event()
        self._allow_history_publish.set()
        self._queue = Queue()

    def block_publishing(self):
        self._allow_history_publish.clear()

    def allow_publishing(self):
        if not self._allow_history_publish.is_set():
            self.__publish_queue()
            self._allow_history_publish.set()

    @staticmethod
    def __get_publisher() -> Optional[ice_image_grab.HistoryChangedObserverPrx]:
        return ice_manager().get_publisher_for_topic(ice_image_grab.HistoryChangedObserverPrx, ImageGrabData.HISTORY_CHANGED_TOPIC)

    def __publish_queue(self, publisher: Optional[ice_image_grab.HistoryChangedObserverPrx] = None):
        if publisher is None:
            publisher = self.__get_publisher()
        if publisher is None:
            get_main_logger().error("Failed to get publisher for history changed topic, queue changes *not* sent.")
            return
        #
        while True:
            try:
                changes = self._queue.get_nowait()
                publisher.changed(changes)
            except Empty:
                return

    def publish_change(self, history_id: int, trigger_type: ice_image_grab.TriggerType, history_changed_event: ice_image_grab.HistoryChangedEvent,
                       sensor_group_ids: Optional[List[str]] = None) -> None:
        history_change = ice_image_grab.HistoryChange()
        if history_changed_event == ice_image_grab.HistoryChangedEvent.REMOVED:
            # Item has just been removed from DB so no use to look it up.
            history_item = ice_image_grab.HistoryItem()
            history_item.historyId = history_id
            history_item.trigger = trigger_type
            history_change.historyItem = history_item
        else:
            history_change.historyItem = db_item_to_history_item(DB.get(history_id))
        history_change.event = history_changed_event
        history_change.sensorGroupIds = list() if sensor_group_ids is None else sensor_group_ids
        self.publish_changes(history_change)

    def publish_changes(self, *history_changes: ice_image_grab.HistoryChange) -> None:
        if self._allow_history_publish.is_set():
            publisher = self.__get_publisher()
            if publisher is not None:
                self.__publish_queue(publisher)
                publisher.changed(list(history_changes))
            else:
                get_main_logger().error("Failed to get publisher for history changed topic, changes *not* sent.")
        else:
            get_main_logger().info("PublishHistoryChanges: publishing is blocked")
            self._queue.put(list(history_changes))


def publish_history_changes(*history_changes: ice_image_grab.HistoryChange):
    PublishHistoryChanges().publish_changes(*history_changes)


def publish_history_change(history_id: int, trigger_type: ice_image_grab.TriggerType, history_changed_event: ice_image_grab.HistoryChangedEvent,
                           sensor_group_ids: Optional[List[str]] = None):
    PublishHistoryChanges().publish_change(history_id, trigger_type, history_changed_event, sensor_group_ids)


def publish_simulation_invalidated(timestamp: float, sensor_group_id: str, simulator_id: str):
    publisher = ice_manager().get_publisher_for_topic(
        ice_image_grab.SimulationInvalidObserverPrx, ImageGrabData.SIMULATION_INVALIDATED_TOPIC)
    if publisher is not None:
        event = ice_image_grab.SimulationInvalidated()
        event.timestamp = timestamp
        event.sensorGroupId = sensor_group_id
        event.simulatorId = simulator_id
        publisher.simulationInvalidated(event)
    else:
        get_main_logger().error("Failed to get publisher for simulation invalidated topic, event *not* sent.")


def publish_finished(identifier: str, method: Optional[str], success: bool, message: str) -> None:
    method = method if method is not None else "<Undefined>"
    get_main_logger().debug("Publishing finished: identifier: {} / method: {} / success: {} / message: {}.".format(
        identifier, method, success, message
    ))
    result = ice_image_grab.Result()
    result.identifier = identifier
    result.method = method
    result.success = success
    result.message = message
    #
    publisher = ice_manager().get_publisher_for_topic(ice_image_grab.FinishedObserverPrx, ImageGrabData.PROGRESS_TOPIC)
    if publisher is not None:
        publisher.finished(result)
    else:
        get_main_logger().error("Failed to get publisher for finished topic, event *not* sent. Event identifier: {}".format(
            identifier
        ))
