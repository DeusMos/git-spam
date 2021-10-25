# -*- coding: utf-8 -*-

import threading

from typing import Optional, List, Dict, Tuple

from pfe.common.utils.image.image_grab.image_grab import TriggerThreshold

from pfe.interfaces.cltb.gui.lib.project.images.image_false_coloring import ImageDisplayOptions

from pfe.common.configuration.system import System
from pfe.common.utils.display_options import IImageDisplayOptions, ImageVisual
from pfe.interfaces.recipe_builder.core.nodes.recipe_node import RecipeNode
from pfe.services.image_grab.definitions import ImageGrabData
from pfe.interfaces.cltb.gui.lib.export.cltb_reader import CltbFactory
from pfe.interfaces.common.exporter.common import NUM_ATTRIBUTES
from pfe.interfaces.recipe_builder.core.nodes.node_type import NodeType
from pfe.interfaces.recipe_builder.core.recipe_project import recipe_project
from pfe.services.image_grab.files.images import ImageCompatibilityInfo, Images
from pfe.services.image_grab.files.images_watch import KIFInfo
from pfe.services.image_grab.ice import ice_image_grab
from pfe.services.image_grab.loggers import get_main_logger
from pfe.services.image_grab.utils.decorators import methdispatch
from pfe.services.image_grab.utils.recipe import currently_loaded_project, get_last_loaded_recipe_uuid


def lock_and_check_ready(f):
    # noinspection PyProtectedMember
    def wrapper(*args):
        self_ = args[0]
        with self_._project_info_lock:
            if not self_._is_ready.is_set():
                get_main_logger().info("Project info: not ready, rebuilding.")
                self_._rebuild()
            if get_last_loaded_recipe_uuid() != self_._vpf_uuid:
                get_main_logger().info("Project info: current VPF UUID differs from Recipe Service one, rebuilding.")
                self_._rebuild()
            return f(*args)
    return wrapper


class ProjectInfo(object):
    def __init__(self):
        super(ProjectInfo, self).__init__()
        self._is_ready = threading.Event()  # type: threading.Event
        self._reset()
        self._project_info_lock = threading.Lock()
        self._rebuild_lock = threading.Lock()
        self._system = System()

    @property
    def is_ready(self):
        with self._project_info_lock:
            return self._is_ready.is_set()

    @property
    @lock_and_check_ready
    def sensor_groups(self):
        return self._sensor_groups

    @property
    @lock_and_check_ready
    def project_uuid(self):
        return self._project_uuid

    @property
    @lock_and_check_ready
    def recipe_name(self):
        return self._recipe_name

    @property
    @lock_and_check_ready
    def background_color(self):
        return self._background_color

    @property
    @lock_and_check_ready
    def eject_attribute_colors(self):
        return self._eject_attribute_colors

    @property
    @lock_and_check_ready
    def default_attribute_colors(self):
        return self._default_attribute_colors

    @property
    @lock_and_check_ready
    def display_options(self) -> Dict[str, IImageDisplayOptions]:
        return self._display_options

    @lock_and_check_ready
    def get_tile_name(self, sensor_group_uuid: str) -> str:
        return self._tile_names[sensor_group_uuid]

    @lock_and_check_ready
    def get_sensor_group_name(self, sensor_group_uuid: str) -> str:
        return self._sensor_group_names[sensor_group_uuid]

    @lock_and_check_ready
    def get_color_classifier_for_tnd_sensor_group(self, sensor_group_uuid):
        # type: (str) -> str
        return self._tnd_sensor_groups[sensor_group_uuid]

    @lock_and_check_ready
    def get_attribute_info(self, sensor_group_uuid):
        result = list()
        for attribute_index, (default_color, eject_color) in \
                enumerate(zip(self._default_attribute_colors, self._eject_attribute_colors)):
            attribute_info = ice_image_grab.AttributeInfo()
            attribute_info.index = attribute_index
            attribute_info.name = self._attribute_names[sensor_group_uuid][attribute_index]
            attribute_info.defaultColor = self.convert_tuple_color_to_hex(default_color)
            attribute_info.ejectColor = self.convert_tuple_color_to_hex(eject_color)
            attribute_info.inSensorGroup = False
            for sensor_group_uuid_, pipelines_, assignment_uuid_ in self._sensor_groups:
                if sensor_group_uuid_ == sensor_group_uuid and attribute_index in \
                        self._sensor_group_attributes[sensor_group_uuid]:
                    attribute_info.inSensorGroup = True
            result.append(attribute_info)
        return result

    @lock_and_check_ready
    def get_image_compatibility_info(self, sensor_group_uuid):
        if ImageGrabData.IMAGES_VALIDATION_ENABLED:
            return self._image_compatibility_info_per_sensor_group[sensor_group_uuid]
        else:
            return ImageCompatibilityInfo(
                display_options=self._image_compatibility_info_per_sensor_group[sensor_group_uuid].display_options)

    @lock_and_check_ready
    def get_sensor_groups_for_sensor_id(self, sensor_id):
        return self.__get_sensor_groups_for_sensor_id(sensor_id)

    def __get_sensor_groups_for_sensor_id(self, sensor_id):
        # type: (int) -> List[str]
        result = list()
        for sensor_group_uuid_, pipelines_, assignment_uuid_ in self._sensor_groups:
            if sensor_id in pipelines_:
                result.append(sensor_group_uuid_)
        return result

    @lock_and_check_ready
    def get_assignment_uuid_for_sensor_group(self, sensor_group_uuid):
        # type: (str) -> Optional[str]
        for uuid, _, assignment_uuid in self._sensor_groups:
            if sensor_group_uuid == uuid:
                return assignment_uuid
        return None

    @methdispatch
    @lock_and_check_ready
    def get_display_options(self, sensor_group_uuid: Optional[str] or KIFInfo or int) -> IImageDisplayOptions:
        return self.__get_display_options(sensor_group_uuid)

    @get_display_options.register(KIFInfo)
    @lock_and_check_ready
    def _(self, kif_info: KIFInfo) -> IImageDisplayOptions:
        # step 1: try to use sensor id
        for sensor in self._system.get_sensors():
            if sensor.name() == kif_info.sensor_name:
                sensor_id = sensor.pipeline()
                for sensor_group in self.__get_sensor_groups_for_sensor_id(sensor_id):
                    return self.__get_display_options(sensor_group)
        # step 2: using the sensor id didn't yield success, so we just get the first match based on number of channels
        num_channels = kif_info.number_of_channels
        for training_set_uuid, display_options in self._display_options.items():
            r, g, b = display_options.get_rgb()
            if r < num_channels and g < num_channels and b < num_channels:
                return display_options
        # step 3: return the fallback display options
        return self.__get_display_options(None)

    @get_display_options.register(int)
    @lock_and_check_ready
    def _(self, sensor_id: int) -> IImageDisplayOptions:
        for sensor_group in self.__get_sensor_groups_for_sensor_id(sensor_id):
            return self.__get_display_options(sensor_group)
        return self.__get_display_options(None)

    def __get_display_options(self, sensor_group_uuid: Optional[str]) -> IImageDisplayOptions:
        if sensor_group_uuid is not None and sensor_group_uuid in self._image_compatibility_info_per_sensor_group:
            return self._image_compatibility_info_per_sensor_group[sensor_group_uuid].display_options
        # fallback, just return PCA
        display_options = ImageDisplayOptions()
        display_options.display_method = ImageVisual.PCA_IMAGE
        return display_options

    @lock_and_check_ready
    def get_thresholds(self, sensor_group_uuid: str) -> List[TriggerThreshold]:
        return self._thresholds[sensor_group_uuid]

    @staticmethod
    def _parse_pipelines(pipelines):
        # type: (str) -> List[int]
        # noinspection PyBroadException
        try:
            return [int(pipeline) for pipeline in str(pipelines).split(",")]
        except Exception:
            return list()

    @staticmethod
    def convert_tuple_color_to_hex(color_as_tuple, alpha=False):
        r, g, b, a = color_as_tuple
        if alpha:
            return "#{:02x}{:02x}{:02x}{:02x}".format(int(r), int(g), int(b), int(a))
        else:
            return "#{:02x}{:02x}{:02x}".format(int(r), int(g), int(b))

    def _rebuild(self, rp_project_path: str = None, threaded: bool = False):
        if threaded:
            threading.Thread(target=self.__rebuild, args=(rp_project_path, )).start()
        else:
            self.__rebuild(rp_project_path)

    def __rebuild(self, rp_project_path=None):
        # type: (Optional[str]) -> None
        with self._rebuild_lock:
            get_main_logger().info("Project info: start rebuild.")
            if rp_project_path is None and get_last_loaded_recipe_uuid() == self._vpf_uuid:
                get_main_logger().info("Project info: rebuilding not needed VPF UUID up-to-date.")
                return
            self._reset()
            if rp_project_path is None:
                with currently_loaded_project(sync=True, max_num_tries=2, retry_timeout=0.5) as downloaded_recipe:
                    if downloaded_recipe is None:
                        get_main_logger().info("Project info: rebuilding failed, state is *not* ready.")
                    else:
                        self._recipe_name = downloaded_recipe.recipe_name
                        self._vpf_uuid = downloaded_recipe.vpf_uuid
                        self.__rebuild_with_rp_project(downloaded_recipe.rp_path)
                        self._is_ready.set()
                        get_main_logger().info("Project info: rebuilding done, state is ready.")
            else:
                # noinspection PyBroadException
                try:
                    self.__rebuild_with_rp_project(rp_project_path)
                    self._is_ready.set()
                    get_main_logger().info("Project info: rebuilding done, state is ready.")
                except Exception:
                    get_main_logger().info("Project info: rebuilding failed, state is *not* ready.")

    def __rebuild_with_rp_project(self, rp_project_path):
        # type: (str) -> None
        get_main_logger().info("Project info: rebuilding with project: '{}'.".format(rp_project_path))
        with recipe_project(rp_project_path, load_node_data=True) as rp:
            for tile_node in rp.recipe_node_data.get(NodeType.TILE):
                tile_name = tile_node.name

                sensor_group_nodes = tile_node.get_descendants_by_type(NodeType.SENSOR_GROUP)

                for sensor_group_node in sensor_group_nodes:
                    sensor_group_name = sensor_group_node.name
                    uuid = sensor_group_node.params.uuid

                    sensor_group_thresholds = list()
                    shape_nodes = sensor_group_node.get_descendants_by_type(NodeType.SHAPE)
                    for shape_node in shape_nodes:
                        threshold = TriggerThreshold()
                        threshold.attribute_id = shape_node.params.attribute
                        threshold.algorithm = shape_node.params.algorithm
                        threshold.threshold_index = shape_node.params.threshold
                        sensor_group_thresholds.append(threshold)
                    shape_combo_nodes = sensor_group_node.get_descendants_by_type(NodeType.SHAPE_COMBO)
                    for shape_combo_node in shape_combo_nodes:
                        threshold = TriggerThreshold()
                        threshold.attribute_id = shape_combo_node.params.attribute
                        threshold.algorithm = shape_combo_node.params.algorithm
                        threshold.threshold_index = 0
                        sensor_group_thresholds.append(threshold)
                    self._thresholds[uuid] = sensor_group_thresholds

                    self._tile_names[uuid] = tile_name
                    self._sensor_group_names[uuid] = sensor_group_name

                    pipelines = sensor_group_node.params.pipelines
                    assignment_uuid = sensor_group_node.assignment_table_uuid
                    self._sensor_groups.append((uuid, pipelines, assignment_uuid))
                    #
                    attributes = sensor_group_node.get_descendants_by_type(NodeType.ATTRIBUTE)
                    self._sensor_group_attributes[uuid] = [attribute.params.attribute for attribute in attributes]
                    #
                    if sensor_group_node.params.reteachable:
                        color_nodes = sensor_group_node.get_descendants_by_type(NodeType.COLOR)
                        if len(color_nodes) != 1:
                            get_main_logger().error(
                                "Found {} color nodes for reteachable sensor group node with uuid '{}'. "
                                "The number of color nodes should be exactly 1.".format(
                                    len(color_nodes), uuid)
                            )
                        else:
                            self._tnd_sensor_groups[uuid] = color_nodes[0].params.uuid

            recipe_node = rp.recipe_node_data.get(NodeType.RECIPE)[0]
            self._project_uuid = recipe_node.params.uuid
            self._background_color = self.get_background_color_from_recipe_node(recipe_node)
            self._default_attribute_colors = self.get_default_attribute_colors_from_recipe_node(recipe_node)
            self._eject_attribute_colors = self.get_eject_colors_from_recipe_node(recipe_node)

            with CltbFactory.get_read_access(rp.cltb_path(), setup_logging=False) as cltb_reader:
                for training_set_uuid in cltb_reader.get_training_set_uuids():
                    self._display_options[training_set_uuid] = \
                        cltb_reader.get_training_set_display_options(training_set_uuid)

                for sensor_group_uuid, pipelines, assignment_uuid in self._sensor_groups:
                    training_set_uuid = cltb_reader.get_training_set_uuid_of_assignment(assignment_uuid)
                    bit_depth, number_of_channels, is_signed = cltb_reader.get_training_set_compatibility_details(
                        training_set_uuid)
                    display_options = cltb_reader.get_training_set_display_options(training_set_uuid)
                    self._image_compatibility_info_per_sensor_group[sensor_group_uuid] = \
                        ImageCompatibilityInfo(bit_depth, number_of_channels, is_signed, display_options, pipelines)
                    self._attribute_names[sensor_group_uuid] = \
                        cltb_reader.get_all_attributes_index_to_name_dict(assignment_uuid)

    def rebuild(self, rp_project_path=None):
        # type: (Optional[str]) -> None
        with self._project_info_lock:
            self._rebuild(rp_project_path, threaded=True)

    def _reset(self):
        # type: () -> None
        get_main_logger().info("Project info: resetting.")
        self._recipe_name = ""  # type: str
        self._vpf_uuid = ""  # type: str
        self._project_uuid = ""  # type: str
        self._background_color = None  # type: Optional[str]
        self._eject_attribute_colors = list()  # type: List[str]
        self._default_attribute_colors = list()  # type: List[str]
        self._image_compatibility_info_per_sensor_group = dict()  # type: Dict[str, ImageCompatibilityInfo]
        self._sensor_groups = list()  # type: List[Tuple[str, List[int], str]]
        self._sensor_group_attributes = dict()  # type: Dict[str, List[int]]
        self._tile_names = dict()  # type: Dict[str, str]
        self._sensor_group_names = dict()  # type: Dict[str, str]
        self._tnd_sensor_groups = dict()  # type: Dict[str, str]
        self._attribute_names = dict()  # type: Dict[str, Dict[int, str]]
        self._is_ready.clear()  # type: threading.Event
        self._project_checksum = ""  # type: str
        self._display_options = dict()  # type: Dict[str, IImageDisplayOptions]
        self._thresholds = dict()  # type: Dict[str, List[TriggerThreshold]]

    def reset(self):
        # type: () -> None
        with self._project_info_lock, self._rebuild_lock:
            self._reset()

    @lock_and_check_ready
    def dump(self, logger=None):
        log = logger if logger is not None else get_main_logger()
        log.info("Project info")
        log.info("================================================")
        log.info("Colors")
        log.info("------------------------------------------------")
        log.info("Background color: {}".format(self._background_color))
        log.info("Default attribute colors: {}".format(", ".join(
            ["{}".format(color) for color in self._default_attribute_colors]
        )))
        log.info("Eject attribute colors: {}".format(", ".join(
            ["{}".format(color) for color in self._eject_attribute_colors]
        )))
        log.info("")
        log.info("Sensor groups")
        log.info("------------------------------------------------")
        for sensor_group_uuid, pipelines, assignment_uuid in self._sensor_groups:
            log.info("Sensor group '{}'".format(sensor_group_uuid))
            log.info("   pipelines: {}".format(", ".join([str(pipeline) for pipeline in pipelines])))
            log.info("   assignment UUID: {}".format(assignment_uuid))
            log.info("   attributes: {}".format(", ".join(
                ["{} ({})".format(attribute_name, attribute_index)
                 for attribute_index, attribute_name in self._attribute_names[sensor_group_uuid].items()
                 ])))
        log.info("================================================")

    @lock_and_check_ready
    def get_compatible_sensor_group_uuids(self, kif_file_path: str) -> List[str]:
        result = list()
        for sensor_group_uuid, compatibility_info in self._image_compatibility_info_per_sensor_group.items():
            if Images.check_compatibility_with_kif(kif_file_path, compatibility_info) is not None:
                result.append(sensor_group_uuid)
        return result

    ###################
    #

    @staticmethod
    def get_background_color_from_recipe_node(recipe_node):
        # type: (RecipeNode) -> str
        return recipe_node.params.color_background

    @staticmethod
    def get_default_attribute_colors_from_recipe_node(recipe_node):
        # type: (RecipeNode) -> List[str]
        return [getattr(recipe_node.params, "color_attribute_{}".format(i)) for i in range(NUM_ATTRIBUTES)]

    @staticmethod
    def get_eject_colors_from_recipe_node(recipe_node):
        # type: (RecipeNode) -> List[str]
        return [getattr(recipe_node.params, "eject_marker_attribute_{}".format(i)) for i in range(NUM_ATTRIBUTES)]
