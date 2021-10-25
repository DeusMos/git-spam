# -*- coding: utf-8 -*-

import re
from contextlib import contextmanager
from typing import Union, Optional

from pfe.common.utils.db.db_sqlite import Json, DBSQLite
from pfe.common.utils.db.history import History
from pfe.services.image_grab.definitions import ImageGrabData
from pfe.services.image_grab.ice import ice_image_grab
from pfe.services.image_grab.loggers import get_main_logger


@contextmanager
def open_database(open_history_database=None):
    # type: (Optional[DB]) -> DB
    if open_history_database is not None:
        yield open_history_database
    else:
        with DB() as db:
            yield db


class DB(object):
    # History flags
    HISTORY_ITEM_NOCACHE = 1 << 0

    HISTORY_LATEST_VERSION = 14
    # Order is important, newest last
    HISTORY_VERSIONS = [6, 7, 8, 9, 10, 11, 12, 13, 14]

    def __init__(self, db_path=None, version=HISTORY_LATEST_VERSION):
        super(DB, self).__init__()
        self._db_path = ImageGrabData.HISTORY_DB_PATH if db_path is None else db_path
        self._version = version
        self._db = self.__check_and_create_db()

    ######################
    # <BUILD FUNCTIONS>  #
    ######################

    @staticmethod
    def _build_triggered_by(**old_key_values):
        if 'triggered' not in old_key_values and 'fm_alert' not in old_key_values:
            return ice_image_grab.TriggerType.UNKOWN.value

        triggered = old_key_values['triggered'] == 1 if 'triggered' in old_key_values else False
        fm_alert = old_key_values['fm_alert'] == 1 if 'fm_alert' in old_key_values else False

        if not triggered:
            return ice_image_grab.TriggerType.MANUAL.value
        else:
            if fm_alert:
                return ice_image_grab.TriggerType.FMALERT.value
            else:
                return ice_image_grab.TriggerType.TRIGGERED.value

    @staticmethod
    def _build_functions():
        return {
            'triggered_by': DB._build_triggered_by
        }

    ######################
    # </BUILD FUNCTIONS> #
    ######################

    def __check_and_create_db(self):
        current_version = None
        with DBSQLite(self._db_path) as db:
            for version in self.HISTORY_VERSIONS:
                if db.table_exists(self._history_table_name(version)):
                    current_version = version

        if current_version is not None and current_version != self._version:
            get_main_logger().debug("Updating DB at '{}' from version {} to {}.".format(
                self._db_path,
                current_version,
                self._version
            ))
            History.alter_table(
                db_path=self._db_path,
                from_table_name=self._history_table_name(current_version),
                to_table_name=self._history_table_name(self._version),
                to_table_fields=[(f[0], f[1], f[2]) for f in self.fields(self._version)],
                delete_old_table=False,
                conversion_functions=None,
                build_functions=self._build_functions()
            )

        fields = list([(field[0], field[1]) for field in self.fields(self._version)])
        return History(self._db_path, self._history_table_name(self._version), fields)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._db.close()

    @classmethod
    def _history_table_name(cls, version=HISTORY_LATEST_VERSION):
        return "image_grab_service_v{version}".format(version=version)

    @classmethod
    def _from_version(cls, from_version):
        return cls.HISTORY_VERSIONS[cls.HISTORY_VERSIONS.index(from_version):]

    @classmethod
    def fields(cls, version=HISTORY_LATEST_VERSION):
        # (field name, type, default value, available in version(s))
        fields = [
            # sensor id
            ('sensor_id', int, -1, cls._from_version(6)),
            # sensor name
            ('sensor_name', str, "", cls._from_version(7)),
            # saved: 1 or not saved: 0
            ('saved', int, 0, cls._from_version(6)),
            # grab images was triggered: 1 or not triggered: 0
            ('triggered', int, 0, cls._from_version(6)),
            # true if was triggered by an fm alert
            ('fm_alert', int, 0, cls._from_version(10)),
            # trigger type: UNKNOWN, MANUAL, TRIGGERED or FMALERT, as defined by triggered and fm_alert field,
            # added for convenience.
            # See enum TriggerType defined in slice.
            ('triggered_by', int, ice_image_grab.TriggerType.UNKNOWN.value, cls._from_version(11)),
            # if triggered, trigger_thresholds are stored here in following format:
            # for each threshold: ATTR(<attribute_id>)ALGO(<algorithm_index>)THRES(<threshold_index>)
            # joined together with semicolons (;)
            ('trigger_thresholds', str, "", cls._from_version(6)),
            # if triggered given a sensor group id, this id is stored here.
            ('trigger_sensor_group_id', str, "", cls._from_version(6)),
            # if triggered given a sensor group id, this name is stored here.
            ('trigger_sensor_group_name', str, "", cls._from_version(7)),
            # from which tile was the image grab done
            ('tile_name', str, "", cls._from_version(7)),
            # Short (user specified) description for the image
            ('description', str, "", cls._from_version(8)),
            # image width
            ('width', int, 0, cls._from_version(6)),
            # image height
            ('height', int, 0, cls._from_version(6)),
            # extra flags
            ('flags', int, 0, cls._from_version(6)),
            # project checksum
            ('project_uuid', str, "", cls._from_version(6)),
            # project recipe (.vpf) UUID
            ('project_vpf_uuid', str, "", cls._from_version(8)),
            # recipe builder (.rp) project UUID
            ('project_rp_uuid', str, "NO_UUID", cls._from_version(8)),
            # project (operator UI) name
            ('project_name', str, "", cls._from_version(7)),
            # extra data as json str
            ('data', Json, None, cls._from_version(7)),
            # is image available to all?
            ('available_to_all', int, 0, cls._from_version(13)),
            # image was imported by user
            ('user_imported', int, 0, cls._from_version(14)),
        ]

        image_types = [
            # original image types
            (ImageGrabData.RAW_IMAGE, None),
            (ImageGrabData.SEGMENTATION_OVERLAY, None),
            (ImageGrabData.OVERALL_SEGMENTATION_OVERLAY, None),
            (ImageGrabData.EJECTS_OVERLAY, None),
            (ImageGrabData.OVERALL_EJECTS_OVERLAY, None),
            (ImageGrabData.CONTOURS_OVERLAY, None),
            (ImageGrabData.OVERALL_CONTOURS_OVERLAY, None),
            (ImageGrabData.ALGORITHMS_OVERLAY, None),
            (ImageGrabData.OVERALL_ALGORITHMS_OVERLAY, None),
            # introduced from DB version 9, so all fields available from version 9
            (ImageGrabData.BACKGROUND_OVERLAY, cls._from_version(9)),
            (ImageGrabData.SELECTED_REGION_OVERLAY, cls._from_version(12)),
        ]

        def _v(__image_type_from_version, __default):
            return __default if __image_type_from_version is None else __image_type_from_version

        for image_type_info in image_types:
            image_type, from_version = image_type_info
            fields.append((cls.project_checksum_field(image_type), str, "", _v(from_version, cls._from_version(6))))
            fields.append((cls.project_name_field(image_type), str, "", _v(from_version, cls._from_version(7))))
            fields.append((cls.sensor_group_id_field(image_type), str, "", _v(from_version, cls._from_version(6))))
            fields.append((cls.sensor_group_name_field(image_type), str, "", _v(from_version, cls._from_version(7))))
            fields.append((cls.tile_name_field(image_type), str, "", _v(from_version, cls._from_version(7))))

        return list([field for field in fields if version in field[3]])

    @classmethod
    def default_value(cls, field, version=HISTORY_LATEST_VERSION):
        for f in cls.fields(version=version):
            if f[0] == field:
                return f[2]
        return None

    @classmethod
    def project_checksum_field(cls, image_type):
        return "{}_project_checksum".format(image_type)

    @classmethod
    def project_name_field(cls, image_type):
        return "{}_project_name".format(image_type)

    @classmethod
    def sensor_group_id_field(cls, image_type):
        return "{}_sensor_group_id".format(image_type)

    @classmethod
    def sensor_group_name_field(cls, image_type):
        return "{}_sensor_group_name".format(image_type)

    @classmethod
    def tile_name_field(cls, image_type):
        return "{}_tile_name".format(image_type)

    @classmethod
    def _default_values(cls, version=HISTORY_LATEST_VERSION, **kwargs):
        values = list()
        for field in cls.fields(version):
            if field[0] in kwargs:
                values.append(kwargs[field[0]])
            else:
                values.append(field[2])
        return values

    def add(self, **kwargs):
        # check it data field is already wrapped in a Json object
        # if not, let's do it here, before handing it to the db
        if 'data' in kwargs and type(kwargs['data']) is not Json:
            # wrap it
            kwargs['data'] = Json(kwargs['data'])
        return self._db.add(*self._default_values(**kwargs))

    def update(self, history_id, **to_update):
        tuples = list()
        for key, value in to_update.items():
            # check it data field is already wrapped in a Json object
            # if not, let's do it here, before handing it to the db
            if key == 'data' and type(value) is not Json:
                tuples.append((key, Json(value)))
            else:
                tuples.append((key, value))

        self._db.update(history_id, tuples)

    def dump(self):
        print(", ".join([field[0] for field in self.fields(self._version)]))
        for item in self._db:
            print(", ".join([str(i) for i in item]))

    def where_flags(self, flags, history_id=None):
        if history_id is None:
            return list(self._db.find_where("flags & {} = {}".format(flags, flags)))
        else:
            return list(self._db.find_where("flags & {} = {} AND {} = {}".format(
                flags, flags, History.ID_COLUMN, history_id)))

    def has_flags(self, flags, history_id):
        return len(self.where_flags(flags, history_id)) == 1

    def __contains__(self, history_id):
        return self._db.has_id(history_id)

    def __getitem__(self, history_id):
        filtered = list(self._db.filter_by([(History.ID_COLUMN, history_id)]))
        if len(filtered) != 1:
            raise ice_image_grab.RunTimeError("Invalid history id: {}.".format(history_id))
        history_item = filtered[0]
        return history_item

    def get_data(self, history_id):
        """
        returns the *actual* value of the data field (so already decoded)
        """
        return self[history_id].data.value

    def update_data(self, history_id, data):
        # type: (int, Union[dict, list]) -> None
        item = self[history_id]
        if item.data and type(item.data.value) is dict and type(data) is dict:
            # merge it
            new_data = item.data.value.copy()
            new_data.update(data)
            self.update(history_id, data=new_data)
        else:
            # replace it
            self.update(history_id, data=data)

    @property
    def filter(self):
        return self._db.where

    def count(self, **to_find):
        return self._db.count([
            (key, value) for key, value in to_find.items()
        ])

    def ordered_by(self, order_by_list, asc=True, limit=None, **to_find):
        return self._db.ordered_by([
            (key, value) for key, value in to_find.items()
        ], order_by_list, asc, limit)

    def remove(self, history_id, commit=True):
        self._db.remove(history_id, commit=commit)

    @classmethod
    def get(cls, history_id, db_path=None):
        with DB(db_path) as db:
            return db[history_id]

    @classmethod
    def update_img(cls, history_id, img_type, db_path=ImageGrabData.HISTORY_DB_PATH, **kwargs):

        fields = DB.fields()
        field_names = [field[0] for field in fields]
        values = {}

        def _field_name(field_):
            if img_type == ImageGrabData.KIF:
                return field_
            else:
                prefixed = {
                    'project_checksum': DB.project_checksum_field(img_type),
                    'project_name': DB.project_name_field(img_type),
                    'sensor_group_id': DB.sensor_group_id_field(img_type),
                    'sensor_group_name': DB.sensor_group_name_field(img_type),
                    'tile_name': DB.tile_name_field(img_type)
                }
                try:
                    return prefixed[field_]
                except KeyError:
                    return field_

        for key, value in kwargs.items():
            field_name = _field_name(key)
            if value is not None and field_name in field_names:
                values[field_name] = value

        with DB(db_path) as db:
            db.update(history_id, **values)

    def commit(self):
        self._db.save()

    def __iter__(self):
        return self._db.__iter__()


class Conversions(object):
    def __init__(self):
        super(Conversions, self).__init__()

    @staticmethod
    def trigger_threshold_tuples_to_db(*thresholds):
        """
        Tuple order should be: (attribute, algorithm_id, threshold_index).
        """
        return ";".join(["ATTR({})ALGO({})THRES({})".format(attr, algo, thres) for (attr, algo, thres) in thresholds])

    @staticmethod
    def trigger_thresholds_to_db(thresholds):
        return ";".join(list(
            ["ATTR({})ALGO({})THRES({})".format(t.attributeId, t.algorithmIndex, t.thresholdIndex) for t in
             thresholds]))

    trigger_threshold_format = re.compile(r"ATTR\((\d+)\)ALGO\((\d+)\)THRES\((\d+)\)")

    @classmethod
    def trigger_thresholds_from_db(cls, thresholds):
        thresholds = thresholds.split(';')
        results = list()
        for threshold in thresholds:
            result = cls.trigger_threshold_format.match(threshold)
            if result is not None:
                threshold = ice_image_grab.Threshold()
                threshold.attributeId = int(result.group(1))
                threshold.algorithmIndex = int(result.group(2))
                threshold.thresholdIndex = int(result.group(3))
                results.append(threshold)
        return results

    @classmethod
    def trigger_type_value_to_trigger_type(cls, value):
        return {
            0: ice_image_grab.TriggerType.UNKNOWN,
            1: ice_image_grab.TriggerType.MANUAL,
            2: ice_image_grab.TriggerType.TRIGGERED,
            3: ice_image_grab.TriggerType.FMALERT,
            4: ice_image_grab.TriggerType.FROMFILE,
            5: ice_image_grab.TriggerType.MULTIMANUAL,
            6: ice_image_grab.TriggerType.MULTISCHEDULED,
            7: ice_image_grab.TriggerType.MULTITRIGGERED,
            8: ice_image_grab.TriggerType.AIADDON,
        }[value]


def check_current_history_count(project_uuid, open_history_database=None, raise_exception=True):
    # type: (Optional[DB], bool) -> int
    with open_database(open_history_database) as history:
        total_count = history.count(saved=1, project_rp_uuid=project_uuid)
        fm_count = history.count(saved=1, project_rp_uuid=project_uuid,
                                 triggered_by=ice_image_grab.TriggerType.FMALERT.value)

        if total_count >= fm_count:
            count = total_count - fm_count
        else:
            raise ice_image_grab.RunTimeError(
                "Incorrect saved images state. "
                "Total project images count ({}) should be higher or equal to the project FM images count ({})".format(
                    total_count, fm_count))

        get_main_logger().debug("Allowed saved items {}, "
                                "current project images count {} "
                                "(total project images count {}, project FM images count {})".format(
            ImageGrabData.MAX_HISTORY, count, total_count, fm_count))
        if raise_exception and count >= ImageGrabData.MAX_HISTORY:
            raise ice_image_grab.MaxSavedImages(
                "Maximum number ({}) of saved items per project reached! Current count {}.".format(
                    ImageGrabData.MAX_HISTORY, count),
                count,
                ImageGrabData.MAX_HISTORY
            )
        return count
