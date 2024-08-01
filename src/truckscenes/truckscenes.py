# Copyright 2021 Motional
# Copyright 2024 MAN Truck & Bus SE

from __future__ import annotations

import json
import os.path as osp
import sys
import time
import warnings

from bisect import bisect_left
from collections import OrderedDict
from importlib import import_module
from typing import Dict, List, Tuple, Union

import numpy as np

from pyquaternion import Quaternion

from truckscenes.utils import colormap
from truckscenes.utils.data_classes import Box
from truckscenes.utils.geometry_utils import box_in_image, BoxVisibility

PYTHON_VERSION = sys.version_info[0]

if not PYTHON_VERSION == 3:
    raise ValueError("truckscenes dev-kit only supports Python version 3.")


class TruckScenes:
    """
    Database class for truckscenes to help query and retrieve information from the database.
    """

    def __init__(self,
                 version: str = 'v1.0-mini',
                 dataroot: str = '/data/truckscenes',
                 verbose: bool = True):
        """
        Loads database and creates reverse indexes and shortcuts.

        Arguments:
            version: Version to load (e.g. "v1.0-mini", ...).
            dataroot: Path to the tables and data.
            verbose: Whether to print status messages during load.
        """
        self.version = version
        self.dataroot = dataroot
        self.verbose = verbose
        self.table_names = ['attribute', 'calibrated_sensor', 'category', 'ego_motion_cabin',
                            'ego_motion_chassis', 'ego_pose', 'instance', 'sample',
                            'sample_annotation', 'sample_data', 'scene', 'sensor', 'visibility']

        assert osp.exists(self.table_root), \
            f'Database version not found: {self.table_root}'

        start_time = time.time()
        if verbose:
            print(f"======\nLoading truckscenes tables for version {self.version}...")

        # Explicitly assign tables to help the IDE determine valid class members.
        self.attribute = self.__load_table__('attribute')
        self.calibrated_sensor = self.__load_table__('calibrated_sensor')
        self.category = self.__load_table__('category')
        self.ego_motion_cabin = self.__load_table__('ego_motion_cabin')
        self.ego_motion_chassis = self.__load_table__('ego_motion_chassis')
        self.ego_pose = self.__load_table__('ego_pose')
        self.instance = self.__load_table__('instance')
        self.sample = self.__load_table__('sample')
        self.sample_annotation = self.__load_table__('sample_annotation')
        self.sample_data = self.__load_table__('sample_data')
        self.scene = self.__load_table__('scene')
        self.sensor = self.__load_table__('sensor')
        self.visibility = self.__load_table__('visibility')

        # Initialize the colormap which maps from class names to RGB values.
        self.colormap = colormap.get_colormap()

        if verbose:
            for table in self.table_names:
                print("{} {},".format(len(getattr(self, table)), table))
            print("Done loading in {:.3f} seconds.\n======".format(time.time() - start_time))

        # Make reverse indexes for common lookups.
        self.__make_reverse_index__(verbose)

        # Initialize TruckScenesExplorer class.
        try:
            explorer = getattr(import_module("truckscenes.utils.visualization_utils"),
                               "TruckScenesExplorer")
        except ModuleNotFoundError:
            warnings.warn('''The visualization dependencies are not installed on your system! '''
                          '''Run 'pip install "truckscenes-devkit[all]"'.''')
        else:
            self.explorer = explorer(self)

    @property
    def table_root(self) -> str:
        """ Returns the folder where the tables are stored for the relevant version. """
        return osp.join(self.dataroot, self.version)

    def __load_table__(self, table_name) -> dict:
        """ Loads a table. """
        with open(osp.join(self.table_root, '{}.json'.format(table_name))) as f:
            table = json.load(f)
        return table

    def __make_reverse_index__(self, verbose: bool) -> None:
        """
        De-normalizes database to create reverse indices for common cases.

        Arguments:
            verbose: Whether to print outputs.
        """
        start_time = time.time()
        if verbose:
            print("Reverse indexing ...")

        # Store the mapping from token to table index for each table.
        self._token2ind = dict()
        for table in self.table_names:
            self._token2ind[table] = dict()

            for ind, member in enumerate(getattr(self, table)):
                self._token2ind[table][member['token']] = ind

        # Decorate (adds short-cut) sample_annotation table with for category name.
        for record in self.sample_annotation:
            inst = self.get('instance', record['instance_token'])
            record['category_name'] = self.get('category', inst['category_token'])['name']

        # Decorate (adds short-cut) sample_data with sensor information.
        for record in self.sample_data:
            cs_record = self.get('calibrated_sensor', record['calibrated_sensor_token'])
            sensor_record = self.get('sensor', cs_record['sensor_token'])
            record['sensor_modality'] = sensor_record['modality']
            record['channel'] = sensor_record['channel']

        # Reverse-index samples with sample_data and annotations.
        for record in self.sample:
            record['data'] = {}
            record['anns'] = []

        for record in self.sample_data:
            if record['is_key_frame']:
                sample_record = self.get('sample', record['sample_token'])
                sample_record['data'][record['channel']] = record['token']

        for ann_record in self.sample_annotation:
            sample_record = self.get('sample', ann_record['sample_token'])
            sample_record['anns'].append(ann_record['token'])

        # Reverse-index timestamp to token for fast closest search
        self._timestamp2token = dict()
        for table in ['ego_pose', 'ego_motion_cabin', 'ego_motion_chassis',
                      'sample', 'sample_data']:
            tt = [(elem['timestamp'], elem['token']) for elem in getattr(self, table)]
            tt = sorted(tt, key=lambda e: e[0])
            self._timestamp2token[table] = OrderedDict(tt)

        if verbose:
            print(
                "Done reverse indexing in {:.1f} seconds.\n======".format(time.time() - start_time)
            )

    def get(self, table_name: str, token: str) -> dict:
        """
        Returns a record from table in constant runtime.

        Arguments:
            table_name: Table name.
            token: Token of the record.

        Returns:
            Table record. See README.md for record details for each table.
        """
        assert table_name in self.table_names, "Table {} not found".format(table_name)
        return getattr(self, table_name)[self.getind(table_name, token)]

    def getind(self, table_name: str, token: str) -> int:
        """
        This returns the index of the record in a table in constant runtime.

        Arguments:
            table_name: Table name.
            token: Token of the record.

        Returns:
            The index of the record in table, table is an array.
        """
        return self._token2ind[table_name][token]

    def getclosest(self, table_name: str, timestamp: int) -> dict:
        """
        This returns the element with the closest timestamp.
        Complexity: O(log n)

        Source: Lauritz V. Thaulow - https://stackoverflow.com/questions/12141150\
            /from-list-of-integers-get-number-closest-to-a-given-value

        Arguments:
            table_name: Table name.
            timestamp: Timestamp to compare with.

        Returns:
            Element of the table with the closest timestamp.
        """
        assert table_name in {'ego_pose', 'ego_motion_cabin', 'ego_motion_chassis',
                              'sample', 'sample_data'}, \
            f"Table {table_name} has no timestamp"

        # Helper function
        def _getclosest(timestamps, t):
            """
            Assumes myList is sorted. Returns closest value to myNumber.
            If two numbers are equally close, return the smallest number.
            """
            pos = bisect_left(timestamps, t)
            if pos == 0:
                return timestamps[0]
            if pos == len(timestamps):
                return timestamps[-1]
            before = timestamps[pos - 1]
            after = timestamps[pos]
            if after - t < t - before:
                return after
            else:
                return before

        # Find closest timestamp in given table (name)
        closest_timestamp = _getclosest(list(self._timestamp2token[table_name].keys()), timestamp)

        return self.get(table_name, self._timestamp2token[table_name][closest_timestamp])

    def field2token(self, table_name: str, field: str, query) -> List[str]:
        """
        This function queries all records for a certain field value,
        and returns the tokens for the matching records.

        Warning: this runs in linear time.

        Arguments:
            table_name: Table name.
            field: Field name. See README.md for details.
            query: Query to match against. Needs to type match the content of the query field.

        Returns:
            List of tokens for the matching records.
        """
        matches = []
        for member in getattr(self, table_name):
            if member[field] == query:
                matches.append(member['token'])
        return matches

    def get_sample_data_path(self, sample_data_token: str) -> str:
        """ Returns the path to a sample_data. """

        sd_record = self.get('sample_data', sample_data_token)
        return osp.join(self.dataroot, sd_record['filename'])

    def get_sample_data(self, sample_data_token: str,
                        box_vis_level: BoxVisibility = BoxVisibility.ANY,
                        selected_anntokens: List[str] = None,
                        use_flat_vehicle_coordinates: bool = False) -> \
            Tuple[str, List[Box], np.ndarray]:
        """
        Returns the data path as well as all annotations related to that sample_data.
        Note that the boxes are transformed into the current sensor's coordinate frame.

        Arguments:
            sample_data_token: Sample_data token.
            box_vis_level: If sample_data is an image, this sets required visibility for boxes.
            selected_anntokens: If provided only return the selected annotation.
            use_flat_vehicle_coordinates: Instead of the current sensor's coordinate frame,
                use ego frame which is aligned to z-plane in the world.

        Returns:
            (data_path, boxes, camera_intrinsic <np.array: 3, 3>)
        """
        # Retrieve sensor & pose records
        sd_record = self.get('sample_data', sample_data_token)
        cs_record = self.get('calibrated_sensor', sd_record['calibrated_sensor_token'])
        sensor_record = self.get('sensor', cs_record['sensor_token'])
        s_record = self.get('sample', sd_record['sample_token'])
        s_pose_record = self.getclosest('ego_pose', s_record['timestamp'])

        data_path = self.get_sample_data_path(sample_data_token)

        if sensor_record['modality'] == 'camera':
            cam_intrinsic = np.array(cs_record['camera_intrinsic'])
            imsize = (sd_record['width'], sd_record['height'])
        else:
            cam_intrinsic = None
            imsize = None

        # Retrieve all sample annotations and map to sensor coordinate system.
        if selected_anntokens is not None:
            boxes = list(map(self.get_box, selected_anntokens))
        else:
            boxes = self.get_boxes(sample_data_token)

        # Make list of Box objects including coord system transforms.
        box_list = []
        for box in boxes:
            if use_flat_vehicle_coordinates:
                # Move box to ego vehicle coord system parallel to world z plane.
                sd_yaw = Quaternion(s_pose_record['rotation']).yaw_pitch_roll[0]
                box.translate(-np.array(s_pose_record['translation']))
                box.rotate(Quaternion(scalar=np.cos(sd_yaw / 2),
                                      vector=[0, 0, np.sin(sd_yaw / 2)]).inverse)

                # Rotate upwards
                box.rotate(Quaternion(axis=box.orientation.rotate([0, 0, 1]), angle=np.pi/2))
            else:
                # Move box to ego vehicle coord system.
                box.translate(-np.array(s_pose_record['translation']))
                box.rotate(Quaternion(s_pose_record['rotation']).inverse)

                # Move box to sensor coord system.
                box.translate(-np.array(cs_record['translation']))
                box.rotate(Quaternion(cs_record['rotation']).inverse)

            if sensor_record['modality'] == 'camera' and not \
                    box_in_image(box, cam_intrinsic, imsize, vis_level=box_vis_level):
                continue

            box_list.append(box)

        return data_path, box_list, cam_intrinsic

    def get_box(self, sample_annotation_token: str) -> Box:
        """
        Instantiates a Box class from a sample annotation record.

        Arguments:
            sample_annotation_token: Unique sample_annotation identifier.
        """
        record = self.get('sample_annotation', sample_annotation_token)
        return Box(record['translation'], record['size'], Quaternion(record['rotation']),
                   name=record['category_name'], token=record['token'])

    def get_boxes(self, sample_data_token: str) -> List[Box]:
        """
        Instantiates Boxes for all annotation for a particular sample_data record.
        If the sample_data is a keyframe, this returns the annotations for that sample.
        But if the sample_data is an intermediate sample_data, a linear interpolation
        is applied to estimate the location of the boxes at the time the sample_data was captured.

        Arguments:
            sample_data_token: Unique sample_data identifier.
        """
        # Retrieve sensor & pose records
        sd_record = self.get('sample_data', sample_data_token)
        curr_sample_record = self.get('sample', sd_record['sample_token'])

        if curr_sample_record['prev'] == "" or sd_record['is_key_frame']:
            # If no previous annotations available,
            # or if sample_data is keyframe just return the current ones.
            boxes = list(map(self.get_box, curr_sample_record['anns']))

        else:
            prev_sample_record = self.get('sample', curr_sample_record['prev'])

            curr_ann_recs = [
                self.get('sample_annotation', token) for token in curr_sample_record['anns']
            ]
            prev_ann_recs = [
                self.get('sample_annotation', token) for token in prev_sample_record['anns']
            ]

            # Maps instance tokens to prev_ann records
            prev_inst_map = {entry['instance_token']: entry for entry in prev_ann_recs}

            t0 = prev_sample_record['timestamp']
            t1 = curr_sample_record['timestamp']
            t = sd_record['timestamp']

            # There are rare situations where the timestamps in the DB are off
            # so ensure that t0 < t < t1.
            t = max(t0, min(t1, t))

            boxes = []
            for curr_ann_rec in curr_ann_recs:

                if curr_ann_rec['instance_token'] in prev_inst_map:
                    # If the annotated instance existed in the previous frame,
                    # interpolate center & orientation.
                    prev_ann_rec = prev_inst_map[curr_ann_rec['instance_token']]

                    # Interpolate center.
                    center = [
                        np.interp(t, [t0, t1], [c0, c1])
                        for c0, c1 in zip(prev_ann_rec['translation'], curr_ann_rec['translation'])
                    ]

                    # Interpolate orientation.
                    rotation = Quaternion.slerp(q0=Quaternion(prev_ann_rec['rotation']),
                                                q1=Quaternion(curr_ann_rec['rotation']),
                                                amount=(t - t0) / (t1 - t0))

                    box = Box(center, curr_ann_rec['size'],
                              rotation, name=curr_ann_rec['category_name'],
                              token=curr_ann_rec['token'])
                else:
                    # If not, simply grab the current annotation.
                    box = self.get_box(curr_ann_rec['token'])

                boxes.append(box)
        return boxes

    def box_velocity(self, sample_annotation_token: str, max_time_diff: float = 1.5) -> np.ndarray:
        """
        Estimate the velocity for an annotation.
        If possible, we compute the centered difference between the previous and next frame.
        Otherwise we use the difference between the current and previous/next frame.
        If the velocity cannot be estimated, values are set to np.nan.

        Arguments:
            sample_annotation_token: Unique sample_annotation identifier.
            max_time_diff: Max allowed time diff between consecutive samples
                that are used to estimate velocities.

        Returns:
            <np.float: 3>. Velocity in x/y/z direction in m/s.
        """
        current = self.get('sample_annotation', sample_annotation_token)
        has_prev = current['prev'] != ''
        has_next = current['next'] != ''

        # Cannot estimate velocity for a single annotation.
        if not has_prev and not has_next:
            return np.array([np.nan, np.nan, np.nan])

        if has_prev:
            first = self.get('sample_annotation', current['prev'])
        else:
            first = current

        if has_next:
            last = self.get('sample_annotation', current['next'])
        else:
            last = current

        pos_last = np.array(last['translation'])
        pos_first = np.array(first['translation'])
        pos_diff = pos_last - pos_first

        time_last = 1e-6 * self.get('sample', last['sample_token'])['timestamp']
        time_first = 1e-6 * self.get('sample', first['sample_token'])['timestamp']
        time_diff = time_last - time_first

        if has_next and has_prev:
            # If doing centered difference, allow for up to double the max_time_diff.
            max_time_diff *= 2

        if time_diff > max_time_diff:
            # If time_diff is too big, don't return an estimate.
            return np.array([np.nan, np.nan, np.nan])
        else:
            return pos_diff / time_diff

    def list_categories(self) -> None:
        self.explorer.list_categories()

    def list_attributes(self) -> None:
        self.explorer.list_attributes()

    def list_scenes(self) -> None:
        self.explorer.list_scenes()

    def list_sample(self, sample_token: str) -> None:
        self.explorer.list_sample(sample_token)

    def render_pointcloud_in_image(self, sample_token: str, dot_size: int = 5,
                                   pointsensor_channel: str = 'LIDAR_LEFT',
                                   camera_channel: str = 'CAMERA_LEFT_FRONT',
                                   out_path: str = None,
                                   render_intensity: bool = False,
                                   cmap: str = 'viridis',
                                   verbose: bool = True) -> None:
        self.explorer.render_pointcloud_in_image(sample_token, dot_size,
                                                 pointsensor_channel=pointsensor_channel,
                                                 camera_channel=camera_channel,
                                                 out_path=out_path,
                                                 render_intensity=render_intensity,
                                                 cmap=cmap,
                                                 verbose=verbose)

    def render_sample(self, sample_token: str,
                      box_vis_level: BoxVisibility = BoxVisibility.ANY,
                      nsweeps: int = 1,
                      out_path: str = None,
                      verbose: bool = True) -> None:
        self.explorer.render_sample(token=sample_token, box_vis_level=box_vis_level,
                                    nsweeps=nsweeps, out_path=out_path, verbose=verbose)

    def render_sample_data(self, sample_data_token: str,
                           with_anns: bool = True, selected_anntokens: List[str] = None,
                           box_vis_level: BoxVisibility = BoxVisibility.ANY,
                           axes_limit: float = 40, ax = None,
                           nsweeps: int = 1, out_path: str = None,
                           use_flat_vehicle_coordinates: bool = True,
                           point_scale: float = 1.0,
                           verbose: bool = True,
                           cmap: str = 'viridis',
                           cnorm: bool = True) -> None:
        self.explorer.render_sample_data(sample_data_token=sample_data_token,
                                         with_anns=with_anns,
                                         selected_anntokens=selected_anntokens,
                                         box_vis_level=box_vis_level,
                                         axes_limit=axes_limit,
                                         ax=ax,
                                         nsweeps=nsweeps,
                                         out_path=out_path,
                                         use_flat_vehicle_coordinates=use_flat_vehicle_coordinates,
                                         point_scale=point_scale,
                                         verbose=verbose,
                                         cmap=cmap,
                                         cnorm=cnorm)

    def render_annotation(self, sample_annotation_token: str, margin: float = 10,
                          view: np.ndarray = np.eye(4),
                          use_flat_vehicle_coordinates: bool = True,
                          box_vis_level: BoxVisibility = BoxVisibility.ANY,
                          out_path: str = None,
                          extra_info: bool = False) -> None:
        self.explorer.render_annotation(anntoken=sample_annotation_token,
                                        margin=margin, view=view,
                                        use_flat_vehicle_coordinates=use_flat_vehicle_coordinates,
                                        box_vis_level=box_vis_level, out_path=out_path,
                                        extra_info=extra_info)

    def render_instance(self, instance_token: str, margin: float = 10,
                        view: np.ndarray = np.eye(4),
                        box_vis_level: BoxVisibility = BoxVisibility.ANY,
                        out_path: str = None,
                        extra_info: bool = False) -> None:
        self.explorer.render_instance(instance_token=instance_token, margin=margin, view=view,
                                      box_vis_level=box_vis_level, out_path=out_path,
                                      extra_info=extra_info)

    def render_scene(self, scene_token: str, freq: float = 10,
                     imsize: Tuple[float, float] = (640, 360),
                     out_path: str = None) -> None:
        self.explorer.render_scene(scene_token=scene_token, freq=freq, imsize=imsize,
                                   out_path=out_path)

    def render_scene_channel(self, scene_token: str, channel: str = 'CAMERA_LEFT_FRONT',
                             freq: float = 10, imsize: Tuple[float, float] = (640, 360),
                             out_path: str = None) -> None:
        self.explorer.render_scene_channel(scene_token, channel=channel, freq=freq,
                                           imsize=imsize, out_path=out_path)

    def render_pointcloud(self,
                          sample_rec: Dict,
                          chans: Union[str, List[str]],
                          ref_chan: str,
                          with_anns: bool = True,
                          box_vis_level: BoxVisibility = BoxVisibility.ANY,
                          nsweeps: int = 1,
                          min_distance: float = 1.0,
                          cmap: str = 'viridis',
                          out_path: str = None) -> None:
        self.explorer.render_pointcloud(sample_rec=sample_rec,
                                        chans=chans,
                                        ref_chan=ref_chan,
                                        with_anns=with_anns,
                                        box_vis_level=box_vis_level,
                                        nsweeps=nsweeps,
                                        min_distance=min_distance,
                                        cmap=cmap,
                                        out_path=out_path)

    def render_calibrated_sensor(self,
                                 sample_token: str,
                                 out_path: str = None) -> None:
        self.explorer.render_calibrated_sensor(sample_token=sample_token,
                                               out_path=out_path)
