# Copyright 2021 Motional
# Copyright 2024 MAN Truck & Bus SE

from __future__ import annotations

import json
import os
import os.path as osp
import sys
import time
from bisect import bisect_left
from collections import OrderedDict
from datetime import datetime
from typing import Dict, List, Tuple, Union

import cv2
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d

from PIL import Image
from matplotlib import rcParams
from matplotlib.axes import Axes
from matplotlib.colors import Colormap, Normalize
from matplotlib.cm import ScalarMappable
from pyquaternion import Quaternion

from truckscenes.utils.data_classes import LidarPointCloud, RadarPointCloud, Box
from truckscenes.utils.geometry_utils import view_points, box_in_image, transform_matrix, \
    BoxVisibility
from truckscenes.utils import color_map

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
        self.colormap = color_map.get_colormap()

        if verbose:
            for table in self.table_names:
                print("{} {},".format(len(getattr(self, table)), table))
            print("Done loading in {:.3f} seconds.\n======".format(time.time() - start_time))

        # Make reverse indexes for common lookups.
        self.__make_reverse_index__(verbose)

        # Initialize TruckScenesExplorer class.
        self.explorer = TruckScenesExplorer(self)

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
                           axes_limit: float = 40, ax: Axes = None,
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


class TruckScenesExplorer:
    """ Helper class to list and visualize truckscenes data. These are meant to serve
    as tutorials and templates for working with the data.
    """
    def __init__(self, trucksc: TruckScenes):
        self.trucksc = trucksc

    def get_color(self, category_name: str) -> Tuple[int, int, int]:
        """
        Provides the default colors based on the category names.
        This method works for the general truckscenes categories,
        as well as the truckscenes detection categories.
        """
        return self.trucksc.colormap[category_name]

    def list_categories(self) -> None:
        """ Print categories, counts and stats. These stats only cover the split
        specified in trucksc.version.
        """
        print('Category stats for split %s:' % self.trucksc.version)

        # Add all annotations.
        categories = dict()
        for record in self.trucksc.sample_annotation:
            if record['category_name'] not in categories:
                categories[record['category_name']] = []
            categories[record['category_name']].append(
                record['size'] + [record['size'][1] / record['size'][0]]
            )

        # Print stats.
        for name, stats in sorted(categories.items()):
            stats = np.array(stats)
            print(
                f'{name[:27]:27} n={stats.shape[0]:5}, '
                f'width={np.mean(stats[:, 0]):5.2f}\u00B1{np.std(stats[:, 0]):.2f}, '
                f'len={np.mean(stats[:, 1]):5.2f}\u00B1{np.std(stats[:, 1]):.2f}, '
                f'height={np.mean(stats[:, 2]):5.2f}\u00B1{np.std(stats[:, 2]):.2f}, '
                f'lw_aspect={np.mean(stats[:, 3]):5.2f}\u00B1{np.std(stats[:, 3]):.2f}'
            )

    def list_attributes(self) -> None:
        """ Prints attributes and counts. """
        attribute_counts = dict()
        for record in self.trucksc.sample_annotation:
            for attribute_token in record['attribute_tokens']:
                att_name = self.trucksc.get('attribute', attribute_token)['name']
                if att_name not in attribute_counts:
                    attribute_counts[att_name] = 0
                attribute_counts[att_name] += 1

        for name, count in sorted(attribute_counts.items()):
            print('{}: {}'.format(name, count))

    def list_scenes(self) -> None:
        """ Lists all scenes with some meta data. """

        def ann_count(record):
            count = 0
            sample = self.trucksc.get('sample', record['first_sample_token'])
            while not sample['next'] == "":
                count += len(sample['anns'])
                sample = self.trucksc.get('sample', sample['next'])
            return count

        recs = [
            (self.trucksc.get('sample', record['first_sample_token'])['timestamp'], record)
            for record in self.trucksc.scene
        ]

        for start_time, record in sorted(recs):
            first_sample_timestamp = \
                self.trucksc.get('sample', record['first_sample_token'])['timestamp']
            last_sample_timestamp = \
                self.trucksc.get('sample', record['last_sample_token'])['timestamp']

            start_time = first_sample_timestamp / 1000000
            length_time = last_sample_timestamp / 1000000 - start_time
            desc = record['description']
            if len(desc) > 55:
                desc = desc[:51] + "..."

            print('[{}] {}: length: {:.2f}s, #anns: {}, desc: {:16}'.format(
                datetime.utcfromtimestamp(start_time).strftime('%y-%m-%d %H:%M:%S'),
                record['name'], length_time, ann_count(record), desc))

    def list_sample(self, sample_token: str) -> None:
        """ Prints sample_data tokens and sample_annotation tokens related to the sample_token. """

        sample_record = self.trucksc.get('sample', sample_token)
        print('Sample: {}\n'.format(sample_record['token']))

        for sd_token in sample_record['data'].values():
            sd_record = self.trucksc.get('sample_data', sd_token)
            print(f"sample_data_token: {sd_token}"
                  f", mod: {sd_record['sensor_modality']}"
                  f", channel: {sd_record['channel']}")

        print('')
        for ann_token in sample_record['anns']:
            ann_record = self.trucksc.get('sample_annotation', ann_token)
            print(f"sample_annotation_token: {ann_record['token']}"
                  f", category: {ann_record['category_name']}")

    def map_pointcloud_to_image(self,
                                pointsensor_token: str,
                                camera_token: str,
                                min_dist: float = 1.0,
                                render_intensity: bool = False,
                                cmap: str = 'viridis',
                                cnorm: bool = True) -> Tuple:
        """
        Given a point sensor (lidar/radar) token and camera sample_data token,
        load pointcloud and map it to the image plane.

        Arguments:
            pointsensor_token: Lidar/radar sample_data token.
            camera_token: Camera sample_data token.
            min_dist: Distance from the camera below which points are discarded.
            render_intensity: Whether to render lidar intensity instead of point depth.

        Returns:
            (pointcloud <np.float: 2, n)>, coloring <np.float: n>, image <Image>).
        """
        if not isinstance(cmap, Colormap):
            cmap = color_map.colormaps[cmap]

        cam = self.trucksc.get('sample_data', camera_token)
        pointsensor = self.trucksc.get('sample_data', pointsensor_token)
        pcl_path = osp.join(self.trucksc.dataroot, pointsensor['filename'])
        if pointsensor['sensor_modality'] == 'lidar':
            pc = LidarPointCloud.from_file(pcl_path)
        else:
            pc = RadarPointCloud.from_file(pcl_path)
        im = Image.open(osp.join(self.trucksc.dataroot, cam['filename']))

        # Points live in the point sensor frame. So they need to be transformed
        # via global to the image plane.
        # First step: transform the pointcloud to the ego vehicle frame
        # for the timestamp of the sweep.
        cs_record = self.trucksc.get('calibrated_sensor', pointsensor['calibrated_sensor_token'])
        pc.rotate(Quaternion(cs_record['rotation']).rotation_matrix)
        pc.translate(np.array(cs_record['translation']))

        # Second step: transform from ego to the global frame.
        poserecord = self.trucksc.get('ego_pose', pointsensor['ego_pose_token'])
        pc.rotate(Quaternion(poserecord['rotation']).rotation_matrix)
        pc.translate(np.array(poserecord['translation']))

        # Third step: transform from global into the ego vehicle frame
        # for the timestamp of the image.
        poserecord = self.trucksc.get('ego_pose', cam['ego_pose_token'])
        pc.translate(-np.array(poserecord['translation']))
        pc.rotate(Quaternion(poserecord['rotation']).rotation_matrix.T)

        # Fourth step: transform from ego into the camera.
        cs_record = self.trucksc.get('calibrated_sensor', cam['calibrated_sensor_token'])
        pc.translate(-np.array(cs_record['translation']))
        pc.rotate(Quaternion(cs_record['rotation']).rotation_matrix.T)

        # Fifth step: actually take a "picture" of the point cloud.
        # Grab the depths (camera frame z axis points away from the camera).
        depths = pc.points[2, :]

        if render_intensity:
            if pointsensor['sensor_modality'] == 'lidar':
                # Retrieve the color from the intensities.
                coloring = pc.points[3, :]
            else:
                # Retrieve the color from the rcs.
                coloring = pc.points[6, :]
        else:
            # Retrieve the color from the depth.
            coloring = depths

        # Color mapping
        if cnorm:
            norm = Normalize(vmin=np.quantile(coloring, 0.5),
                             vmax=np.quantile(coloring, 0.95), clip=True)
        else:
            norm = None
        mapper = ScalarMappable(norm=norm, cmap=cmap)
        coloring = mapper.to_rgba(coloring)[..., :3]

        # Take the actual picture (matrix multiplication with camera-matrix + renormalization).
        points = view_points(pc.points[:3, :], np.array(cs_record['camera_intrinsic']),
                             normalize=True)

        # Remove points that are either outside or behind the camera.
        # Leave a margin of 1 pixel for aesthetic reasons.
        # Also make sure points are at least 1m in front of the camera to
        # avoid seeing the lidar points on the camera
        # casing for non-keyframes which are slightly out of sync.
        mask = np.ones(depths.shape[0], dtype=bool)
        mask = np.logical_and(mask, depths > min_dist)
        mask = np.logical_and(mask, points[0, :] > 1)
        mask = np.logical_and(mask, points[0, :] < im.size[0] - 1)
        mask = np.logical_and(mask, points[1, :] > 1)
        mask = np.logical_and(mask, points[1, :] < im.size[1] - 1)
        points = points[:, mask]
        coloring = coloring[mask, :]

        return points, coloring, im

    def render_pointcloud_in_image(self,
                                   sample_token: str,
                                   dot_size: int = 4,
                                   pointsensor_channel: str = 'LIDAR_LEFT',
                                   camera_channel: str = 'CAMERA_LEFT_FRONT',
                                   out_path: str = None,
                                   render_intensity: bool = False,
                                   ax: Axes = None,
                                   cmap: str = 'viridis',
                                   verbose: bool = True):
        """
        Scatter-plots a pointcloud on top of image.

        Arguments:
            sample_token: Sample token.
            dot_size: Scatter plot dot size.
            pointsensor_channel: RADAR or LIDAR channel name, e.g. 'LIDAR_LEFT'.
            camera_channel: Camera channel name, e.g. 'CAM_FRONT'.
            out_path: Optional path to save the rendered figure to disk.
            render_intensity: Whether to render lidar intensity instead of point depth.
            ax: Axes onto which to render.
            verbose: Whether to display the image in a window.
        """
        if not isinstance(cmap, Colormap):
            cmap = color_map.colormaps[cmap]

        sample_record = self.trucksc.get('sample', sample_token)

        # Here we just grab the front camera and the point sensor.
        pointsensor_token = sample_record['data'][pointsensor_channel]
        camera_token = sample_record['data'][camera_channel]

        points, coloring, im = self.map_pointcloud_to_image(pointsensor_token, camera_token,
                                                            render_intensity=render_intensity,
                                                            cmap=cmap)

        # Init axes.
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(9, 16))
            fig.canvas.set_window_title(sample_token)
        else:  # Set title on if rendering as part of render_sample.
            ax.set_title(camera_channel)
        ax.imshow(im)
        ax.scatter(points[0, :], points[1, :], marker='o', c=coloring,
                   s=dot_size, edgecolors='none')
        ax.axis('off')
        plt.tight_layout()

        if out_path is not None:
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            plt.savefig(out_path, bbox_inches='tight', pad_inches=0, transparent=True, dpi=800)

    def render_sample(self,
                      token: str,
                      box_vis_level: BoxVisibility = BoxVisibility.ANY,
                      nsweeps: int = 1,
                      out_path: str = None,
                      verbose: bool = True) -> None:
        """
        Render all LIDAR and camera sample_data in sample along with annotations.

        Arguments:
            token: Sample token.
            box_vis_level: If sample_data is an image, this sets required visibility for boxes.
            nsweeps: Number of sweeps for lidar and radar.
            out_path: Optional path to save the rendered figure to disk.
            verbose: Whether to show the rendered sample in a window or not.
        """
        record = self.trucksc.get('sample', token)

        # Separate radar from lidar and camera.
        radar_data = {}
        camera_data = {}
        lidar_data = {}
        for channel, token in record['data'].items():
            sd_record = self.trucksc.get('sample_data', token)
            sensor_modality = sd_record['sensor_modality']

            if sensor_modality == 'camera':
                camera_data[channel] = token
            elif sensor_modality == 'lidar':
                lidar_data[channel] = token
            else:
                radar_data[channel] = token

        # Create plots.
        num_radar_plots = 1 if len(radar_data) > 0 else 0
        num_lidar_plots = 1 if len(lidar_data) > 0 else 0
        n = num_radar_plots + len(camera_data) + num_lidar_plots
        cols = 2
        ax_count = 0
        fig, axes = plt.subplots(int(np.ceil(n / cols)), cols, figsize=(16, 14))

        # Plot radars into a single subplot.
        if len(radar_data) > 0:
            for i, (_, sd_token) in enumerate(radar_data.items()):
                ax: Axes = axes.flatten()[ax_count]
                self.render_sample_data(sd_token, with_anns=i == 0, box_vis_level=box_vis_level,
                                        axes_limit=(84, 40), ax=ax, nsweeps=nsweeps, verbose=False)
            ax_count += 1
            ax.set_title('Fused RADARs')

        # Plot lidar into a single subplot.
        if len(lidar_data) > 0:
            for i, (_, sd_token) in enumerate(lidar_data.items()):
                ax: Axes = axes.flatten()[ax_count]
                self.render_sample_data(sd_token, with_anns=i == 0, box_vis_level=box_vis_level,
                                        axes_limit=(84, 40), ax=ax, nsweeps=nsweeps, verbose=False)
            ax_count += 1
            ax.set_title('Fused LIDARs')

        # Plot cameras in separate subplots.
        if len(camera_data) > 0:
            for i, (_, sd_token) in enumerate(camera_data.items()):
                ax: Axes = axes.flatten()[ax_count]
                self.render_sample_data(sd_token, box_vis_level=box_vis_level,
                                        ax=ax, nsweeps=nsweeps, verbose=False)
                ax_count += 1

        # Change plot settings and write to disk.
        axes.flatten()[-1].axis('off')
        scene = self.trucksc.get('scene', record["scene_token"])
        plt.suptitle(scene["name"])
        plt.tight_layout()
        fig.subplots_adjust()


        if out_path is not None:
            plt.savefig(out_path)

    def render_sample_data(self,
                           sample_data_token: str,
                           with_anns: bool = True,
                           selected_anntokens: List[str] = None,
                           box_vis_level: BoxVisibility = BoxVisibility.ANY,
                           axes_limit: Union[List[float], Tuple[float], float] = 40,
                           ax: Axes = None,
                           nsweeps: int = 1,
                           out_path: str = None,
                           use_flat_vehicle_coordinates: bool = True,
                           point_scale: float = 1.0,
                           cmap: str = 'viridis',
                           cnorm: bool = True,
                           verbose: bool = True) -> None:
        """
        Render sample data onto axis.

        Arguments:
            sample_data_token: Sample_data token.
            with_anns: Whether to draw box annotations.
            box_vis_level: If sample_data is an image, this sets required visibility for boxes.
            axes_limit: Axes limit for lidar and radar (measured in meters).
            ax: Axes onto which to render.
            nsweeps: Number of sweeps for lidar and radar.
            out_path: Optional path to save the rendered figure to disk.
            use_flat_vehicle_coordinates: Instead of the current sensor's coordinate frame,
                use ego frame which is aligned to z-plane in the world.
                Note: Previously this method did not use flat vehicle coordinates, which can lead
                to small errors when the vertical axis of the global frame and lidar are not
                aligned. The new setting is more correct and rotates the plot by ~90 degrees.
            verbose: Whether to display the image after it is rendered.
        """
        if not isinstance(sample_data_token, list):
            sample_data_token = [sample_data_token]

        if not isinstance(cmap, Colormap):
            cmap = color_map.colormaps[cmap]

        if not isinstance(axes_limit, (list, tuple)):
            axes_limit = [axes_limit, axes_limit]

        # Determine sensor modality
        sensor_modality = self.trucksc.get('sample_data', sample_data_token[0])['sensor_modality']

        # Render Point Cloud data
        if sensor_modality in ['lidar', 'radar']:
            points = []
            intensities = []

            for sd_token in sample_data_token:
                sd_record = self.trucksc.get('sample_data', sd_token)

                sample_rec = self.trucksc.get('sample', sd_record['sample_token'])
                chan = sd_record['channel']
                ref_chan = 'LIDAR_LEFT'
                ref_sd_token = sample_rec['data'][ref_chan]
                ref_sd_record = self.trucksc.get('sample_data', ref_sd_token)

                if sensor_modality == 'lidar':
                    # Get aggregated lidar point cloud in lidar frame.
                    pc, _ = LidarPointCloud.from_file_multisweep(self.trucksc, sample_rec,
                                                                 chan, ref_chan,
                                                                 nsweeps=nsweeps)
                    velocities = None
                    intensity = pc.points[3, :]
                else:
                    # Get aggregated radar point cloud in reference frame.
                    # The point cloud is transformed to the reference frame
                    # for visualization purposes.
                    pc, _ = RadarPointCloud.from_file_multisweep(self.trucksc, sample_rec,
                                                                 chan, ref_chan,
                                                                 nsweeps=nsweeps)

                    # Transform radar velocities (x is front, y is left),
                    # as these are not transformed when loading the
                    # point cloud.
                    radar_cs_record = self.trucksc.get('calibrated_sensor',
                                                     sd_record['calibrated_sensor_token'])
                    ref_cs_record = self.trucksc.get('calibrated_sensor',
                                                   ref_sd_record['calibrated_sensor_token'])
                    velocities = pc.points[3:5, :]
                    velocities = np.vstack((velocities, np.zeros(pc.points.shape[1])))
                    velocities = np.dot(Quaternion(radar_cs_record['rotation']).rotation_matrix,
                                        velocities)
                    velocities = np.dot(Quaternion(ref_cs_record['rotation']).rotation_matrix.T,
                                        velocities)
                    velocities[2, :] = np.zeros(pc.points.shape[1])
                    intensity = pc.points[6, :]

                # By default we render the sample_data top down in the sensor frame.
                # This is slightly inaccurate when rendering the map as the sensor frame may
                # not be perfectly upright.
                if use_flat_vehicle_coordinates:
                    # Retrieve transformation matrices for reference point cloud.
                    cs_record = self.trucksc.get('calibrated_sensor',
                                               ref_sd_record['calibrated_sensor_token'])
                    pose_record = self.trucksc.get('ego_pose',
                                                 ref_sd_record['ego_pose_token'])
                    ref_to_ego = transform_matrix(translation=cs_record['translation'],
                                                  rotation=Quaternion(cs_record["rotation"]))

                    # Compute rotation between 3D vehicle pose and "flat" vehicle pose
                    # (parallel to global z plane).
                    ego_yaw = Quaternion(pose_record['rotation']).yaw_pitch_roll[0]
                    rotation_vehicle_flat_from_vehicle = np.dot(
                        Quaternion(scalar=np.cos(ego_yaw / 2),
                                   vector=[0, 0, np.sin(ego_yaw / 2)]).rotation_matrix,
                        Quaternion(pose_record['rotation']).inverse.rotation_matrix)
                    vehicle_flat_from_vehicle = np.eye(4)
                    vehicle_flat_from_vehicle[:3, :3] = rotation_vehicle_flat_from_vehicle
                    viewpoint = np.dot(vehicle_flat_from_vehicle, ref_to_ego)

                    # Rotate upwards
                    vehicle_flat_up_from_vehicle_flat = np.eye(4)
                    rotation_axis = Quaternion(matrix=viewpoint[:3, :3])
                    vehicle_flat_up_from_vehicle_flat[:3, :3] = \
                        Quaternion(axis=rotation_axis.rotate([0, 0, 1]),
                                   angle=np.pi/2).rotation_matrix
                    viewpoint = np.dot(vehicle_flat_up_from_vehicle_flat, viewpoint)
                else:
                    viewpoint = np.eye(4)

                # Init axes.
                if ax is None:
                    _, ax = plt.subplots(1, 1, figsize=(9, 9))

                # Show point cloud.
                points.append(view_points(pc.points[:3, :], viewpoint, normalize=False))
                intensities.append(intensity)

            points = np.concatenate(points, axis=1)
            intensities = np.concatenate(intensities, axis=0)

            # Colormapping
            if cnorm:
                norm = Normalize(vmin=np.min(intensities), vmax=np.max(intensities), clip=True)
            else:
                norm = None
            mapper = ScalarMappable(norm=norm, cmap=cmap)
            colors = mapper.to_rgba(intensities)[..., :3]

            point_scale = point_scale * 0.4 if sensor_modality == 'lidar' else point_scale * 3.0
            ax.scatter(points[0, :], points[1, :], marker='o',
                       c=colors, s=point_scale, edgecolors='none')

            # Show ego vehicle.
            ax.plot(0, 0, 'x', color='red')

            # Get boxes in lidar frame.box_vis_level
            _, boxes, _ = self.trucksc.get_sample_data(
                ref_sd_token, box_vis_level=box_vis_level, selected_anntokens=selected_anntokens,
                use_flat_vehicle_coordinates=use_flat_vehicle_coordinates
            )

            # Show boxes.
            if with_anns:
                for box in boxes:
                    c = np.array(self.get_color(box.name)) / 255.0
                    box.render(ax, view=np.eye(4), colors=(c, c, c), linewidth=2.0)

            # Limit visible range.
            ax.set_xlim(-axes_limit[0], axes_limit[0])
            ax.set_ylim(-axes_limit[1], axes_limit[1])

        # Render Camera data
        elif sensor_modality == 'camera':
            sd_record = self.trucksc.get('sample_data', sample_data_token[0])

            # Load boxes and image.
            data_path, boxes, camera_intrinsic = \
                self.trucksc.get_sample_data(
                    sample_data_token[0], box_vis_level=box_vis_level,
                    selected_anntokens=selected_anntokens
                )
            data = Image.open(data_path)

            # Init axes.
            if ax is None:
                _, ax = plt.subplots(1, 1, figsize=(9, 16))

            # Show image.
            ax.imshow(data)

            # Show boxes.
            if with_anns:
                for box in boxes:
                    c = np.array(self.get_color(box.name)) / 255.0
                    box.render(ax, view=camera_intrinsic, normalize=True,
                               colors=(c, c, c), linewidth=1.0)

            # Limit visible range.
            ax.set_xlim(0, data.size[0])
            ax.set_ylim(data.size[1], 0)

        else:
            raise ValueError("Error: Unknown sensor modality!")

        ax.axis('off')
        ax.set_title('{} {labels_type}'.format(sensor_modality.upper(), labels_type=''))
        ax.set_aspect('equal')

        if out_path is not None:
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            plt.savefig(out_path, bbox_inches='tight', pad_inches=0, transparent=True, dpi=800)

    def render_annotation(self,
                          anntoken: str,
                          margin: float = 10,
                          view: np.ndarray = np.eye(4),
                          use_flat_vehicle_coordinates: bool = True,
                          box_vis_level: BoxVisibility = BoxVisibility.ANY,
                          out_path: str = None,
                          extra_info: bool = False) -> None:
        """
        Render selected annotation.

        Arguments:
            anntoken: Sample_annotation token.
            margin: How many meters in each direction to include in LIDAR view.
            view: LIDAR view point.
            box_vis_level: If sample_data is an image, this sets required visibility for boxes.
            out_path: Optional path to save the rendered figure to disk.
            extra_info: Whether to render extra information below camera view.
        """
        ann_record = self.trucksc.get('sample_annotation', anntoken)
        sample_record = self.trucksc.get('sample', ann_record['sample_token'])

        _, axes = plt.subplots(1, 2, figsize=(18, 9))

        # Figure out which camera the object is fully visible in (this may return nothing).
        boxes, cam = [], []
        cams = [key for key in sample_record['data'].keys() if 'CAMERA' in key]
        for cam in cams:
            _, boxes, _ = self.trucksc.get_sample_data(sample_record['data'][cam],
                                                     box_vis_level=box_vis_level,
                                                     selected_anntokens=[anntoken])
            if len(boxes) > 0:
                break  # We found an image that matches. Let's abort.
        assert len(boxes) > 0, 'Error: Could not find image where annotation is visible. ' \
                               'Try using e.g. BoxVisibility.ANY.'
        assert len(boxes) < 2, 'Error: Found multiple annotations. Something is wrong!'

        cam = sample_record['data'][cam]

        # Plot LIDAR view.
        lidar = sample_record['data']['LIDAR_LEFT']
        data_path, boxes, _ = self.trucksc.get_sample_data(
            lidar, selected_anntokens=[anntoken],
            use_flat_vehicle_coordinates=use_flat_vehicle_coordinates
        )

        self.render_sample_data([v for k, v in sample_record['data'].items() if 'LIDAR' in k],
                                with_anns=True, selected_anntokens=[anntoken],
                                box_vis_level=box_vis_level, ax=axes[0],
                                use_flat_vehicle_coordinates=use_flat_vehicle_coordinates,
                                point_scale=4)

        # Render annotations
        for box in boxes:
            corners = view_points(boxes[0].corners(), view, False)[:2, :]
            axes[0].set_xlim([np.min(corners[0, :]) - margin, np.max(corners[0, :]) + margin])
            axes[0].set_ylim([np.min(corners[1, :]) - margin, np.max(corners[1, :]) + margin])
            axes[0].axis('off')
            axes[0].set_aspect('equal')

        # Plot CAMERA view.
        data_path, boxes, camera_intrinsic = self.trucksc.get_sample_data(
            cam, selected_anntokens=[anntoken]
        )
        im = Image.open(data_path)
        axes[1].imshow(im)
        axes[1].set_title(self.trucksc.get('sample_data', cam)['channel'])
        axes[1].axis('off')
        axes[1].set_aspect('equal')
        for box in boxes:
            c = np.array(self.get_color(box.name)) / 255.0
            box.render(axes[1], view=camera_intrinsic, normalize=True, colors=(c, c, c))

        # Print extra information about the annotation below the camera view.
        if extra_info:
            rcParams['font.family'] = 'monospace'

            w, l, h = ann_record['size']
            category = ann_record['category_name']
            lidar_points = ann_record['num_lidar_pts']
            radar_points = ann_record['num_radar_pts']

            sample_data_record = self.trucksc.get('sample_data', sample_record['data']['LIDAR_LEFT'])
            pose_record = self.trucksc.get('ego_pose', sample_data_record['ego_pose_token'])
            dist = np.linalg.norm(
                np.array(pose_record['translation']) - np.array(ann_record['translation'])
            )

            information = ' \n'.join(['category: {}'.format(category),
                                      '',
                                      '# lidar points: {0:>4}'.format(lidar_points),
                                      '# radar points: {0:>4}'.format(radar_points),
                                      '',
                                      'distance: {:>7.3f}m'.format(dist),
                                      '',
                                      'width:  {:>7.3f}m'.format(w),
                                      'length: {:>7.3f}m'.format(l),
                                      'height: {:>7.3f}m'.format(h)])

            plt.annotate(information, (0, 0), (0, -20), xycoords='axes fraction',
                         textcoords='offset points', va='top')

        if out_path is not None:
            plt.savefig(out_path)

    def render_instance(self,
                        instance_token: str,
                        margin: float = 10,
                        view: np.ndarray = np.eye(4),
                        box_vis_level: BoxVisibility = BoxVisibility.ANY,
                        out_path: str = None,
                        extra_info: bool = False) -> None:
        """
        Finds the annotation of the given instance that is closest to the vehicle,
        and then renders it.

        Arguments:
            instance_token: The instance token.
            margin: How many meters in each direction to include in LIDAR view.
            view: LIDAR view point.
            box_vis_level: If sample_data is an image, this sets required visibility for boxes.
            out_path: Optional path to save the rendered figure to disk.
            extra_info: Whether to render extra information below camera view.
        """
        ann_tokens = self.trucksc.field2token('sample_annotation', 'instance_token', instance_token)
        closest = [np.inf, None]
        for ann_token in ann_tokens:
            ann_record = self.trucksc.get('sample_annotation', ann_token)
            sample_record = self.trucksc.get('sample', ann_record['sample_token'])
            sample_data_record = self.trucksc.get('sample_data', sample_record['data']['LIDAR_LEFT'])
            pose_record = self.trucksc.get('ego_pose', sample_data_record['ego_pose_token'])
            dist = np.linalg.norm(
                np.array(pose_record['translation']) - np.array(ann_record['translation'])
            )
            if dist < closest[0]:
                closest[0] = dist
                closest[1] = ann_token

        self.render_annotation(anntoken=closest[1], margin=margin, view=view,
                               box_vis_level=box_vis_level, out_path=out_path,
                               extra_info=extra_info)

    def render_scene(self,
                     scene_token: str,
                     freq: float = 10,
                     imsize: Tuple[float, float] = (640, 360),
                     out_path: str = None) -> None:
        """
        Renders a full scene with all camera channels.

        Arguments:
            scene_token: Unique identifier of scene to render.
            freq: Display frequency (Hz).
            imsize: Size of image to render. The larger the slower this will run.
            out_path: Optional path to write a video file of the rendered frames.
        """

        assert imsize[0] / imsize[1] == 16 / 9, "Aspect ratio should be 16/9."

        if out_path is not None:
            assert osp.splitext(out_path)[-1] == '.avi'

        # Get records from DB.
        scene_rec = self.trucksc.get('scene', scene_token)
        first_sample_rec = self.trucksc.get('sample', scene_rec['first_sample_token'])
        last_sample_rec = self.trucksc.get('sample', scene_rec['last_sample_token'])

        # Set some display parameters.
        layout = {
            'CAMERA_LEFT_FRONT': (0, 0),
            'CAMERA_RIGHT_FRONT': (imsize[0], 0),
            'CAMERA_LEFT_BACK': (0, imsize[1]),
            'CAMERA_RIGHT_BACK': (imsize[0], imsize[1]),
        }

        # Flip these for aesthetic reasons.
        # horizontal_flip = ['CAMERA_LEFT_BACK','CAMERA_RIGHT_BACK']
        horizontal_flip = []

        time_step = 1 / freq * 1e6  # Time-stamps are measured in micro-seconds.

        window_name = '{}'.format(scene_rec['name'])
        cv2.namedWindow(window_name)
        cv2.moveWindow(window_name, 0, 0)

        canvas = np.ones((2 * imsize[1], 2 * imsize[0], 3), np.uint8)
        if out_path is not None:
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            out = cv2.VideoWriter(out_path, fourcc, freq, canvas.shape[1::-1])
        else:
            out = None

        # Load first sample_data record for each channel.
        current_recs = {}  # Holds the current record to be displayed by channel.
        prev_recs = {}  # Hold the previous displayed record by channel.
        for channel in layout:
            current_recs[channel] = self.trucksc.get('sample_data',
                                                   first_sample_rec['data'][channel])
            prev_recs[channel] = None

        current_time = first_sample_rec['timestamp']

        while current_time < last_sample_rec['timestamp']:

            current_time += time_step

            # For each channel, find first sample that has time > current_time.
            for channel, sd_rec in current_recs.items():
                while sd_rec['timestamp'] < current_time and sd_rec['next'] != '':
                    sd_rec = self.trucksc.get('sample_data', sd_rec['next'])
                    current_recs[channel] = sd_rec

            # Now add to canvas
            for channel, sd_rec in current_recs.items():

                # Only update canvas if we have not already rendered this one.
                if not sd_rec == prev_recs[channel]:

                    # Get annotations and params from DB.
                    impath, boxes, camera_intrinsic = self.trucksc.get_sample_data(
                        sd_rec['token'], box_vis_level=BoxVisibility.ANY
                    )

                    # Load and render.
                    if not osp.exists(impath):
                        raise Exception('Error: Missing image %s' % impath)
                    im = cv2.imread(impath)
                    for box in boxes:
                        c = self.get_color(box.name)
                        box.render_cv2(im, view=camera_intrinsic,
                                       normalize=True, colors=(c, c, c))

                    im = cv2.resize(im, imsize)
                    if channel in horizontal_flip:
                        im = im[:, ::-1, :]

                    canvas[
                        layout[channel][1]: layout[channel][1] + imsize[1],
                        layout[channel][0]:layout[channel][0] + imsize[0], :
                    ] = im

                    # Store here so we don't render the same image twice.
                    prev_recs[channel] = sd_rec

            # Show updated canvas.
            cv2.imshow(window_name, canvas)
            if out_path is not None:
                out.write(canvas)

            key = cv2.waitKey(1)  # Wait a very short time (1 ms).

            if key == 32:  # if space is pressed, pause.
                key = cv2.waitKey()

            if key == 27:  # if ESC is pressed, exit.
                cv2.destroyAllWindows()
                break

        cv2.destroyAllWindows()
        if out_path is not None:
            out.release()

    def render_pointcloud(self,
                          sample_rec: Dict,
                          chans: Union[str, List[str]],
                          ref_chan: str,
                          with_anns: bool = True,
                          box_vis_level: BoxVisibility = BoxVisibility.ANY,
                          nsweeps: int = 5,
                          min_distance: float = 1.0,
                          cmap: str = 'viridis',
                          out_path: str = None) -> None:
        """Renders a 3D representation of all given point clouds.

        Arguments:
            sample_rec: Sample record.
            chans: Sensor channels to render.
            ref_chan: Reference sensor channel.
            with_anns: Whether to draw box annotations.
            box_vis_level: If sample_data is an image, this sets required visibility for boxes.
            nsweeps: Number of sweeps for lidar and radar.
            min_distance: Minimum distance to include points.
            cmap: Colormap or colormap name.
            out_path: Optional path to write a image file of the rendered point clouds.
        """
        # Convert chans to list
        if not isinstance(chans, list):
            chans = [chans]

        # Initialize point clouds and intensities
        point_clouds = []
        intensities = []

        for chan in chans:
            # Get sensor modality
            sd_record = self.trucksc.get('sample_data', sample_rec['data'][chan])
            sensor_modality = sd_record['sensor_modality']

            # Load point cloud
            if sensor_modality in {'lidar'}:
                point_obj, _ = LidarPointCloud.from_file_multisweep(self.trucksc, sample_rec,
                                                                    chan, ref_chan, nsweeps,
                                                                    min_distance)
                pc = point_obj.points.T
                intens = pc[:, 3]

            elif sensor_modality in {'radar'}:
                point_obj, _ = RadarPointCloud.from_file_multisweep(self.trucksc, sample_rec,
                                                                    chan, ref_chan, nsweeps,
                                                                    min_distance)
                pc = point_obj.points.T
                intens = pc[:, 6]

            # Add channel data to channels collection
            point_clouds.append(pc[:, :3])
            intensities.append(intens)

        # Concatenate all channels
        point_clouds = np.concatenate(point_clouds, axis=0)
        intensities = np.concatenate(intensities, axis=0)

        # Convert intensities to colors
        rgb = color_map.colormaps[cmap](intensities)[..., :3]

        # Initialize vizualization objets
        vis_obj = []

        # Define point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(point_clouds)
        pcd.colors = o3d.utility.Vector3dVector(rgb)
        vis_obj.append(pcd)

        if with_anns:
            # Get boxes in reference sensor frame
            ref_sd_token = sample_rec['data'][ref_chan]
            _, boxes, _ = self.trucksc.get_sample_data(ref_sd_token, box_vis_level=box_vis_level)

            # Define bounding boxes
            for box in boxes:
                bbox = o3d.geometry.OrientedBoundingBox()
                bbox.center = box.center
                bbox.extent = box.wlh[[1, 0, 2]]
                bbox.R = Quaternion(box.orientation).rotation_matrix
                bbox.color = np.asarray(color_map.get_colormap()[box.name]) / 255
                vis_obj.append(bbox)

        # Visualize point cloud
        rend = o3d.visualization.RenderOption()
        rend.line_width = 8.0
        vis = o3d.visualization.Visualizer()
        vis.update_renderer()
        vis.create_window()
        for obj in vis_obj:
            vis.add_geometry(obj)
            vis.poll_events()
            vis.update_geometry(obj)

        # Save visualization
        if out_path is not None:
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            vis.capture_screen_image(filename=out_path)
            vis.destroy_window()
        else:
            vis.run()

    def render_scene_channel(self,
                             scene_token: str,
                             channel: str = 'CAMERA_LEFT_FRONT',
                             freq: float = 10,
                             imsize: Tuple[float, float] = (640, 360),
                             out_path: str = None) -> None:
        """
        Renders a full scene for a particular camera channel.

        Arguments:
            scene_token: Unique identifier of scene to render.
            channel: Channel to render.
            freq: Display frequency (Hz).
            imsize: Size of image to render. The larger the slower this will run.
            out_path: Optional path to write a video file of the rendered frames.
        """
        valid_channels = ['CAMERA_LEFT_FRONT', 'CAMERA_RIGHT_FRONT',
                          'CAMERA_LEFT_BACK',  'CAMERA_RIGHT_BACK']

        assert imsize[0] / imsize[1] == 16 / 9, "Error: Aspect ratio should be 16/9."
        assert channel in valid_channels, 'Error: Input channel {} not valid.'.format(channel)

        if out_path is not None:
            assert osp.splitext(out_path)[-1] == '.avi'

        # Get records from DB.
        scene_rec = self.trucksc.get('scene', scene_token)
        sample_rec = self.trucksc.get('sample', scene_rec['first_sample_token'])
        sd_rec = self.trucksc.get('sample_data', sample_rec['data'][channel])

        # Open CV init.
        name = '{}: {} (Space to pause, ESC to exit)'.format(scene_rec['name'], channel)
        cv2.namedWindow(name)
        cv2.moveWindow(name, 0, 0)

        if out_path is not None:
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            out = cv2.VideoWriter(out_path, fourcc, freq, imsize)
        else:
            out = None

        has_more_frames = True
        while has_more_frames:

            # Get data from DB.
            impath, boxes, camera_intrinsic = self.trucksc.get_sample_data(
                sd_rec['token'], box_vis_level=BoxVisibility.ANY
            )

            # Load and render.
            if not osp.exists(impath):
                raise Exception('Error: Missing image %s' % impath)
            im = cv2.imread(impath)
            for box in boxes:
                c = self.get_color(box.name)
                box.render_cv2(im, view=camera_intrinsic, normalize=True, colors=(c, c, c))

            # Render.
            im = cv2.resize(im, imsize)
            cv2.imshow(name, im)
            if out_path is not None:
                out.write(im)

            key = cv2.waitKey(10)  # Images stored at approx 10 Hz, so wait 10 ms.
            if key == 32:  # If space is pressed, pause.
                key = cv2.waitKey()

            if key == 27:  # If ESC is pressed, exit.
                cv2.destroyAllWindows()
                break

            if not sd_rec['next'] == "":
                sd_rec = self.trucksc.get('sample_data', sd_rec['next'])
            else:
                has_more_frames = False

        cv2.destroyAllWindows()
        if out_path is not None:
            out.release()

    def render_calibrated_sensor(self,
                                 sample_token: str,
                                 out_path: str = None) -> None:
        """Renderes the sensor coordinate frames in 3D space.

        Arguments:
            sample_token: Sample token.
            out_path: Optional path to write a image file of the rendered coordinate frames.
        """
        # Get sample record
        sa_record = self.trucksc.get('sample', sample_token)

        # Initialize visualization objects
        vis_obj = []

        # Create coordinate frame visualization for every sensor
        for _, sd_token in sa_record['data'].items():
            # Get calibrated sensor record
            sd_record = self.trucksc.get('sample_data', sd_token)
            cs_record = self.trucksc.get('calibrated_sensor', sd_record['calibrated_sensor_token'])

            # Create coordinate frame
            coord = o3d.geometry.TriangleMesh.create_coordinate_frame()
            coord.rotate(Quaternion(cs_record['rotation']).rotation_matrix, center=(0, 0, 0))
            coord.translate(cs_record['translation'])

            vis_obj.append(coord)

        # Visualize coordinate frames
        rend = o3d.visualization.RenderOption()
        rend.line_width = 8.0
        vis = o3d.visualization.Visualizer()
        vis.update_renderer()
        vis.create_window()
        for obj in vis_obj:
            vis.add_geometry(obj)
            vis.poll_events()
            vis.update_geometry(obj)

        # Save visualization
        if out_path is not None:
            vis.capture_screen_image(path=out_path, do_render=False)
            vis.destroy_window()
        else:
            vis.run()
