# Copyright 2021 Motional
# Copyright 2024 MAN Truck & Bus SE


from __future__ import annotations

import copy
import os.path as osp
import warnings

from abc import ABC, abstractmethod
from functools import reduce
from importlib import import_module
from typing import Tuple, List, Dict

import numpy as np
import pypcd4

from pyquaternion import Quaternion

from truckscenes.utils.geometry_utils import transform_matrix


class PointCloud(ABC):
    """
    Abstract class for manipulating and viewing point clouds.
    Every point cloud (lidar and radar) consists of points where:
    - Dimensions 0, 1, 2 represent x, y, z coordinates.
        These are modified when the point cloud is rotated or translated.
    - All other dimensions are optional. Hence these have to be manually modified
        if the reference frame changes.
    """

    def __init__(self, points: np.ndarray, timestamps: np.ndarray = None):
        """
        Initialize a point cloud and check it has the correct dimensions.
        :param points: <np.float: d, n>. d-dimensional input point cloud matrix.
        """
        assert points.shape[0] == self.nbr_dims(), \
            'Error: Pointcloud points must have format: %d x n' % self.nbr_dims()
        self.points = points
        self.timestamps = timestamps

    @staticmethod
    @abstractmethod
    def nbr_dims() -> int:
        """
        Returns the number of dimensions.
        :return: Number of dimensions.
        """
        pass

    @classmethod
    @abstractmethod
    def from_file(cls, file_name: str) -> PointCloud:
        """
        Loads point cloud from disk.
        :param file_name: Path of the pointcloud file on disk.
        :return: PointCloud instance.
        """
        pass

    @classmethod
    def from_file_multisweep(cls,
                             trucksc,
                             sample_rec: Dict,
                             chan: str,
                             ref_chan: str,
                             nsweeps: int = 5,
                             min_distance: float = 1.0) -> Tuple[PointCloud, np.ndarray]:
        """
        Return a point cloud that aggregates multiple sweeps.
        As every sweep is in a different coordinate frame, we need to map the coordinates
        to a single reference frame.
        As every sweep has a different timestamp, we need to account for that in
        the transformations and timestamps.

        Arguments:
            trucksc: A TruckScenes instance.
            sample_rec: The current sample.
            chan: The lidar/radar channel from which we track back n sweeps to aggregate
                the point cloud.
            ref_chan: The reference channel of the current sample_rec that the
                point clouds are mapped to.
            nsweeps: Number of sweeps to aggregated.
            min_distance: Distance below which points are discarded.

        Returns:
            all_pc: The aggregated point cloud.
            all_times: The aggregated timestamps.
        """
        # Init.
        points = np.zeros((cls.nbr_dims(), 0), dtype=np.float64)
        timestamps = np.zeros((1, 0), dtype=np.uint64)
        all_pc = cls(points, timestamps)
        all_times = np.zeros((1, 0))

        # Get reference pose and timestamp.
        ref_sd_token = sample_rec['data'][ref_chan]
        ref_sd_rec = trucksc.get('sample_data', ref_sd_token)
        ref_pose_rec = trucksc.get('ego_pose', ref_sd_rec['ego_pose_token'])
        ref_cs_rec = trucksc.get('calibrated_sensor', ref_sd_rec['calibrated_sensor_token'])
        ref_time = 1e-6 * ref_sd_rec['timestamp']

        # Homogeneous transform from ego car frame to reference frame.
        ref_from_car = transform_matrix(ref_cs_rec['translation'],
                                        Quaternion(ref_cs_rec['rotation']),
                                        inverse=True)

        # Homogeneous transformation matrix from global to _current_ ego car frame.
        car_from_global = transform_matrix(ref_pose_rec['translation'],
                                           Quaternion(ref_pose_rec['rotation']),
                                           inverse=True)

        # Aggregate current and previous sweeps.
        sample_data_token = sample_rec['data'][chan]
        current_sd_rec = trucksc.get('sample_data', sample_data_token)

        for _ in range(nsweeps):
            # Load up the pointcloud and remove points close to the sensor.
            current_pc = cls.from_file(osp.join(trucksc.dataroot, current_sd_rec['filename']))
            current_pc.remove_close(min_distance)

            # Get past pose.
            current_pose_rec = trucksc.get('ego_pose', current_sd_rec['ego_pose_token'])
            global_from_car = transform_matrix(current_pose_rec['translation'],
                                               Quaternion(current_pose_rec['rotation']),
                                               inverse=False)

            # Homogeneous transformation matrix from sensor coordinate frame to ego car frame.
            current_cs_rec = trucksc.get('calibrated_sensor',
                                       current_sd_rec['calibrated_sensor_token'])
            car_from_current = transform_matrix(current_cs_rec['translation'],
                                                Quaternion(current_cs_rec['rotation']),
                                                inverse=False)

            # Fuse four transformation matrices into one and perform transform.
            trans_matrix = reduce(np.dot, [ref_from_car, car_from_global,
                                           global_from_car, car_from_current])
            current_pc.transform(trans_matrix)

            # Add time vector which can be used as a temporal feature.
            if current_pc.timestamps is not None:
                # Per point difference
                time_lag = ref_time - 1e-6 * current_pc.timestamps
            else:
                # Difference to sample data
                time_lag = ref_time - 1e-6 * current_sd_rec['timestamp']
                time_lag = time_lag * np.ones((1, current_pc.nbr_points()))
            all_times = np.hstack((all_times, time_lag))

            # Merge with key pc.
            all_pc.points = np.hstack((all_pc.points, current_pc.points))
            if current_pc.timestamps is not None:
                all_pc.timestamps = np.hstack((all_pc.timestamps, current_pc.timestamps))

            # Abort if there are no previous sweeps.
            if current_sd_rec['prev'] == '':
                break
            else:
                current_sd_rec = trucksc.get('sample_data', current_sd_rec['prev'])

        return all_pc, all_times

    def nbr_points(self) -> int:
        """
        Returns the number of points.
        :return: Number of points.
        """
        return self.points.shape[1]

    def subsample(self, ratio: float) -> None:
        """
        Sub-samples the pointcloud.
        :param ratio: Fraction to keep.
        """
        selected_ind = np.random.choice(np.arange(0, self.nbr_points()),
                                        size=int(self.nbr_points() * ratio))
        self.points = self.points[:, selected_ind]

    def remove_close(self, radius: float) -> None:
        """
        Removes point too close within a certain radius from origin.
        :param radius: Radius below which points are removed.
        """

        x_filt = np.abs(self.points[0, :]) < radius
        y_filt = np.abs(self.points[1, :]) < radius
        not_close = np.logical_not(np.logical_and(x_filt, y_filt))
        self.points = self.points[:, not_close]

        if self.timestamps is not None:
            self.timestamps = self.timestamps[:, not_close]

    def translate(self, x: np.ndarray) -> None:
        """
        Applies a translation to the point cloud.
        :param x: <np.float: 3, 1>. Translation in x, y, z.
        """
        for i in range(3):
            self.points[i, :] = self.points[i, :] + x[i]

    def rotate(self, rot_matrix: np.ndarray) -> None:
        """
        Applies a rotation.
        :param rot_matrix: <np.float: 3, 3>. Rotation matrix.
        """
        self.points[:3, :] = np.dot(rot_matrix, self.points[:3, :])

    def transform(self, transf_matrix: np.ndarray) -> None:
        """
        Applies a homogeneous transform.
        :param transf_matrix: <np.float: 4, 4>. Homogenous transformation matrix.
        """
        self.points[:3, :] = transf_matrix.dot(
            np.vstack((self.points[:3, :], np.ones(self.nbr_points())))
        )[:3, :]

    def render_height(self,
                      ax,
                      view: np.ndarray = np.eye(4),
                      x_lim: Tuple[float, float] = (-20, 20),
                      y_lim: Tuple[float, float] = (-20, 20),
                      marker_size: float = 1) -> None:
        """
        Very simple method that applies a transformation and then scatter plots the points
        colored by height (z-value).

        :param ax: Axes on which to render the points.
        :param view: <np.float: n, n>. Defines an arbitrary projection (n <= 4).
        :param x_lim: (min, max). x range for plotting.
        :param y_lim: (min, max). y range for plotting.
        :param marker_size: Marker size.
        """
        # Initialize visualization methods
        try:
            _render_helper = getattr(import_module("truckscenes.utils.visualization_utils"),
                                     "_render_pc_helper")
        except ModuleNotFoundError:
            print('''The visualization dependencies are not installed on your system! '''
                  '''Run 'pip install "truckscenes-devkit[all]"'.''')

        # Render point cloud
        _render_helper(2, ax, view, x_lim, y_lim, marker_size)

    def render_intensity(self,
                         ax,
                         view: np.ndarray = np.eye(4),
                         x_lim: Tuple[float, float] = (-20, 20),
                         y_lim: Tuple[float, float] = (-20, 20),
                         marker_size: float = 1) -> None:
        """
        Very simple method that applies a transformation and then scatter plots the points
        colored by intensity.

        :param ax: Axes on which to render the points.
        :param view: <np.float: n, n>. Defines an arbitrary projection (n <= 4).
        :param x_lim: (min, max).
        :param y_lim: (min, max).
        :param marker_size: Marker size.
        """
        # Initialize visualization methods
        try:
            _render_helper = getattr(import_module("truckscenes.utils.visualization_utils"),
                                     "_render_pc_helper")
        except ModuleNotFoundError:
            warnings.warn('''The visualization dependencies are not installed on your system! '''
                          '''Run 'pip install "truckscenes-devkit[all]"'.''')

        # Render point cloud
        _render_helper(3, ax, view, x_lim, y_lim, marker_size)


class LidarPointCloud(PointCloud):

    @staticmethod
    def nbr_dims() -> int:
        """
        Returns the number of dimensions.
        :return: Number of dimensions.
        """
        return 4

    @classmethod
    def from_file(cls, file_name: str) -> LidarPointCloud:
        """
        Loads LIDAR data from binary numpy format. Data is stored
        as (x, y, z, intensity, ring index).

        :param file_name: Path of the pointcloud file on disk.
        :return: LidarPointCloud instance (x, y, z, intensity).
        """

        assert file_name.endswith('.pcd'), 'Unsupported filetype {}'.format(file_name)

        lidar = pypcd4.PointCloud.from_path(file_name)

        lidar_data = lidar.pc_data
        points = np.array([lidar_data["x"], lidar_data["y"], lidar_data["z"],
                           lidar_data["intensity"]], dtype=np.float64)

        return cls(points, np.atleast_2d(lidar_data["timestamp"]))


class RadarPointCloud(PointCloud):

    @staticmethod
    def nbr_dims() -> int:
        """
        Returns the number of dimensions.
        :return: Number of dimensions.
        """
        return 7

    @classmethod
    def from_file(cls,
                  file_name: str) -> RadarPointCloud:

        assert file_name.endswith('.pcd'), 'Unsupported filetype {}'.format(file_name)

        radar = pypcd4.PointCloud.from_path(file_name)

        radar_data = radar.pc_data
        points = np.array([radar_data["x"], radar_data["y"], radar_data["z"],
                           radar_data["vrel_x"], radar_data["vrel_y"], radar_data["vrel_z"],
                           radar_data["rcs"]], dtype=np.float64)

        return cls(points)


class Box:
    """ Simple data class representing a 3d box including, label, score and velocity. """

    def __init__(self,
                 center: List[float],
                 size: List[float],
                 orientation: Quaternion,
                 label: int = np.nan,
                 score: float = np.nan,
                 velocity: Tuple = (np.nan, np.nan, np.nan),
                 name: str = None,
                 token: str = None):
        """
        :param center: Center of box given as x, y, z.
        :param size: Size of box in width, length, height.
        :param orientation: Box orientation.
        :param label: Integer label, optional.
        :param score: Classification score, optional.
        :param velocity: Box velocity in x, y, z direction.
        :param name: Box name, optional. Can be used e.g. for denote category name.
        :param token: Unique string identifier from DB.
        """
        assert not np.any(np.isnan(center))
        assert not np.any(np.isnan(size))
        assert len(center) == 3
        assert len(size) == 3
        assert type(orientation) == Quaternion

        self.center = np.array(center)
        self.wlh = np.array(size)
        self.orientation = orientation
        self.label = int(label) if not np.isnan(label) else label
        self.score = float(score) if not np.isnan(score) else score
        self.velocity = np.array(velocity)
        self.name = name
        self.token = token

    def __eq__(self, other):
        center = np.allclose(self.center, other.center)
        wlh = np.allclose(self.wlh, other.wlh)
        orientation = np.allclose(self.orientation.elements, other.orientation.elements)
        label = (self.label == other.label) or (np.isnan(self.label) and np.isnan(other.label))
        score = (self.score == other.score) or (np.isnan(self.score) and np.isnan(other.score))
        vel = (np.allclose(self.velocity, other.velocity) or
               (np.all(np.isnan(self.velocity)) and np.all(np.isnan(other.velocity))))

        return center and wlh and orientation and label and score and vel

    def __repr__(self):
        repr_str = \
            f'label: {self.label}, ' \
            f'score: {self.score:.2f}, ' \
            f'xyz: [{self.center[0]:.2f}, {self.center[1]:.2f}, {self.center[2]:.2f}], ' \
            f'wlh: [{self.wlh[0]:.2f}, {self.wlh[1]:.2f}, {self.wlh[2]:.2f}], ' \
            f'rot axis: [{self.orientation.axis[0]:.2f}, {self.orientation.axis[1]:.2f}, ' \
            f'{self.orientation.axis[2]:.2f}], ' \
            f'ang(degrees): {self.orientation.degrees:.2f}, ' \
            f'ang(rad): {self.orientation.radians:.2f}, ' \
            f'vel: {self.velocity[0]:.2f}, {self.velocity[1]:.2f}, {self.velocity[2]:.2f}, ' \
            f'name: {self.name}, ' \
            f'token: {self.token}'

        return repr_str

    @property
    def rotation_matrix(self) -> np.ndarray:
        """
        Return a rotation matrix.
        :return: <np.float: 3, 3>. The box's rotation matrix.
        """
        return self.orientation.rotation_matrix

    def translate(self, x: np.ndarray) -> None:
        """
        Applies a translation.
        :param x: <np.float: 3, 1>. Translation in x, y, z direction.
        """
        self.center += x

    def rotate(self, quaternion: Quaternion) -> None:
        """
        Rotates box.
        :param quaternion: Rotation to apply.
        """
        self.center = np.dot(quaternion.rotation_matrix, self.center)
        self.orientation = quaternion * self.orientation
        self.velocity = np.dot(quaternion.rotation_matrix, self.velocity)

    def corners(self, wlh_factor: float = 1.0) -> np.ndarray:
        """
        Returns the bounding box corners.
        :param wlh_factor: Multiply w, l, h by a factor to scale the box.
        :return: <np.float: 3, 8>. First four corners are the ones facing forward.
            The last four are the ones facing backwards.
        """
        w, l, h = self.wlh * wlh_factor

        # 3D bounding box corners. (Convention: x points forward, y to the left, z up.)
        x_corners = l / 2 * np.array([1,  1,  1,  1, -1, -1, -1, -1])
        y_corners = w / 2 * np.array([1, -1, -1,  1,  1, -1, -1,  1])
        z_corners = h / 2 * np.array([1,  1, -1, -1,  1,  1, -1, -1])
        corners = np.vstack((x_corners, y_corners, z_corners))

        # Rotate
        corners = np.dot(self.orientation.rotation_matrix, corners)

        # Translate
        x, y, z = self.center
        corners[0, :] = corners[0, :] + x
        corners[1, :] = corners[1, :] + y
        corners[2, :] = corners[2, :] + z

        return corners

    def bottom_corners(self) -> np.ndarray:
        """
        Returns the four bottom corners.
        :return: <np.float: 3, 4>. Bottom corners. First two face forward, last two face backwards.
        """
        return self.corners()[:, [2, 3, 7, 6]]
    
    def render(self,
               axis,
               view: np.ndarray = np.eye(3),
               normalize: bool = False,
               colors: Tuple = ('b', 'r', 'k'),
               linewidth: float = 2) -> None:
        """
        Renders the box in the provided Matplotlib axis.
        :param axis: Axis onto which the box should be drawn.
        :param view: <np.array: 3, 3>. Define a projection in needed
            (e.g. for drawing projection in an image).
        :param normalize: Whether to normalize the remaining coordinate.
        :param colors: (<Matplotlib.colors>: 3). Valid Matplotlib colors
            (<str> or normalized RGB tuple) for front, back and sides.
        :param linewidth: Width in pixel of the box sides.
        """
        # Initialize visualization methods
        try:
            render_box = getattr(import_module("truckscenes.utils.visualization_utils"),
                                 "render_box")
        except ModuleNotFoundError:
            warnings.warn('''The visualization dependencies are not installed on your system! '''
                          '''Run 'pip install "truckscenes-devkit[all]"'.''')
        
        # Render box
        render_box(self, axis, view, normalize, colors, linewidth)
    
    def render_cv2(self,
                   im: np.ndarray,
                   view: np.ndarray = np.eye(3),
                   normalize: bool = False,
                   colors: Tuple = ((0, 0, 255), (255, 0, 0), (155, 155, 155)),
                   linewidth: int = 2) -> None:
        """
        Renders box using OpenCV2.
        :param im: <np.array: width, height, 3>. Image array. Channels are in BGR order.
        :param view: <np.array: 3, 3>. Define a projection if needed
            (e.g. for drawing projection in an image).
        :param normalize: Whether to normalize the remaining coordinate.
        :param colors: ((R, G, B), (R, G, B), (R, G, B)). Colors for front, side & rear.
        :param linewidth: Linewidth for plot.
        """
        # Initialize visualization methods
        try:
            render_box_cv2 = getattr(import_module("truckscenes.utils.visualization_utils"),
                                     "render_box_cv2")
        except ModuleNotFoundError:
            warnings.warn('''The visualization dependencies are not installed on your system! '''
                          '''Run 'pip install "truckscenes-devkit[all]"'.''')
        
        # Render box
        render_box_cv2(self, im, view, normalize, colors, linewidth)

    def copy(self) -> Box:
        """
        Create a copy of self.
        :return: A copy.
        """
        return copy.deepcopy(self)
