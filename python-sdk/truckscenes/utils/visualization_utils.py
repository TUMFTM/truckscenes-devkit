import os
import os.path as osp

from datetime import datetime
from typing import Dict, List, Tuple, Union

import cv2
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d

from PIL import Image
from matplotlib import cm, rcParams
from matplotlib.axes import Axes
from matplotlib.colors import Colormap, Normalize
from matplotlib.cm import ScalarMappable
from pyquaternion import Quaternion

from truckscenes.utils import colormap
from truckscenes.utils.data_classes import LidarPointCloud, RadarPointCloud
from truckscenes.utils.geometry_utils import view_points, transform_matrix, \
    BoxVisibility


class TruckScenesExplorer:
    """ Helper class to list and visualize truckscenes data. These are meant to serve
    as tutorials and templates for working with the data.
    """
    def __init__(self, trucksc):
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
            cmap = plt.get_c[cmap]

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
            cmap = cm.get_cmap(cmap)

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
            ax.set_title(sample_token)
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
            cmap = cm.get_cmap(cmap)

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
        rgb = cm.get_cmap(cmap)(intensities)[..., :3]

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
                bbox.color = np.asarray(colormap.get_colormap()[box.name]) / 255
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


def render_box(box,
               axis: Axes,
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
        corners = view_points(box.corners(), view, normalize=normalize)[:2, :]

        def draw_rect(selected_corners, color):
            prev = selected_corners[-1]
            for corner in selected_corners:
                axis.plot([prev[0], corner[0]], [prev[1], corner[1]],
                          color=color, linewidth=linewidth)
                prev = corner

        # Draw the sides
        for i in range(4):
            axis.plot([corners.T[i][0], corners.T[i + 4][0]],
                      [corners.T[i][1], corners.T[i + 4][1]],
                      color=colors[2], linewidth=linewidth)

        # Draw front (first 4 corners) and rear (last 4 corners) rectangles(3d)/lines(2d)
        draw_rect(corners.T[:4], colors[0])
        draw_rect(corners.T[4:], colors[1])

        # Draw line indicating the front
        center_bottom_forward = np.mean(corners.T[2:4], axis=0)
        center_bottom = np.mean(corners.T[[2, 3, 7, 6]], axis=0)
        axis.plot([center_bottom[0], center_bottom_forward[0]],
                  [center_bottom[1], center_bottom_forward[1]],
                  color=colors[0], linewidth=linewidth)


def render_box_cv2(box,
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
    corners = view_points(box.corners(), view, normalize=normalize)[:2, :]

    def draw_rect(selected_corners, color):
        prev = selected_corners[-1]
        for corner in selected_corners:
            cv2.line(im,
                     (int(prev[0]), int(prev[1])),
                     (int(corner[0]), int(corner[1])),
                     color, linewidth)
            prev = corner

    # Draw the sides
    for i in range(4):
        cv2.line(im,
                 (int(corners.T[i][0]), int(corners.T[i][1])),
                 (int(corners.T[i + 4][0]), int(corners.T[i + 4][1])),
                 colors[2][::-1], linewidth)

    # Draw front (first 4 corners) and rear (last 4 corners) rectangles(3d)/lines(2d)
    draw_rect(corners.T[:4], colors[0][::-1])
    draw_rect(corners.T[4:], colors[1][::-1])

    # Draw line indicating the front
    center_bottom_forward = np.mean(corners.T[2:4], axis=0)
    center_bottom = np.mean(corners.T[[2, 3, 7, 6]], axis=0)
    cv2.line(im,
             (int(center_bottom[0]), int(center_bottom[1])),
             (int(center_bottom_forward[0]), int(center_bottom_forward[1])),
             colors[0][::-1], linewidth)


def _render_pc_helper(pc,
                      color_channel: int,
                      ax: Axes,
                      view: np.ndarray,
                      x_lim: Tuple[float, float],
                      y_lim: Tuple[float, float],
                      marker_size: float) -> None:
    """
    Helper function for rendering.
    :param color_channel: Point channel to use as color.
    :param ax: Axes on which to render the points.
    :param view: <np.float: n, n>. Defines an arbitrary projection (n <= 4).
    :param x_lim: (min, max).
    :param y_lim: (min, max).
    :param marker_size: Marker size.
    """
    points = view_points(pc.points[:3, :], view, normalize=False)
    ax.scatter(points[0, :], points[1, :], c=pc.points[color_channel, :], s=marker_size)
    ax.set_xlim(x_lim[0], x_lim[1])
    ax.set_ylim(y_lim[0], y_lim[1])
