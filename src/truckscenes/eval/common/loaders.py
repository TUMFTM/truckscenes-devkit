# Copyright 2021 Motional
# Copyright 2024 MAN Truck & Bus SE

import json
from typing import Dict, Tuple

import numpy as np
import tqdm

from collections import defaultdict
from pyquaternion import Quaternion

from truckscenes import TruckScenes
from truckscenes.eval.common.constants import TAG_NAMES
from truckscenes.eval.common.data_classes import EvalBoxes
from truckscenes.eval.detection.data_classes import DetectionBox
from truckscenes.eval.detection.utils import category_to_detection_name
from truckscenes.utils.data_classes import Box
from truckscenes.utils.geometry_utils import points_in_box
from truckscenes.utils.splits import create_splits_scenes


def load_prediction(result_path: str, max_boxes_per_sample: int, box_cls, verbose: bool = False) \
        -> Tuple[EvalBoxes, Dict]:
    """
    Loads object predictions from file.
    :param result_path: Path to the .json result file provided by the user.
    :param max_boxes_per_sample: Maximim number of boxes allowed per sample.
    :param box_cls: Type of box to load, e.g. DetectionBox or TrackingBox.
    :param verbose: Whether to print messages to stdout.
    :return: The deserialized results and meta data.
    """

    # Load from file and check that the format is correct.
    with open(result_path) as f:
        data = json.load(f)
    assert 'results' in data, 'Error: No field `results` in result file.'

    # Deserialize results and get meta data.
    all_results = EvalBoxes.deserialize(data['results'], box_cls)
    meta = data['meta']
    if verbose:
        print("Loaded results from {}. Found detections for {} samples."
              .format(result_path, len(all_results.sample_tokens)))

    # Check that each sample has no more than x predicted boxes.
    for sample_token in all_results.sample_tokens:
        assert len(all_results.boxes[sample_token]) <= max_boxes_per_sample, \
            "Error: Only <= %d boxes per sample allowed!" % max_boxes_per_sample

    return all_results, meta


def load_gt(trucksc: TruckScenes, eval_split: str, box_cls, verbose: bool = False) -> EvalBoxes:
    """
    Loads ground truth boxes from DB.
    :param trucksc: A TruckScenes instance.
    :param eval_split: The evaluation split for which we load GT boxes.
    :param box_cls: Type of box to load, e.g. DetectionBox or TrackingBox.
    :param verbose: Whether to print messages to stdout.
    :return: The GT boxes.
    """
    # Init.
    if box_cls == DetectionBox:
        attribute_map = {a['token']: a['name'] for a in trucksc.attribute}

    if verbose:
        print(f'Loading annotations for {eval_split} split'
              f'from TruckScenes version: {trucksc.version}')
    # Read out all sample_tokens in DB.
    sample_tokens_all = [s['token'] for s in trucksc.sample]
    assert len(sample_tokens_all) > 0, "Error: Database has no samples!"

    # Only keep samples from this split.
    splits = create_splits_scenes()

    # Check compatibility of split with trucksc_version.
    version = trucksc.version
    if eval_split in {'train', 'val', 'train_detect', 'train_track'}:
        assert version.endswith('trainval'), \
            f'Error: Requested split {eval_split} ' \
            f'which is not compatible with TruckScenes version {version}'
    elif eval_split in {'mini_train', 'mini_val'}:
        assert version.endswith('mini'), \
            f'Error: Requested split {eval_split} ' \
            f'which is not compatible with TruckScenes version {version}'
    elif eval_split == 'test':
        assert version.endswith('test'), \
            f'Error: Requested split {eval_split} ' \
            f'which is not compatible with TruckScenes version {version}'
    else:
        raise ValueError(
            f'Error: Requested split {eval_split} '
            f'which this function cannot map to the correct TruckScenes version.'
        )

    if eval_split == 'test':
        # Check that you aren't trying to cheat :).
        assert len(trucksc.sample_annotation) > 0, \
            'Error: You are trying to evaluate on the test set but you dont have the annotations!'

    sample_tokens = []
    for sample_token in sample_tokens_all:
        scene_token = trucksc.get('sample', sample_token)['scene_token']
        scene_record = trucksc.get('scene', scene_token)
        if scene_record['name'] in splits[eval_split]:
            sample_tokens.append(sample_token)

    all_annotations = EvalBoxes()

    # Load annotations and filter predictions and annotations.
    for sample_token in tqdm.tqdm(sample_tokens, leave=verbose):

        sample = trucksc.get('sample', sample_token)
        sample_annotation_tokens = sample['anns']

        sample_boxes = []
        for sample_annotation_token in sample_annotation_tokens:

            sample_annotation = trucksc.get('sample_annotation', sample_annotation_token)
            if box_cls == DetectionBox:
                # Get label name in detection task and filter unused labels.
                detection_name = category_to_detection_name(sample_annotation['category_name'])
                if detection_name is None:
                    continue

                # Get attribute_name.
                attr_tokens = sample_annotation['attribute_tokens']
                attr_count = len(attr_tokens)
                if attr_count == 0:
                    attribute_name = ''
                elif attr_count == 1:
                    attribute_name = attribute_map[attr_tokens[0]]
                else:
                    raise Exception('Error: GT annotations must not have more than one attribute!')

                num_pts = sample_annotation['num_lidar_pts'] + sample_annotation['num_radar_pts']

                sample_boxes.append(
                    box_cls(
                        sample_token=sample_token,
                        translation=sample_annotation['translation'],
                        size=sample_annotation['size'],
                        rotation=sample_annotation['rotation'],
                        velocity=trucksc.box_velocity(sample_annotation['token'])[:2],
                        num_pts=num_pts,
                        detection_name=detection_name,
                        detection_score=-1.0,  # GT samples do not have a score.
                        attribute_name=attribute_name
                    )
                )
            else:
                raise NotImplementedError('Error: Invalid box_cls %s!' % box_cls)

        all_annotations.add_boxes(sample_token, sample_boxes)

    if verbose:
        print(f"Loaded ground truth annotations for {len(all_annotations.sample_tokens)} samples.")

    return all_annotations


def add_center_dist(trucksc: TruckScenes,
                    eval_boxes: EvalBoxes):
    """
    Adds the cylindrical (xy) center distance from ego vehicle to each box.
    :param trucksc: The TruckScenes instance.
    :param eval_boxes: A set of boxes, either GT or predictions.
    :return: eval_boxes augmented with center distances.
    """
    for sample_token in eval_boxes.sample_tokens:
        sample_rec = trucksc.get('sample', sample_token)
        sd_record = trucksc.get('sample_data', sample_rec['data']['LIDAR_LEFT'])
        pose_record = trucksc.get('ego_pose', sd_record['ego_pose_token'])

        for box in eval_boxes[sample_token]:
            # Both boxes and ego pose are given in global coord system,
            # so distance can be calculated directly.
            # Note that the z component of the ego pose is 0.
            ego_translation = (box.translation[0] - pose_record['translation'][0],
                               box.translation[1] - pose_record['translation'][1],
                               box.translation[2] - pose_record['translation'][2])
            if isinstance(box, DetectionBox):
                box.ego_translation = ego_translation
            else:
                raise NotImplementedError

    return eval_boxes


def get_scene_tag_masks(trucksc: TruckScenes,
                        eval_boxes: EvalBoxes):
    """
    Adds masks for the individual scene tags.
    :param trucksc: The TruckScenes instance.
    :param eval_boxes: A set of boxes, either GT or predictions.
    :return: eval_boxes augmented with center distances.
    """
    masks: Dict[str, Dict[str, bool]] = defaultdict(dict)

    for sample_token in eval_boxes.sample_tokens:
        sample_rec = trucksc.get('sample', sample_token)
        scene_rec = trucksc.get('scene', sample_rec['scene_token'])

        # Get scene tags from scene description
        tags = set(scene_rec['description'].split(';'))

        for tag_name in TAG_NAMES:
            if tag_name in tags:
                masks[tag_name][sample_token] = True
            else:
                masks[tag_name][sample_token] = False

    return masks


def filter_eval_boxes(trucksc: TruckScenes,
                      eval_boxes: EvalBoxes,
                      max_dist: Dict[str, float],
                      verbose: bool = False) -> EvalBoxes:
    """
    Applies filtering to boxes. Distance, bike-racks and points per box.
    :param trucksc: An instance of the TruckScenes class.
    :param eval_boxes: An instance of the EvalBoxes class.
    :param max_dist: Maps the detection name to the eval distance threshold for that class.
    :param verbose: Whether to print to stdout.
    """
    # Retrieve box type for detectipn/tracking boxes.
    class_field = _get_box_class_field(eval_boxes)

    # Accumulators for number of filtered boxes.
    total, dist_filter, point_filter, bike_rack_filter = 0, 0, 0, 0
    for ind, sample_token in enumerate(eval_boxes.sample_tokens):

        # Filter on distance first.
        total += len(eval_boxes[sample_token])
        eval_boxes.boxes[sample_token] = [
            box for box in eval_boxes[sample_token] if
            box.ego_dist < max_dist[box.__getattribute__(class_field)]
        ]
        dist_filter += len(eval_boxes[sample_token])

        # Then remove boxes with zero points in them. Eval boxes have -1 points by default.
        eval_boxes.boxes[sample_token] = [
            box for box in eval_boxes[sample_token] if not box.num_pts == 0
        ]
        point_filter += len(eval_boxes[sample_token])

        # Perform bike-rack filtering.
        sample_anns = trucksc.get('sample', sample_token)['anns']
        bikerack_recs = [
            trucksc.get('sample_annotation', ann) for ann in sample_anns if
            trucksc.get('sample_annotation', ann)['category_name'] == 'static_object.bicycle_rack'
        ]
        bikerack_boxes = [
            Box(rec['translation'], rec['size'], Quaternion(rec['rotation']))
            for rec in bikerack_recs
        ]
        filtered_boxes = []
        for box in eval_boxes[sample_token]:
            if box.__getattribute__(class_field) in ['bicycle', 'motorcycle']:
                in_a_bikerack = False
                for bikerack_box in bikerack_boxes:
                    num_points_in_box = np.sum(
                        points_in_box(
                            bikerack_box,
                            np.expand_dims(np.array(box.translation), axis=1)
                        )
                    )
                    if num_points_in_box > 0:
                        in_a_bikerack = True
                if not in_a_bikerack:
                    filtered_boxes.append(box)
            else:
                filtered_boxes.append(box)

        eval_boxes.boxes[sample_token] = filtered_boxes
        bike_rack_filter += len(eval_boxes.boxes[sample_token])

    if verbose:
        print("=> Original number of boxes: %d" % total)
        print("=> After distance based filtering: %d" % dist_filter)
        print("=> After LIDAR and RADAR points based filtering: %d" % point_filter)
        print("=> After bike rack filtering: %d" % bike_rack_filter)

    return eval_boxes


def _get_box_class_field(eval_boxes: EvalBoxes) -> str:
    """
    Retrieve the name of the class field in the boxes.
    This parses through all boxes until it finds a valid box.
    If there are no valid boxes, this function throws an exception.
    :param eval_boxes: The EvalBoxes used for evaluation.
    :return: The name of the class field in the boxes, e.g. detection_name or tracking_name.
    """
    assert len(eval_boxes.boxes) > 0
    box = None
    for val in eval_boxes.boxes.values():
        if len(val) > 0:
            box = val[0]
            break
    if isinstance(box, DetectionBox):
        class_field = 'detection_name'
    else:
        raise Exception('Error: Invalid box type: %s' % box)

    return class_field
