# Copyright 2021 Motional
# Copyright 2024 MAN Truck & Bus SE

from typing import Callable, List

import numpy as np

from truckscenes.eval.common.data_classes import EvalBoxes
from truckscenes.eval.common.utils import center_distance, scale_iou, yaw_diff, velocity_l2, \
    attr_acc, cummean
from truckscenes.eval.detection.data_classes import DetectionMetricData


def accumulate(gt_boxes: EvalBoxes,
               pred_boxes: EvalBoxes,
               class_name: str,
               dist_fcn: Callable,
               dist_th: float,
               gt_mask: List[bool] = None,
               pred_mask: List[bool] = None,
               verbose: bool = False) -> DetectionMetricData:
    """
    Average Precision over predefined different recall thresholds for a single distance threshold.
    The recall/conf thresholds and other raw metrics will be used in secondary metrics.

    Arguments:
        gt_boxes: Maps every sample_token to a list of its sample_annotations.
        pred_boxes: Maps every sample_token to a list of its sample_results.
        class_name: Class to compute AP on.
        dist_fcn: Distance function used to match detections and ground truths.
        dist_th: Distance threshold for a match.
        gt_mask: Mask for ground truth boxes.
        pred_mask: Mask for predicted boxes.
        verbose: If true, print debug messages.

    Returns:
        The average precision value and raw data for a number of metrics (average_prec, metrics).
    """
    # ---------------------------------------------
    # Organize input and initialize accumulators.
    # ---------------------------------------------

    # Count the positives.
    npos = len([1 for gt_box in gt_boxes.all if gt_box.detection_name == class_name])

    # Mask ground truth boxes
    if gt_mask is not None:
        npos -= len([
            1 for gt_box in gt_boxes.all
            if gt_box.detection_name == class_name and not gt_mask[gt_box.sample_token]
        ])

    if verbose:
        print("Found {} GT of class {} out of {} total across {} samples.".
              format(npos, class_name, len(gt_boxes.all), len(gt_boxes.sample_tokens)))

    # For missing classes in the GT, return a data structure corresponding to no predictions.
    if npos == 0:
        return DetectionMetricData.no_predictions()

    # Organize the predictions in a single list.
    pred_boxes_list = [box for box in pred_boxes.all if box.detection_name == class_name]

    # Mask predicted boxes
    if pred_mask is not None:
        pred_boxes_list = [box for box in pred_boxes_list if pred_mask[box.sample_token]]

    pred_confs = [box.detection_score for box in pred_boxes_list]

    if verbose:
        print(f"Found {len(pred_confs)} PRED of class {class_name} out of {len(pred_boxes.all)} "
              f"total across {len(pred_boxes.sample_tokens)} samples.")

    # Sort by confidence.
    sortind = [i for (_, i) in sorted((v, i) for (i, v) in enumerate(pred_confs))][::-1]

    # Do the actual matching.
    tp = []  # Accumulator of true positives.
    fp = []  # Accumulator of false positives.
    conf = []  # Accumulator of confidences.

    # match_data holds the extra metrics we calculate for each match.
    match_data = {'trans_err': [],
                  'vel_err': [],
                  'scale_err': [],
                  'orient_err': [],
                  'attr_err': [],
                  'conf': []}

    # ---------------------------------------------
    # Match and accumulate match data.
    # ---------------------------------------------

    taken = set()  # Initially no gt bounding box is matched.
    for ind in sortind:
        pred_box = pred_boxes_list[ind]
        sample_token = pred_box.sample_token
        detection_score = pred_box.detection_score
        min_dist = np.inf
        match_gt_idx = None

        for gt_idx, gt_box in enumerate(gt_boxes[sample_token]):

            # Find closest match among ground truth boxes
            if (gt_box.detection_name == class_name and not (sample_token, gt_idx) in taken):
                this_distance = dist_fcn(gt_box, pred_box)
                if this_distance < min_dist:
                    min_dist = this_distance
                    match_gt_idx = gt_idx

        # If the closest match is close enough according to threshold we have a match!
        is_match = min_dist < dist_th

        if is_match:
            taken.add((sample_token, match_gt_idx))

            #  Update tp, fp and confs.
            tp.append(1)
            fp.append(0)
            conf.append(detection_score)

            # Since it is a match, update match data also.
            gt_box_match = gt_boxes[sample_token][match_gt_idx]

            match_data['trans_err'].append(center_distance(gt_box_match, pred_box))
            match_data['vel_err'].append(velocity_l2(gt_box_match, pred_box))
            match_data['scale_err'].append(1 - scale_iou(gt_box_match, pred_box))

            # Barrier and traffic_sign orientation is only determined up to 180 degree.
            # (For cones orientation is discarded later)
            if class_name in {'barrier', 'traffic_sign'}:
                period = np.pi
            else:
                period = 2 * np.pi
            match_data['orient_err'].append(yaw_diff(gt_box_match, pred_box, period=period))

            match_data['attr_err'].append(1 - attr_acc(gt_box_match, pred_box))
            match_data['conf'].append(detection_score)

        else:
            # No match. Mark this as a false positive.
            tp.append(0)
            fp.append(1)
            conf.append(detection_score)

    # Check if we have any matches. If not, just return a "no predictions" array.
    if len(match_data['trans_err']) == 0:
        return DetectionMetricData.no_predictions()

    # ---------------------------------------------
    # Calculate and interpolate precision and recall
    # ---------------------------------------------

    # Accumulate.
    tp = np.cumsum(tp).astype(float)
    fp = np.cumsum(fp).astype(float)
    conf = np.array(conf)

    # Calculate precision and recall.
    prec = tp / (fp + tp)
    rec = tp / float(npos)

    # 101 steps, from 0% to 100% recall.
    rec_interp = np.linspace(0, 1, DetectionMetricData.nelem)
    prec = np.interp(rec_interp, rec, prec, right=0)
    conf = np.interp(rec_interp, rec, conf, right=0)
    rec = rec_interp

    # ---------------------------------------------
    # Re-sample the match-data to match, prec, recall and conf.
    # ---------------------------------------------

    for key in match_data.keys():
        if key == "conf":
            # Confidence is used as reference to align with fp and tp. So skip in this step.
            continue

        else:
            # For each match_data, we first calculate the accumulated mean.
            tmp = cummean(np.array(match_data[key]))

            # Then interpolate based on the confidences.
            # (Note reversing since np.interp needs increasing arrays)
            match_data[key] = np.interp(conf[::-1], match_data['conf'][::-1], tmp[::-1])[::-1]

    # ---------------------------------------------
    # Done. Instantiate MetricData and return
    # ---------------------------------------------
    return DetectionMetricData(recall=rec,
                               precision=prec,
                               confidence=conf,
                               trans_err=match_data['trans_err'],
                               vel_err=match_data['vel_err'],
                               scale_err=match_data['scale_err'],
                               orient_err=match_data['orient_err'],
                               attr_err=match_data['attr_err'])


def calc_ap(md: DetectionMetricData, min_recall: float, min_precision: float) -> float:
    """ Calculated average precision. """

    assert 0 <= min_precision < 1
    assert 0 <= min_recall <= 1

    prec = np.copy(md.precision)

    # Clip low recalls. +1 to exclude the min recall bin.
    prec = prec[round(100 * min_recall) + 1:]

    # Clip low precision
    prec -= min_precision
    prec[prec < 0] = 0

    return float(np.mean(prec)) / (1.0 - min_precision)


def calc_tp(md: DetectionMetricData, min_recall: float, metric_name: str) -> float:
    """ Calculates true positive errors. """

    # +1 to exclude the error at min recall.
    first_ind = round(100 * min_recall) + 1

    # First instance of confidence = 0 is index of max achieved recall.
    last_ind = md.max_recall_ind

    if last_ind < first_ind:
        # Assign 1 here. If this happens for all classes, the score for that TP metric will be 0.
        return 1.0
    else:
        # +1 to include error at max recall.
        return float(np.mean(getattr(md, metric_name)[first_ind: last_ind + 1]))
