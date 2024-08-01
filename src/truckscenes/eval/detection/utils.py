# Copyright 2021 Motional
# Copyright 2024 MAN Truck & Bus SE

import json

from typing import List, Optional

import numpy as np

from truckscenes.eval.detection.constants import DETECTION_NAMES, \
    TP_METRICS_UNITS, PRETTY_DETECTION_NAMES


def category_to_detection_name(category_name: str) -> Optional[str]:
    """
    Default label mapping from TruckScenes to TruckScenes detection classes.
    Note that pedestrian does not include personal_mobility, stroller and wheelchair.
    :param category_name: Generic TruckScenes class.
    :return: TruckScenes detection class.
    """
    detection_mapping = {
        'animal': 'animal',
        'movable_object.barrier': 'barrier',
        'vehicle.bicycle': 'bicycle',
        'vehicle.bus.bendy': 'bus',
        'vehicle.bus.rigid': 'bus',
        'vehicle.car': 'car',
        'vehicle.motorcycle': 'motorcycle',
        'vehicle.construction': 'other_vehicle',
        'vehicle.other': 'other_vehicle',
        'human.pedestrian.adult': 'pedestrian',
        'human.pedestrian.child': 'pedestrian',
        'human.pedestrian.construction_worker': 'pedestrian',
        'human.pedestrian.police_officer': 'pedestrian',
        'movable_object.trafficcone': 'traffic_cone',
        'static_object.traffic_sign': 'traffic_sign',
        'vehicle.ego_trailer': 'trailer',
        'vehicle.trailer': 'trailer',
        'vehicle.truck': 'truck'
    }

    if category_name in detection_mapping:
        return detection_mapping[category_name]
    else:
        return None


def detection_name_to_rel_attributes(detection_name: str) -> List[str]:
    """
    Returns a list of relevant attributes for a given detection class.
    :param detection_name: The detection class.
    :return: List of relevant attributes.
    """
    if detection_name in ['pedestrian']:
        rel_attributes = ['pedestrian.moving', 'pedestrian.sitting_lying_down',
                          'pedestrian.standing']
    elif detection_name in ['bicycle', 'motorcycle']:
        rel_attributes = ['cycle.with_rider', 'cycle.without_rider']
    elif detection_name in ['car', 'bus', 'other_vehicle', 'trailer', 'truck']:
        rel_attributes = ['vehicle.moving', 'vehicle.parked', 'vehicle.stopped']
    elif detection_name in ['traffic_sign']:
        rel_attributes = ['traffic_sign.pole_mounted', 'traffic_sign.overhanging',
                          'traffic_sign.temporary']
    elif detection_name in ['barrier', 'traffic_cone', 'animal']:
        # Classes without attributes: barrier, traffic_cone.
        rel_attributes = []
    else:
        raise ValueError('Error: %s is not a valid detection class.' % detection_name)

    return rel_attributes


def detailed_results_table_tex(metrics_path: str, output_path: str) -> None:
    """
    Renders a detailed results table in tex.
    :param metrics_path: path to a serialized DetectionMetrics file.
    :param output_path: path to the output file.
    """
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)

    tex = ''
    tex += '\\begin{table}[]\n'
    tex += '\\small\n'
    tex += '\\begin{tabular}{| c | c | c | c | c | c | c |} \\hline\n'
    tex += '\\textbf{Class}    &   \\textbf{AP}  &   \\textbf{ATE} &   \\textbf{ASE} & ' \
           '\\textbf{AOE}   & ' \
           '\\textbf{AVE}   & ' \
           '\\textbf{AAE}   \\\\ \\hline ' \
           '\\hline\n'
    for name in DETECTION_NAMES:
        ap = np.mean(metrics['label_aps'][name].values()) * 100
        ate = metrics['label_tp_errors'][name]['trans_err']
        ase = metrics['label_tp_errors'][name]['scale_err']
        aoe = metrics['label_tp_errors'][name]['orient_err']
        ave = metrics['label_tp_errors'][name]['vel_err']
        aae = metrics['label_tp_errors'][name]['attr_err']
        tex_name = PRETTY_DETECTION_NAMES[name]
        if name == 'traffic_cone':
            tex += f'{tex_name}  &   {ap:.1f}  &   {ate:.2f}  &   {ase:.2f}  &   ' \
                f'N/A  &   N/A  &   N/A  \\\\ \\hline\n'
        elif name == 'barrier':
            tex += f'{tex_name}  &   {ap:.1f}  &   {ate:.2f}  &   {ase:.2f}  &   ' \
                f'{aoe:.2f}  &   N/A  &   N/A  \\\\ \\hline\n'
        elif name == 'animal':
            tex += f'{tex_name}  &   {ap:.1f}  &   {ate:.2f}  &   {ase:.2f}  &   ' \
                f'{aoe:.2f}  &   {ave:.2f}  &   N/A  \\\\ \\hline\n'
        elif name == 'traffic_sign':
            tex += f'{tex_name}  &   {ap:.1f}  &   {ate:.2f}  &   {ase:.2f}  &   ' \
                f'{aoe:.2f}  &   N/A  &   {aae:.2f}  \\\\ \\hline\n'
        else:
            tex += f'{tex_name}  &   {ap:.1f}  &   {ate:.2f}  &   {ase:.2f}  &   ' \
                f'{aoe:.2f}  &   {ave:.2f}  &   {aae:.2f}  \\\\ \\hline\n'

    map_ = metrics['mean_ap']
    mate = metrics['tp_errors']['trans_err']
    mase = metrics['tp_errors']['scale_err']
    maoe = metrics['tp_errors']['orient_err']
    mave = metrics['tp_errors']['vel_err']
    maae = metrics['tp_errors']['attr_err']
    tex += f'\\hline \\textbf{{Mean}} &   {map_:.1f}  &   {mate:.2f}  &   {mase:.2f}  &   ' \
        f'{maoe:.2f}  &   {mave:.2f}  &   {maae:.2f}  \\\\ ' \
        '\\hline\n'

    tex += '\\end{tabular}\n'

    # All one line
    tex += '\\caption{Detailed detection performance on the val set. \n'
    tex += 'AP: average precision averaged over distance thresholds (%), \n'
    tex += 'ATE: average translation error (${}$), \n'.format(TP_METRICS_UNITS['trans_err'])
    tex += 'ASE: average scale error (${}$), \n'.format(TP_METRICS_UNITS['scale_err'])
    tex += 'AOE: average orientation error (${}$), \n'.format(TP_METRICS_UNITS['orient_err'])
    tex += 'AVE: average velocity error (${}$), \n'.format(TP_METRICS_UNITS['vel_err'])
    tex += 'AAE: average attribute error (${}$). \n'.format(TP_METRICS_UNITS['attr_err'])
    tex += 'nuScenes Detection Score (NDS) = {:.1f} \n'.format(metrics['nd_score'] * 100)
    tex += '}\n'

    tex += '\\end{table}\n'

    with open(output_path, 'w') as f:
        f.write(tex)
