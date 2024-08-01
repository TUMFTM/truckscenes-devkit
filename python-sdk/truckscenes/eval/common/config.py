# Copyright 2021 Motional
# Copyright 2024 MAN Truck & Bus SE

import json
import os

from truckscenes.eval.detection.data_classes import DetectionConfig


def config_factory(configuration_name: str) -> DetectionConfig:
    """Creates a DetectionConfig instance that can be used to initialize a DetectionEval instance

    Note that this only works if the config file is located in the
    truckscenes/eval/common/configs folder.

    Arguments:
        configuration_name: Name of desired configuration in eval_detection_configs.

    Returns:
        cfg: A DetectionConfig instance.
    """
    # Check prefix
    tokens = configuration_name.split('_')
    assert len(tokens) > 1, 'Error: Configuration name must have prefix "detection_"'

    # Check if config exists
    task = tokens[0]
    this_dir = os.path.dirname(os.path.abspath(__file__))
    cfg_path = os.path.join(this_dir, '..', task, 'configs', f'{configuration_name}.json')
    assert os.path.exists(cfg_path), \
        f'Requested unknown configuration {configuration_name}'

    # Load config file and deserialize it
    with open(cfg_path, 'r') as f:
        data = json.load(f)
    if task == 'detection':
        cfg = DetectionConfig.deserialize(data)
    else:
        raise Exception('Error: Invalid config file name: %s' % configuration_name)

    return cfg
