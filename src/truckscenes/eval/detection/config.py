# Copyright 2021 Motional
# Copyright 2024 MAN Truck & Bus SE

import json
import os

from truckscenes.eval.detection.data_classes import DetectionConfig


def config_factory(configuration_name: str) -> DetectionConfig:
    """
    Creates a DetectionConfig instance that can be used to initialize a TruckScenesEval instance.

    Note that this only works if the config file is located in
    the truckscenes/eval/detection/configs folder.

    Arguments:
        configuration_name: Name of desired configuration in eval_detection_configs.

    Returns:
        cfg: DetectionConfig instance.
    """

    # Check if config exists.
    this_dir = os.path.dirname(os.path.abspath(__file__))
    cfg_path = os.path.join(this_dir, 'configs', f'{configuration_name}.json')
    assert os.path.exists(cfg_path), \
        'Requested unknown configuration {}'.format(configuration_name)

    # Load config file and deserialize it.
    with open(cfg_path, 'r') as f:
        data = json.load(f)
    cfg = DetectionConfig.deserialize(data)

    return cfg
