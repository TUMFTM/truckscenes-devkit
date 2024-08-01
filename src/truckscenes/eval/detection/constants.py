# Copyright 2021 Motional
# Copyright 2024 MAN Truck & Bus SE

DETECTION_NAMES = ['car', 'truck', 'bus', 'trailer', 'other_vehicle', 'pedestrian',
                   'motorcycle', 'bicycle', 'traffic_cone', 'barrier', 'animal', 'traffic_sign']

PRETTY_DETECTION_NAMES = {'car': 'Car',
                          'truck': 'Truck',
                          'bus': 'Bus',
                          'trailer': 'Trailer',
                          'other_vehicle': 'Other Veh.',
                          'pedestrian': 'Pedestrian',
                          'motorcycle': 'Motorcycle',
                          'bicycle': 'Bicycle',
                          'traffic_cone': 'Traffic Cone',
                          'barrier': 'Barrier',
                          'animal': 'Animal',
                          'traffic_sign': 'Traffic Sign'}

DETECTION_COLORS = {'car': 'C0',
                    'truck': 'C1',
                    'bus': 'C2',
                    'trailer': 'C3',
                    'other_vehicle': 'C4',
                    'pedestrian': 'C5',
                    'motorcycle': 'C6',
                    'bicycle': 'C7',
                    'traffic_cone': 'C8',
                    'barrier': 'C9',
                    'animal': 'C10',
                    'traffic_sign': 'C11'}

ATTRIBUTE_NAMES = ['pedestrian.moving', 'pedestrian.sitting_lying_down', 'pedestrian.standing',
                   'cycle.with_rider', 'cycle.without_rider', 'vehicle.moving', 'vehicle.parked',
                   'vehicle.stopped', 'traffic_sign.pole_mounted', 'traffic_sign.overhanging',
                   'traffic_sign.temporary']

PRETTY_ATTRIBUTE_NAMES = {'pedestrian.moving': 'Ped. Moving',
                          'pedestrian.sitting_lying_down': 'Ped. Sitting',
                          'pedestrian.standing': 'Ped. Standing',
                          'cycle.with_rider': 'Cycle w/ Rider',
                          'cycle.without_rider': 'Cycle w/o Rider',
                          'vehicle.moving': 'Veh. Moving',
                          'vehicle.parked': 'Veh. Parked',
                          'vehicle.stopped': 'Veh. Stopped',
                          'traffic_sign.pole_mounted': 'Sign mounted',
                          'traffic_sign.overhanging': 'Sign over.',
                          'traffic_sign.temporary': 'Sign temp.'}

TP_METRICS = ['trans_err', 'scale_err', 'orient_err', 'vel_err', 'attr_err']

PRETTY_TP_METRICS = {'trans_err': 'Trans.', 'scale_err': 'Scale', 'orient_err': 'Orient.',
                     'vel_err': 'Vel.', 'attr_err': 'Attr.'}

TP_METRICS_UNITS = {'trans_err': 'm',
                    'scale_err': '1-IOU',
                    'orient_err': 'rad.',
                    'vel_err': 'm/s',
                    'attr_err': '1-acc.'}
