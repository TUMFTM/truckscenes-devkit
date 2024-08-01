# Copyright 2021 Motional
# Copyright 2024 MAN Truck & Bus SE

import argparse
import json
import os
import random
import time
import warnings

from importlib import import_module
from typing import Tuple, Dict, Any

import numpy as np

from truckscenes import TruckScenes
from truckscenes.eval.common.constants import TAG_NAMES
from truckscenes.eval.common.data_classes import EvalBoxes
from truckscenes.eval.common.loaders import load_prediction, load_gt, add_center_dist, \
    get_scene_tag_masks, filter_eval_boxes
from truckscenes.eval.detection.algo import accumulate, calc_ap, calc_tp
from truckscenes.eval.detection.config import config_factory
from truckscenes.eval.detection.constants import TP_METRICS
from truckscenes.eval.detection.data_classes import DetectionConfig, DetectionMetrics, \
    DetectionMetricsList, DetectionBox, DetectionMetricDataList


class DetectionEval:
    """
    This is the official MAN TruckScenes detection evaluation code.
    Results are written to the provided output_dir.

    TruckScenes uses the following detection metrics:
    - Mean Average Precision (mAP): Uses center-distance as matching criterion;
        averaged over distance thresholds.
    - True Positive (TP) metrics: Average of translation, velocity, scale,
        orientation and attribute errors.
    - nuScenes Detection Score (NDS): The weighted sum of the above.

    Here is an overview of the functions in this method:
    - init: Loads GT annotations and predictions stored in JSON format and filters the boxes.
    - run: Performs evaluation and dumps the metric data to disk.
    - render: Renders various plots and dumps to disk.

    We assume that:
    - Every sample_token is given in the results, although there may be not
        predictions for that sample.

    Please see https://www.nuscenes.org/object-detection for more details.
    """
    def __init__(self,
                 trucksc: TruckScenes,
                 config: DetectionConfig,
                 result_path: str,
                 eval_set: str,
                 output_dir: str = None,
                 verbose: bool = True):
        """
        Initialize a DetectionEval object.
        :param trucksc: A TruckScenes object.
        :param config: A DetectionConfig object.
        :param result_path: Path of the TruckScenes JSON result file.
        :param eval_set: The dataset split to evaluate on, e.g. train, val or test.
        :param output_dir: Folder to save plots and results to.
        :param verbose: Whether to print to stdout.
        """
        self.trucksc = trucksc
        self.result_path = result_path
        self.eval_set = eval_set
        self.output_dir = output_dir
        self.verbose = verbose
        self.cfg = config

        # Check result file exists.
        assert os.path.exists(result_path), 'Error: The result file does not exist!'

        # Make dirs.
        self.plot_dir = os.path.join(self.output_dir, 'plots')
        if not os.path.isdir(self.output_dir):
            os.makedirs(self.output_dir)
        if not os.path.isdir(self.plot_dir):
            os.makedirs(self.plot_dir)

        # Load data.
        if verbose:
            print('Initializing MAN TruckScenes detection evaluation')
        self.pred_boxes, self.meta = load_prediction(self.result_path,
                                                     self.cfg.max_boxes_per_sample, DetectionBox,
                                                     verbose=verbose)
        self.gt_boxes = load_gt(self.trucksc, self.eval_set, DetectionBox, verbose=verbose)

        assert set(self.pred_boxes.sample_tokens) == set(self.gt_boxes.sample_tokens), \
            "Samples in split doesn't match samples in predictions."

        # Add center distances.
        self.pred_boxes = add_center_dist(trucksc, self.pred_boxes)
        self.gt_boxes = add_center_dist(trucksc, self.gt_boxes)

        # Filter boxes (distance, points per box, etc.).
        if verbose:
            print('Filtering predictions')
        self.pred_boxes = filter_eval_boxes(trucksc, self.pred_boxes, self.cfg.class_range,
                                            verbose=verbose)
        if verbose:
            print('Filtering ground truth annotations')
        self.gt_boxes = filter_eval_boxes(trucksc, self.gt_boxes, self.cfg.class_range,
                                          verbose=verbose)

        self.sample_tokens = self.gt_boxes.sample_tokens

        # Add scene tag masks.
        self.pred_boxes_masks = get_scene_tag_masks(trucksc, self.pred_boxes)
        self.gt_boxes_masks = get_scene_tag_masks(trucksc, self.gt_boxes)

    def evaluate(self,
                 evaluate_tags: bool = False) -> Tuple[DetectionMetrics, DetectionMetricDataList]:
        """
        Performs the actual evaluation.
        :param evaluate_tags: Whether to evaluate tag wise.
        :return: A tuple of high-level and the raw metric data.
        """
        start_time = time.time()

        # Set tags to evaluate
        tag_names = ['all']
        if evaluate_tags:
            tag_names += TAG_NAMES

        # -----------------------------------
        # Step 1: Accumulate metric data for all classes and distance thresholds.
        # -----------------------------------
        if self.verbose:
            print('Accumulating metric data...')
        metric_data_list = DetectionMetricDataList()
        for tag_name in tag_names:
            for class_name in self.cfg.class_names:
                for dist_th in self.cfg.dist_ths:
                    md = accumulate(self.gt_boxes, self.pred_boxes, class_name,
                                    self.cfg.dist_fcn_callable, dist_th,
                                    self.gt_boxes_masks.get(tag_name),
                                    self.pred_boxes_masks.get(tag_name))
                    metric_data_list.set(tag_name, class_name, dist_th, md)

        # -----------------------------------
        # Step 2: Calculate metrics from the data.
        # -----------------------------------
        if self.verbose:
            print('Calculating metrics...')
        metrics_list = DetectionMetricsList()
        for tag_name in tag_names:
            metrics = DetectionMetrics(self.cfg)
            for class_name in self.cfg.class_names:
                # Compute APs.
                for dist_th in self.cfg.dist_ths:
                    metric_data = metric_data_list[(tag_name, class_name, dist_th)]
                    if tag_name in ['weather.other_weather', 'weather.snow', 'weather.hail',
                                    'area.parking', 'area.other_area', 'season.spring',
                                    'lighting.other_lighting']:
                        ap = np.nan
                    else:
                        ap = calc_ap(metric_data, self.cfg.min_recall, self.cfg.min_precision)
                    metrics.add_label_ap(class_name, dist_th, ap)

                # Compute TP metrics.
                for metric_name in TP_METRICS:
                    metric_data = metric_data_list[(tag_name, class_name, self.cfg.dist_th_tp)]
                    if tag_name in ['weather.other_weather', 'weather.snow', 'weather.hail',
                                    'area.parking', 'area.other_area', 'season.spring',
                                    'lighting.other_lighting']:
                        tp = np.nan
                    elif (
                        class_name in ['traffic_cone'] and
                        metric_name in ['attr_err', 'vel_err', 'orient_err']
                    ):
                        tp = np.nan
                    elif class_name in ['barrier'] and metric_name in ['attr_err', 'vel_err']:
                        tp = np.nan
                    elif class_name in ['animal'] and metric_name in ['attr_err']:
                        tp = np.nan
                    elif class_name in ['traffic_sign'] and metric_name in ['vel_err']:
                        tp = np.nan
                    else:
                        tp = calc_tp(metric_data, self.cfg.min_recall, metric_name)
                    metrics.add_label_tp(class_name, metric_name, tp)

            metrics_list.add_detection_metrics(tag_name, metrics)

        # Compute evaluation time.
        metrics_list.add_runtime(time.time() - start_time)

        return metrics_list, metric_data_list

    def render(self, metrics: DetectionMetrics, md_list: DetectionMetricDataList) -> None:
        """
        Renders various PR and TP curves.
        :param metrics: DetectionMetrics instance.
        :param md_list: DetectionMetricDataList instance.
        """
        # Initialize render module
        try:
            summary_plot = getattr(import_module("truckscenes.eval.detection.render"),
                                   "summary_plot")
            class_pr_curve = getattr(import_module("truckscenes.eval.detection.render"),
                                     "class_pr_curve")
            class_tp_curve = getattr(import_module("truckscenes.eval.detection.render"),
                                     "class_tp_curve")
            dist_pr_curve = getattr(import_module("truckscenes.eval.detection.render"),
                                    "dist_pr_curve")
        except ModuleNotFoundError:
            warnings.warn('''The visualization dependencies are not installed on your system! '''
                          '''Run 'pip install "truckscenes-devkit[all]"'.''')
        else:
            # Render curves
            if self.verbose:
                print('Rendering PR and TP curves')

            def savepath(name):
                return os.path.join(self.plot_dir, name + '.pdf')

            summary_plot(md_list, metrics, min_precision=self.cfg.min_precision,
                        min_recall=self.cfg.min_recall, dist_th_tp=self.cfg.dist_th_tp,
                        savepath=savepath('summary'))

            for detection_name in self.cfg.class_names:
                class_pr_curve(md_list, metrics, detection_name, self.cfg.min_precision,
                            self.cfg.min_recall, savepath=savepath(detection_name + '_pr'))

                class_tp_curve(md_list, metrics, detection_name, self.cfg.min_recall,
                            self.cfg.dist_th_tp, savepath=savepath(detection_name + '_tp'))

            for dist_th in self.cfg.dist_ths:
                dist_pr_curve(md_list, metrics, dist_th, self.cfg.min_precision, self.cfg.min_recall,
                            savepath=savepath('dist_pr_' + str(dist_th)))
            
    def _plot_examples(self, plot_examples: int) -> None:
        """
        Plot randomly selected examples.
        :param plot_examples: Number of examples to plot.
        """
        # Initialize render module
        try:
            visualize_sample = getattr(import_module("truckscenes.eval.detection.render"),
                                       "visualize_sample")
        except ModuleNotFoundError:
            warnings.warn('''The visualization dependencies are not installed on your system! '''
                          '''Run 'pip install "truckscenes-devkit[all]"'.''')
        else:
            # Select a random but fixed subset to plot.
            random.seed(42)
            sample_tokens = list(self.sample_tokens)
            random.shuffle(sample_tokens)
            sample_tokens = sample_tokens[:plot_examples]

            # Visualize samples.
            example_dir = os.path.join(self.output_dir, 'examples')
            if not os.path.isdir(example_dir):
                os.mkdir(example_dir)
            for sample_token in sample_tokens:
                visualize_sample(self.trucksc,
                                 sample_token,
                                 self.gt_boxes if self.eval_set != 'test' else EvalBoxes(),
                                 # Don't render test GT.
                                 self.pred_boxes,
                                 eval_range=max(self.cfg.class_range.values()),
                                 savepath=os.path.join(example_dir, '{}.png'.format(sample_token)))

    def main(self,
             plot_examples: int = 0,
             render_curves: bool = True,
             evaluate_tags: bool = False) -> Dict[str, Any]:
        """
        Main function that loads the evaluation code, visualizes samples,
        runs the evaluation and renders stat plots.

        :param plot_examples: How many example visualizations to write to disk.
        :param render_curves: Whether to render PR and TP curves to disk.
        :return: A dict that stores the high-level metrics and meta data.
        """
        if plot_examples > 0:
            self._plot_examples(plot_examples)

        # Run evaluation.
        metrics_list, metric_data_list = self.evaluate(evaluate_tags)

        # Render PR and TP curves.
        if render_curves:
            self.render(metrics_list['all'], metric_data_list)

        # Dump the metric data, meta and metrics to disk.
        if self.verbose:
            print('Saving metrics to: %s' % self.output_dir)
        metrics_summary = metrics_list.serialize()
        metrics_summary['meta'] = self.meta.copy()
        with open(os.path.join(self.output_dir, 'metrics_summary.json'), 'w') as f:
            json.dump(metrics_summary, f, indent=2)
        with open(os.path.join(self.output_dir, 'metrics_details.json'), 'w') as f:
            json.dump(metric_data_list.serialize(), f, indent=2)

        # Print high-level metrics.
        print()
        print('High-level results:')
        print('mAP: %.4f' % (metrics_summary['all']['mean_ap']))
        err_name_mapping = {
            'trans_err': 'mATE',
            'scale_err': 'mASE',
            'orient_err': 'mAOE',
            'vel_err': 'mAVE',
            'attr_err': 'mAAE'
        }
        for tp_name, tp_val in metrics_summary['all']['tp_errors'].items():
            print('%s: %.4f' % (err_name_mapping[tp_name], tp_val))
        print('NDS: %.4f' % (metrics_summary['all']['nd_score']))
        print('Eval time: %.1fs' % metrics_summary['eval_time'])

        # Print per-class metrics.
        print()
        print('Per-class results:')
        print('%-22s\t%-6s\t%-6s\t%-6s' % ('Object Class', 'AP', 'ATE', 'ASE') +
              '\t%-6s\t%-6s\t%-6s' % ('AOE', 'AVE', 'AAE'))
        class_aps = metrics_summary['all']['mean_dist_aps']
        class_tps = metrics_summary['all']['label_tp_errors']
        for class_name in class_aps.keys():
            print('%-22s\t%-6.3f\t%-6.3f\t%-6.3f\t%-6.3f\t%-6.3f\t%-6.3f' % (
                class_name, class_aps[class_name],
                class_tps[class_name]['trans_err'],
                class_tps[class_name]['scale_err'],
                class_tps[class_name]['orient_err'],
                class_tps[class_name]['vel_err'],
                class_tps[class_name]['attr_err']
            ))

        if not evaluate_tags:
            return metrics_summary

        # Print per-tag metrics.
        print()
        print('Per-tag results:')
        print('%-22s\t%-6s\t%-6s\t%-6s' % ('Scene Tag', 'mAP', 'mATE', 'mASE') +
              '\t%-6s\t%-6s\t%-6s\t%-6s' % ('mAOE', 'mAVE', 'mAAE', 'NDS'))
        for tag_name in TAG_NAMES:
            print('%-22s\t%-6.3f\t%-6.3f\t%-6.3f\t%-6.3f\t%-6.3f\t%-6.3f\t%-6.3f' % (
                tag_name,
                metrics_summary[tag_name]['mean_ap'],
                metrics_summary[tag_name]['tp_errors']['trans_err'],
                metrics_summary[tag_name]['tp_errors']['scale_err'],
                metrics_summary[tag_name]['tp_errors']['orient_err'],
                metrics_summary[tag_name]['tp_errors']['vel_err'],
                metrics_summary[tag_name]['tp_errors']['attr_err'],
                metrics_summary[tag_name]['nd_score']
            ))

        return metrics_summary


class TruckScenesEval(DetectionEval):
    """
    Dummy class for backward-compatibility. Same as DetectionEval.
    """


if __name__ == "__main__":

    # Settings.
    parser = argparse.ArgumentParser(description='Evaluate MAN TruckScenes detection results.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('result_path', type=str, help='The submission as a JSON file.')
    parser.add_argument('--output_dir', type=str, default='~/truckscenes-metrics',
                        help='Folder to store result metrics, graphs and example visualizations.')
    parser.add_argument('--eval_set', type=str, default='val',
                        help='Which dataset split to evaluate on, train, val or test.')
    parser.add_argument('--dataroot', type=str, default='/data/truckscenes',
                        help='Default TruckScenes data directory.')
    parser.add_argument('--version', type=str, default='v1.0-trainval',
                        help='Which version of the TruckScenes dataset to evaluate on')
    parser.add_argument('--config_path', type=str, default='',
                        help='Path to the configuration file.'
                             'If no path given, the CVPR 2024 configuration will be used.')
    parser.add_argument('--evaluate_tags', type=int, default=0,
                        help='Whether to evaluate tag-wise.')
    parser.add_argument('--plot_examples', type=int, default=10,
                        help='How many example visualizations to write to disk.')
    parser.add_argument('--render_curves', type=int, default=1,
                        help='Whether to render PR and TP curves to disk.')
    parser.add_argument('--verbose', type=int, default=1,
                        help='Whether to print to stdout.')
    args = parser.parse_args()

    result_path_ = os.path.expanduser(args.result_path)
    output_dir_ = os.path.expanduser(args.output_dir)
    eval_set_ = args.eval_set
    dataroot_ = args.dataroot
    version_ = args.version
    config_path = args.config_path
    evaluate_tags_ = args.evaluate_tags
    plot_examples_ = args.plot_examples
    render_curves_ = bool(args.render_curves)
    verbose_ = bool(args.verbose)

    if config_path == '':
        cfg_ = config_factory('detection_cvpr_2024')
    else:
        with open(config_path, 'r') as _f:
            cfg_ = DetectionConfig.deserialize(json.load(_f))

    trucksc_ = TruckScenes(version=version_, verbose=verbose_, dataroot=dataroot_)
    trucksc_eval = DetectionEval(trucksc_, config=cfg_, result_path=result_path_,
                                 eval_set=eval_set_, output_dir=output_dir_, verbose=verbose_)
    trucksc_eval.main(plot_examples=plot_examples_, render_curves=render_curves_,
                      evaluate_tags=evaluate_tags_)
