# nuScenes dev-kit.
# Code written by Holger Caesar & Oscar Beijbom, 2018.

import argparse
import json
import os
import random
import time
from typing import Tuple, Dict, Any

import numpy as np
from collections import defaultdict

from nuscenes import NuScenes
from nuscenes.eval.common.config import config_factory
from nuscenes.eval.common.data_classes import EvalBoxes
from nuscenes.eval.common.data_classes import EvalBox
from nuscenes.eval.common.loaders import load_prediction, load_gt, add_center_dist, filter_eval_boxes
from nuscenes.eval.detection.algo import accumulate, calc_ap, calc_ar, calc_tp, calc_fap, calc_far, calc_aap, calc_aar
from nuscenes.eval.detection.constants import TP_METRICS
from nuscenes.eval.detection.data_classes import DetectionConfig, DetectionMetrics, DetectionBox, \
    DetectionMetricDataList

from copy import deepcopy
from nuscenes.eval.detection.render import summary_plot, class_pr_curve, class_tp_curve, dist_pr_curve, visualize_sample
from tqdm import tqdm 
from itertools import tee 

import pdb

def window(iterable, size):
    iters = tee(iterable, size)
    for i in range(1, size):
        for each in iters[i:]:
            next(each, None)

    return zip(*iters)

def center_distance(gt_box: EvalBox, pred_box: EvalBox) -> float:
    """
    L2 distance between the box centers (xy only).
    :param gt_box: GT annotation sample.
    :param pred_box: Predicted sample.
    :return: L2 distance.
    """
    return np.linalg.norm(np.array(pred_box.translation[:2]) - np.array(gt_box.translation[:2]))


def get_time(nusc, src_token, dst_token):
    time_last = 1e-6 * nusc.get('sample', src_token)["timestamp"]
    time_first = 1e-6 * nusc.get('sample', dst_token)["timestamp"]
    time_diff = time_first - time_last

    return time_diff 

def forecast_boxes(nusc, box, velocity):

    forecast_tokens = box.forecast_sample_tokens
    forecast_timediff = [get_time(nusc, token[0], token[1]) for token in window(forecast_tokens, 2)]
    forecast_rotation = box.forecast_rotation
    forecast_velocity = box.forecast_velocity
    forecast_boxes = [box]

    if velocity == "none":
        for i in range(len(forecast_tokens) - 1):
            new_box = deepcopy(forecast_boxes[-1])
            new_box.translation = new_box.translation 
            new_box.rotation = forecast_rotation[0]
            forecast_boxes.append(new_box)

    if velocity == "linear":
        for i in range(len(forecast_tokens) - 1):
            new_box = deepcopy(forecast_boxes[-1])
            new_box.translation = new_box.translation + forecast_timediff[i] * np.append(forecast_velocity[0], 0)
            new_box.rotation = forecast_rotation[0]
            forecast_boxes.append(new_box)
    
    elif velocity == "nonlinear":
        for i in range(len(forecast_tokens) - 1):
            new_box = deepcopy(forecast_boxes[-1])
            new_box.translation = new_box.translation + forecast_timediff[i] * np.append(forecast_velocity[i], 0)
            new_box.rotation = forecast_rotation[i]
            forecast_boxes.append(new_box)    
    
    return forecast_boxes

def trajectory(nusc, box: DetectionBox, thresh : float = 0.5) -> float:
    
    target = forecast_boxes(nusc, box, velocity="nonlinear")[-1]

    static_forecast = forecast_boxes(nusc, box, velocity="none")[-1]
    linear_forecast = forecast_boxes(nusc, box, velocity="linear")[-1]

    if center_distance(target, static_forecast) < thresh:
        return "static"
    elif center_distance(target, linear_forecast) < thresh:
        return "linear"
    else:
        return "nonlinear"

def reverse_boxes(nusc, boxes):
    reverse_tokens = boxes.forecast_sample_tokens
    reverse_timediff = [get_time(nusc, token[0], token[1]) for token in window(reverse_tokens, 2)]
    reverse_rotation = boxes.forecast_rotation
    reverse_velocity = boxes.forecast_velocity
    reverse_boxes = [boxes]

    for i in range(len(reverse_tokens) - 1):
        new_box = deepcopy(reverse_boxes[-1])
        new_box.translation = new_box.translation - reverse_timediff[i] * np.append(reverse_velocity[i], 0)
        new_box.rotation = reverse_rotation[i]
        reverse_boxes.append(new_box)

    forecast_boxes = reverse_boxes[::-1]
    
    sample_token = forecast_boxes[0].sample_token
    forecast_sample_tokens = forecast_boxes[0].forecast_sample_tokens
    reverse_sample_tokens = forecast_boxes[0].reverse_sample_tokens

    forecast_velocity = forecast_boxes[0].forecast_velocity[::-1]
    forecast_rotation = forecast_boxes[0].forecast_rotation[::-1]
    forecast_rvelocity = forecast_boxes[0].forecast_rvelocity[::-1]
    forecast_rrotation = forecast_boxes[0].forecast_rrotation[::-1]
    velocity = forecast_velocity[0]
    rotation = forecast_rotation[0]
    rvelocity = forecast_rvelocity[0]
    rrotation = forecast_rrotation[0]
    translation = forecast_boxes[0].translation
    size = forecast_boxes[0].size
    num_pts = forecast_boxes[0].num_pts
    ego_translation = forecast_boxes[0].ego_translation
    detection_name = forecast_boxes[0].detection_name
    detection_score = forecast_boxes[0].detection_score
    attribute_name = forecast_boxes[0].attribute_name

    box = DetectionBox(sample_token,
                        forecast_sample_tokens,
                        reverse_sample_tokens,
                        translation,
                        size,
                        rotation,
                        forecast_rotation,
                        rrotation,
                        forecast_rrotation,
                        velocity,
                        forecast_velocity,
                        rvelocity,
                        forecast_rvelocity,
                        ego_translation,
                        num_pts,
                        detection_name,
                        detection_score,
                        attribute_name)
    return box

class DetectionEval:
    """
    This is the official nuScenes detection evaluation code.
    Results are written to the provided output_dir.

    nuScenes uses the following detection metrics:
    - Mean Average Precision (mAP): Uses center-distance as matching criterion; averaged over distance thresholds.
    - True Positive (TP) metrics: Average of translation, velocity, scale, orientation and attribute errors.
    - nuScenes Detection Score (NDS): The weighted sum of the above.

    Here is an overview of the functions in this method:
    - init: Loads GT annotations and predictions stored in JSON format and filters the boxes.
    - run: Performs evaluation and dumps the metric data to disk.
    - render: Renders various plots and dumps to disk.

    We assume that:
    - Every sample_token is given in the results, although there may be not predictions for that sample.

    Please see https://www.nuscenes.org/object-detection for more details.
    """
    def __init__(self,
                 nusc: NuScenes,
                 config: DetectionConfig,
                 result_path: str,
                 eval_set: str,
                 output_dir: str = None,
                 verbose: bool = True,
                 forecast: int = 6,
                 tp_pct: float = 0.6,
                 reverse: bool = False,
                 static_only: bool = False,
                 cohort_analysis: bool = False):
        """
        Initialize a DetectionEval object.
        :param nusc: A NuScenes object.
        :param config: A DetectionConfig object.
        :param result_path: Path of the nuScenes JSON result file.
        :param eval_set: The dataset split to evaluate on, e.g. train, val or test.
        :param output_dir: Folder to save plots and results to.
        :param verbose: Whether to print to stdout.
        """
        self.nusc = nusc
        self.result_path = result_path
        self.eval_set = eval_set
        self.output_dir = output_dir
        self.verbose = verbose
        self.cfg = config
        self.forecast = forecast
        self.tp_pct = tp_pct
        self.reverse = reverse
        self.static_only = static_only
        self.cohort_analysis = cohort_analysis

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
            print('Initializing nuScenes detection evaluation')
        self.pred_boxes, self.meta = load_prediction(self.result_path, self.cfg.max_boxes_per_sample, DetectionBox,
                                                     verbose=verbose)
        self.gt_boxes = load_gt(self.nusc, self.eval_set, DetectionBox, verbose=verbose, forecast=forecast)

        assert set(self.pred_boxes.sample_tokens) == set(self.gt_boxes.sample_tokens), \
            "Samples in split doesn't match samples in predictions."
        
        if self.cohort_analysis:
            for sample_token in self.pred_boxes.boxes.keys():
                for box in self.pred_boxes.boxes[sample_token]:
                    label = trajectory(nusc, box)
                    box.detection_name = label + "_" + box.detection_name

            for sample_token in self.gt_boxes.boxes.keys():
                for box in self.gt_boxes.boxes[sample_token]:
                    label = trajectory(nusc, box)
                    box.detection_name = label + "_" + box.detection_name

        if self.static_only:
            for sample_token in self.pred_boxes.boxes.keys():
                self.pred_boxes.boxes[sample_token] = [boxes for boxes in self.pred_boxes.boxes[sample_token] if np.linalg.norm(boxes.velocity) < 0.05]

        if self.reverse:
            for sample_token in self.pred_boxes.boxes.keys():
                self.pred_boxes.boxes[sample_token] = [reverse_boxes(self.nusc, boxes) for boxes in self.pred_boxes.boxes[sample_token]]

        # Add center distances.
        self.pred_boxes = add_center_dist(nusc, self.pred_boxes)
        self.gt_boxes = add_center_dist(nusc, self.gt_boxes)

        # Filter boxes (distance, points per box, etc.).
        if verbose:
            print('Filtering predictions')
        self.pred_boxes = filter_eval_boxes(nusc, self.pred_boxes, self.cfg.class_range, verbose=verbose)
        if verbose:
            print('Filtering ground truth annotations')
        self.gt_boxes = filter_eval_boxes(nusc, self.gt_boxes, self.cfg.class_range, verbose=verbose)
        
        self.sample_tokens = self.gt_boxes.sample_tokens


    def evaluate(self) -> Tuple[DetectionMetrics, DetectionMetricDataList]:
        """
        Performs the actual evaluation.
        :return: A tuple of high-level and the raw metric data.
        """
        start_time = time.time()

        # -----------------------------------
        # Step 1: Accumulate metric data for all classes and distance thresholds.
        # -----------------------------------
        if self.verbose:
            print('Accumulating metric data...')
        metric_data_list = DetectionMetricDataList()
        for class_name in tqdm(self.cfg.class_names):
            for dist_th in self.cfg.dist_ths:
                md = accumulate(self.nusc, self.gt_boxes, self.pred_boxes, class_name, self.cfg.dist_fcn_callable, dist_th)
                metric_data_list.set(class_name, dist_th, md)

        # -----------------------------------
        # Step 2: Calculate metrics from the data.
        # -----------------------------------
        if self.verbose:
            print('Calculating metrics...')
        metrics = DetectionMetrics(self.cfg)
        for class_name in self.cfg.class_names:
            # Compute APs.
            for dist_th in self.cfg.dist_ths:
                metric_data = metric_data_list[(class_name, dist_th)]
                ap = calc_ap(deepcopy(metric_data), self.cfg.min_recall, self.cfg.min_precision)
                ar = calc_ar(deepcopy(metric_data))

                fap = calc_fap(deepcopy(metric_data), self.cfg.min_recall, self.cfg.min_precision)
                far = calc_far(deepcopy(metric_data))

                aap = calc_aap(deepcopy(metric_data), self.cfg.min_recall, self.cfg.min_precision)
                aar = calc_aar(deepcopy(metric_data))

                metrics.add_label_ap(class_name, dist_th, ap)
                metrics.add_label_ar(class_name, dist_th, ar)
                metrics.add_label_fap(class_name, dist_th, fap)
                metrics.add_label_far(class_name, dist_th, far)
                metrics.add_label_aap(class_name, dist_th, aap)
                metrics.add_label_aar(class_name, dist_th, aar)

            # Compute TP metrics.
            for metric_name in TP_METRICS:
                metric_data = metric_data_list[(class_name, self.cfg.dist_th_tp)]
                if class_name in ['traffic_cone'] and metric_name in ['attr_err', 'vel_err', 'orient_err']:
                    tp = np.nan
                elif class_name in ['barrier'] and metric_name in ['attr_err', 'vel_err']:
                    tp = np.nan
                else:
                    tp = calc_tp(metric_data, self.cfg.min_recall, metric_name, self.tp_pct)
                metrics.add_label_tp(class_name, metric_name, tp)

        # Compute evaluation time.
        metrics.add_runtime(time.time() - start_time)

        return metrics, metric_data_list

    def render(self, metrics: DetectionMetrics, md_list: DetectionMetricDataList, cohort_analysis=False) -> None:
        """
        Renders various PR and TP curves.
        :param metrics: DetectionMetrics instance.
        :param md_list: DetectionMetricDataList instance.
        """
        if self.verbose:
            print('Rendering PR and TP curves')

        def savepath(name):
            return os.path.join(self.plot_dir, name + '.pdf')

        summary_plot(md_list, metrics, min_precision=self.cfg.min_precision, min_recall=self.cfg.min_recall,
                     dist_th_tp=self.cfg.dist_th_tp, savepath=savepath('summary'), cohort_analysis=cohort_analysis)

        for detection_name in self.cfg.class_names:
            class_pr_curve(md_list, metrics, detection_name, self.cfg.min_precision, self.cfg.min_recall,
                           savepath=savepath(detection_name + '_pr'))

            class_tp_curve(md_list, metrics, detection_name, self.cfg.min_recall, self.cfg.dist_th_tp,
                           savepath=savepath(detection_name + '_tp'))

        for dist_th in self.cfg.dist_ths:
            dist_pr_curve(md_list, metrics, dist_th, self.cfg.min_precision, self.cfg.min_recall,
                          savepath=savepath('dist_pr_' + str(dist_th)))

    def main(self,
             plot_examples: int = 0,
             render_curves: bool = True,
             cohort_analysis: bool = False) -> Dict[str, Any]:
        """
        Main function that loads the evaluation code, visualizes samples, runs the evaluation and renders stat plots.
        :param plot_examples: How many example visualizations to write to disk.
        :param render_curves: Whether to render PR and TP curves to disk.
        :return: A dict that stores the high-level metrics and meta data.
        """
        if plot_examples > 0:
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
                visualize_sample(self.nusc,
                                 sample_token,
                                 self.gt_boxes if self.eval_set != 'test' else EvalBoxes(),
                                 # Don't render test GT.
                                 self.pred_boxes,
                                 eval_range=max(self.cfg.class_range.values()),
                                 savepath=os.path.join(example_dir, '{}.png'.format(sample_token)))

        # Run evaluation.
        metrics, metric_data_list = self.evaluate()

        # Render PR and TP curves.
        if render_curves:
            self.render(metrics, metric_data_list, cohort_analysis=cohort_analysis)

        # Dump the metric data, meta and metrics to disk.
        if self.verbose:
            print('Saving metrics to: %s' % self.output_dir)
        metrics_summary = metrics.serialize()
        metrics_summary['meta'] = self.meta.copy()
        with open(os.path.join(self.output_dir, 'metrics_summary.json'), 'w') as f:
            json.dump(metrics_summary, f, indent=2)
        with open(os.path.join(self.output_dir, 'metrics_details.json'), 'w') as f:
            json.dump(metric_data_list.serialize(), f, indent=2)

        # Print high-level metrics.
        print('mAP: %.4f' % (metrics_summary['mean_ap']))
        print('mAR: %.4f' % (metrics_summary['mean_ar']))

        print('mFAP: %.4f' % (metrics_summary['mean_fap']))
        print('mFAR: %.4f' % (metrics_summary['mean_far']))

        print('mAAP: %.4f' % (metrics_summary['mean_aap']))
        print('mAAR: %.4f' % (metrics_summary['mean_aar']))

        err_name_mapping = {
            'trans_err': 'mATE',
            'scale_err': 'mASE',
            'orient_err': 'mAOE',
            'vel_err': 'mAVE',
            'attr_err': 'mAAE',
            'avg_disp_err' : 'mADE',
            'final_disp_err' : 'mFDE',
            'miss_rate' : 'mMR',
            #'reverse_avg_disp_err' : 'mRADE',
            #'reverse_final_disp_err' : 'mRFDE',
            #'reverse_miss_rate' : 'mRMR',
        }
        for tp_name, tp_val in metrics_summary['tp_errors'].items():
            print('%s: %.4f' % (err_name_mapping[tp_name], tp_val))
        print('NDS: %.4f' % (metrics_summary['nd_score']))
        print('Eval time: %.1fs' % metrics_summary['eval_time'])

        # Print per-class metrics.
        print()
        print('Per-class results:')
        print('Object Class\tAP\tAR\tFAP\tFAR\tAAP\tAAR\tATE\tASE\tAOE\tAVE\tAAE\tADE\tFDE\tMR')
        class_aps = metrics_summary['mean_dist_aps']
        class_ars = metrics_summary['mean_dist_ars']

        class_faps = metrics_summary['mean_dist_faps']
        class_fars = metrics_summary['mean_dist_fars']

        class_aaps = metrics_summary['mean_dist_aaps']
        class_aars = metrics_summary['mean_dist_aars']

        class_tps = metrics_summary['label_tp_errors']
        for class_name in class_aps.keys():
            print('%s\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f'
                  % (class_name, class_aps[class_name], class_ars[class_name], class_faps[class_name], class_fars[class_name], class_aaps[class_name], class_aars[class_name],
                     class_tps[class_name]['trans_err'],
                     class_tps[class_name]['scale_err'],
                     class_tps[class_name]['orient_err'],
                     class_tps[class_name]['vel_err'],
                     class_tps[class_name]['attr_err'],
                     class_tps[class_name]['avg_disp_err'],
                     class_tps[class_name]['final_disp_err'],
                     class_tps[class_name]['miss_rate'],
                     #class_tps[class_name]['reverse_avg_disp_err'],
                     #class_tps[class_name]['reverse_final_disp_err'],
                     #class_tps[class_name]['reverse_miss_rate'],
                     ))

        return metrics_summary


class NuScenesEval(DetectionEval):
    """
    Dummy class for backward-compatibility. Same as DetectionEval.
    """


if __name__ == "__main__":

    # Settings.
    parser = argparse.ArgumentParser(description='Evaluate nuScenes detection results.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('result_path', type=str, help='The submission as a JSON file.')
    parser.add_argument('--output_dir', type=str, default='~/nuscenes-metrics',
                        help='Folder to store result metrics, graphs and example visualizations.')
    parser.add_argument('--eval_set', type=str, default='val',
                        help='Which dataset split to evaluate on, train, val or test.')
    parser.add_argument('--dataroot', type=str, default='/data/sets/nuscenes',
                        help='Default nuScenes data directory.')
    parser.add_argument('--version', type=str, default='v1.0-trainval',
                        help='Which version of the nuScenes dataset to evaluate on, e.g. v1.0-trainval.')
    parser.add_argument('--config_path', type=str, default='',
                        help='Path to the configuration file.'
                             'If no path given, the CVPR 2019 configuration will be used.')
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
    plot_examples_ = args.plot_examples
    render_curves_ = bool(args.render_curves)
    verbose_ = bool(args.verbose)

    if config_path == '':
        cfg_ = config_factory('detection_cvpr_2019')
    else:
        with open(config_path, 'r') as _f:
            cfg_ = DetectionConfig.deserialize(json.load(_f))

    nusc_ = NuScenes(version=version_, verbose=verbose_, dataroot=dataroot_)
    nusc_eval = DetectionEval(nusc_, config=cfg_, result_path=result_path_, eval_set=eval_set_,
                              output_dir=output_dir_, verbose=verbose_)
    nusc_eval.main(plot_examples=plot_examples_, render_curves=render_curves_)
