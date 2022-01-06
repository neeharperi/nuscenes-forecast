# nuScenes dev-kit.
# Code written by Holger Caesar, Varun Bankiti, and Alex Lang, 2019.

import json
from typing import Any

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from nuscenes import NuScenes
from nuscenes.eval.common.data_classes import EvalBoxes
from nuscenes.eval.common.render import setup_axis
from nuscenes.eval.common.utils import boxes_to_sensor
from nuscenes.eval.detection.constants import TP_METRICS, getDetectionNames, DETECTION_COLORS, TP_METRICS_UNITS, \
    PRETTY_DETECTION_NAMES, PRETTY_TP_METRICS, getDetectionNames
from nuscenes.eval.detection.data_classes import DetectionMetrics, DetectionMetricData, DetectionMetricDataList
from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.utils.geometry_utils import view_points
from nuscenes.utils.data_classes import Box
from pyquaternion import Quaternion

import cv2
Axis = Any
import pdb
from copy import deepcopy
from itertools import tee 

def window(iterable, size):
    iters = tee(iterable, size)
    for i in range(1, size):
        for each in iters[i:]:
            next(each, None)

    return zip(*iters)

def z_offset(points, 
                  meters_max=45,
                  pixels_per_meter=2,
                  hist_max_per_pixel=50,
                  zbins=np.array([-3.,   0.0, 1., 2.,  3., 10.]),
                  hist_normalize=True):
    assert(points.shape[-1] >= 3)
    assert(points.shape[0] > points.shape[1])
    meters_total = meters_max * 2
    pixels_total = meters_total * pixels_per_meter
    xbins = np.linspace(-meters_max, meters_max, pixels_total + 1, endpoint=True)
    ybins = xbins
    # The first left bin edge must match the last right bin edge.
    assert(np.isclose(xbins[0], -1 * xbins[-1]))
    assert(np.isclose(ybins[0], -1 * ybins[-1]))
    
    hist = np.histogramdd(points[..., :3], bins=(xbins, ybins, zbins), normed=False)[0]

    # Clip histogram 
    hist[hist > hist_max_per_pixel] = hist_max_per_pixel

    # Normalize histogram by the maximum number of points in a bin we care about.
    if hist_normalize:
        overhead_splat = hist / hist_max_per_pixel
    else:
        overhead_splat = hist

    return overhead_splat, xbins, ybins, zbins

def visualize_sample(nusc: NuScenes,
                     sample_token: str,
                     gt_boxes: EvalBoxes,
                     pred_boxes: EvalBoxes,
                     nsweeps: int = 1,
                     conf_th: float = 0.15,
                     eval_range: float = 50,
                     verbose: bool = True,
                     savepath: str = None) -> None:
    """
    Visualizes a sample from BEV with annotations and detection results.
    :param nusc: NuScenes object.
    :param sample_token: The nuScenes sample token.
    :param gt_boxes: Ground truth boxes grouped by sample.
    :param pred_boxes: Prediction grouped by sample.
    :param nsweeps: Number of sweeps used for lidar visualization.
    :param conf_th: The confidence threshold used to filter negatives.
    :param eval_range: Range in meters beyond which boxes are ignored.
    :param verbose: Whether to print to stdout.
    :param savepath: If given, saves the the rendering here instead of displaying.
    """
    # Retrieve sensor & pose records.
    sample_rec = nusc.get('sample', sample_token)
    sd_record = nusc.get('sample_data', sample_rec['data']['LIDAR_TOP'])
    cs_record = nusc.get('calibrated_sensor', sd_record['calibrated_sensor_token'])
    pose_record = nusc.get('ego_pose', sd_record['ego_pose_token'])

    # Get boxes.
    if gt_boxes is None and pred_boxes is None:
        boxes_gt_global = []
        boxes_est_global = []
    else:
        boxes_gt_global = gt_boxes[sample_token]
        boxes_est_global = pred_boxes[sample_token]

    # Map GT boxes to lidar.
    boxes_gt = boxes_to_sensor(boxes_gt_global, pose_record, cs_record)

    # Map EST boxes to lidar.
    boxes_est = boxes_to_sensor(boxes_est_global, pose_record, cs_record)

    # Add scores to EST boxes.
    for box_est, box_est_global in zip(boxes_est, boxes_est_global):
        box_est.score = box_est_global.detection_score

    # Get point cloud in lidar frame.
    pc, _ = LidarPointCloud.from_file_multisweep(nusc, sample_rec, 'LIDAR_TOP', 'LIDAR_TOP', nsweeps=20)

    # Init axes.
    _, ax = plt.subplots(1, 1, figsize=(9, 9))

    # Show point cloud.
    points = view_points(pc.points[:3, :], np.eye(4), normalize=False)
    dists = np.sqrt(np.sum(pc.points[:2, :] ** 2, axis=0))
    colors = np.minimum(1, dists / eval_range)
    ax.scatter(points[0, :], points[1, :], c=colors, s=0.2)

    # Show ego vehicle.
    ax.plot(0, 0, 'x', color='black')

    # Show GT boxes.
    for box in boxes_gt:
        box.render(ax, view=np.eye(4), colors=('g', 'g', 'g'), linewidth=2)

    # Show EST boxes.
    for box in boxes_est:
        # Show only predictions with a high score.
        assert not np.isnan(box.score), 'Error: Box score cannot be NaN!'
        if box.score >= conf_th:
            box.render(ax, view=np.eye(4), colors=('b', 'b', 'b'), linewidth=1)

    # Limit visible range.
    axes_limit = eval_range + 3  # Slightly bigger to include boxes that extend beyond the range.
    ax.set_xlim(-axes_limit, axes_limit)
    ax.set_ylim(-axes_limit, axes_limit)

    # Show / save plot.
    if verbose:
        print('Rendering sample token %s' % sample_token)
    plt.title(sample_token)
    if savepath is not None:
        plt.savefig(savepath)
        plt.close()
    else:
        plt.show()

def visualize_sample_forecast(nusc: NuScenes,
                              sample_token: str,
                              boxes_gt_global,
                              boxes_est_global,
                              nsweeps: int = 1,
                              conf_th: float = 0.15,
                              eval_range: float = 50,
                              verbose: bool = True,
                              savepath: str = None,
                              classname = [],
                              dets_only = False) -> None:
    """
    Visualizes a sample from BEV with annotations and detection results.
    :param nusc: NuScenes object.
    :param sample_token: The nuScenes sample token.
    :param gt_boxes: Ground truth boxes grouped by sample.
    :param pred_boxes: Prediction grouped by sample.
    :param nsweeps: Number of sweeps used for lidar visualization.
    :param conf_th: The confidence threshold used to filter negatives.
    :param eval_range: Range in meters beyond which boxes are ignored.
    :param verbose: Whether to print to stdout.
    :param savepath: If given, saves the the rendering here instead of displaying.
    """
    # Retrieve sensor & pose records.
    sample_rec = nusc.get('sample', sample_token)
    sd_record = nusc.get('sample_data', sample_rec['data']['LIDAR_TOP'])
    cs_record = nusc.get('calibrated_sensor', sd_record['calibrated_sensor_token'])
    pose_record = nusc.get('ego_pose', sd_record['ego_pose_token'])
    
    # Map GT boxes to lidar.
    boxes_gt, center_gt = [], []

    for boxes in boxes_gt_global:
        bxs, cntr = [], []

        for box in boxes.forecast_boxes:
            b = Box(center = box["translation"],
                    size = box["size"],
                    orientation = Quaternion(box["rotation"]),
                    velocity = np.array(list(box["velocity"]) + [0]),
                    name = boxes.detection_name,
                    token = box["sample_token"]
                    )

            # Move box to ego vehicle coord system.
            b.translate(-np.array(pose_record['translation']))
            b.rotate(Quaternion(pose_record['rotation']).inverse)

            #  Move box to sensor coord system.
            b.translate(-np.array(cs_record['translation']))
            b.rotate(Quaternion(cs_record['rotation']).inverse)

            bxs.append(b)
            cntr.append(b.center[:2])

        boxes_gt.append(bxs[0])
        center_gt.append(cntr)

    # Map EST boxes to lidar.
    boxes_est, center_est = [], []
    
    for boxes in boxes_est_global:
        bxs, cntr = [], []

        for box in boxes.forecast_boxes:
            b = Box(center = box["translation"],
                    size = box["size"],
                    orientation = Quaternion(box["rotation"]),
                    velocity = np.array(list(box["velocity"]) + [0]),
                    name = boxes.detection_name,
                    score = box["detection_score"],
                    token = box["sample_token"]
                    )

            # Move box to ego vehicle coord system.
            b.translate(-np.array(pose_record['translation']))
            b.rotate(Quaternion(pose_record['rotation']).inverse)

            #  Move box to sensor coord system.
            b.translate(-np.array(cs_record['translation']))
            b.rotate(Quaternion(cs_record['rotation']).inverse)

            bxs.append(b)
            cntr.append(b.center[:2])

        boxes_est.append(bxs[0])
        center_est.append(cntr)

    # Get point cloud in lidar frame.
    pc, _ = LidarPointCloud.from_file_multisweep(nusc, sample_rec, 'LIDAR_TOP', 'LIDAR_TOP', nsweeps=20)

    # Init axes.
    _, ax = plt.subplots(1, 1, figsize=(9, 9))

    # Show point cloud.
    points = view_points(pc.points[:3, :], np.eye(4), normalize=False)
    dists = np.sqrt(np.sum(pc.points[:2, :] ** 2, axis=0))
    colors = '#d3d3d3'
    ax.scatter(points[0, :], points[1, :], c=colors, s=0.2)

    # Show ego vehicle.
    ax.plot(0, 0, 'x', color='black')

    # Show GT boxes.
    for box, center in zip(boxes_gt, center_gt):
        if box.name not in classname:
            continue

        box.render_forecast(ax, view=np.eye(4), colors=('g', 'g', 'g'), linewidth=2, center=center)

    # Show EST boxes.
    for box, center in zip(boxes_est, center_est):
        if box.name not in classname and not dets_only: 
            continue

        # Show only predictions with a high score.
        assert not np.isnan(box.score), 'Error: Box score cannot be NaN!'
        if box.score >= conf_th:
            # Move box to ego vehicle coord system.

            if dets_only:
                if box.name in [0]:
                    clr = ('b', 'b', 'b')
                #elif box.name in [2]:
                #    clr = ('c', 'c', 'c')
                elif box.name in [4]:
                    clr = ('m', 'm', 'm')
                else:
                    continue
            else:
                clr = ('b', 'b', 'b')
            box.render_forecast(ax, view=np.eye(4), colors=clr, linewidth=1, center=center)

    # Limit visible range.
    axes_limit = eval_range + 3  # Slightly bigger to include boxes that extend beyond the range.
    ax.set_xlim(-axes_limit, axes_limit)
    ax.set_ylim(-axes_limit, axes_limit)
    plt.axis('off')

    #plt.title(sample_token)
    if savepath is not None:
        plt.savefig(savepath)
        plt.close()
    else:
        plt.show()

def class_pr_curve(md_list: DetectionMetricDataList,
                   metrics: DetectionMetrics,
                   detection_name: str,
                   min_precision: float,
                   min_recall: float,
                   savepath: str = None,
                   ax: Axis = None) -> None:
    """
    Plot a precision recall curve for the specified class.
    :param md_list: DetectionMetricDataList instance.
    :param metrics: DetectionMetrics instance.
    :param detection_name: The detection class.
    :param min_precision:
    :param min_recall: Minimum recall value.
    :param savepath: If given, saves the the rendering here instead of displaying.
    :param ax: Axes onto which to render.
    """
    # Prepare axis.
    if ax is None:
        ax = setup_axis(title=PRETTY_DETECTION_NAMES[detection_name], xlabel='Recall', ylabel='Precision', xlim=1,
                        ylim=1, min_precision=min_precision, min_recall=min_recall)

    # Get recall vs precision values of given class for each distance threshold.
    data = md_list.get_class_data(detection_name)

    # Plot the recall vs. precision curve for each distance threshold.
    for md, dist_th in data:
        md: DetectionMetricData
        ap = metrics.get_label_ap(detection_name, dist_th)
        ax.plot(md.recall, md.precision, label='Dist. : {}, AP: {:.1f}'.format(dist_th, ap * 100))

    ax.legend(loc='best')
    if savepath is not None:
        plt.savefig(savepath)
        plt.close()


def class_tp_curve(md_list: DetectionMetricDataList,
                   metrics: DetectionMetrics,
                   detection_name: str,
                   min_recall: float,
                   dist_th_tp: float,
                   savepath: str = None,
                   ax: Axis = None) -> None:
    """
    Plot the true positive curve for the specified class.
    :param md_list: DetectionMetricDataList instance.
    :param metrics: DetectionMetrics instance.
    :param detection_name:
    :param min_recall: Minimum recall value.
    :param dist_th_tp: The distance threshold used to determine matches.
    :param savepath: If given, saves the the rendering here instead of displaying.
    :param ax: Axes onto which to render.
    """
    # Get metric data for given detection class with tp distance threshold.
    md = md_list[(detection_name, dist_th_tp)]
    min_recall_ind = round(100 * min_recall)
    if min_recall_ind <= md.max_recall_ind:
        # For traffic_cone and barrier only a subset of the metrics are plotted.
        rel_metrics = [m for m in TP_METRICS if not np.isnan(metrics.get_label_tp(detection_name, m))]
        ylimit = max([max(getattr(md, metric)[min_recall_ind:md.max_recall_ind + 1]) for metric in rel_metrics]) * 1.1
    else:
        ylimit = 1.0

    # Prepare axis.
    if ax is None:
        ax = setup_axis(title=PRETTY_DETECTION_NAMES[detection_name], xlabel='Recall', ylabel='Error', xlim=1,
                        min_recall=min_recall)
    ax.set_ylim(0, ylimit)

    # Plot the recall vs. error curve for each tp metric.
    for metric in TP_METRICS:
        tp = metrics.get_label_tp(detection_name, metric)

        # Plot only if we have valid data.
        if tp is not np.nan and min_recall_ind <= md.max_recall_ind:
            recall, error = md.recall[:md.max_recall_ind + 1], getattr(md, metric)[:md.max_recall_ind + 1]
        else:
            recall, error = [], []

        # Change legend based on tp value
        if tp is np.nan:
            label = '{}: n/a'.format(PRETTY_TP_METRICS[metric])
        elif min_recall_ind > md.max_recall_ind:
            label = '{}: nan'.format(PRETTY_TP_METRICS[metric])
        else:
            label = '{}: {:.2f} ({})'.format(PRETTY_TP_METRICS[metric], tp, TP_METRICS_UNITS[metric])
        ax.plot(recall, error, label=label)
    ax.axvline(x=md.max_recall, linestyle='-.', color=(0, 0, 0, 0.3))
    ax.legend(loc='best')

    if savepath is not None:
        plt.savefig(savepath)
        plt.close()


def dist_pr_curve(md_list: DetectionMetricDataList,
                  metrics: DetectionMetrics,
                  dist_th: float,
                  min_precision: float,
                  min_recall: float,
                  savepath: str = None) -> None:
    """
    Plot the PR curves for different distance thresholds.
    :param md_list: DetectionMetricDataList instance.
    :param metrics: DetectionMetrics instance.
    :param dist_th: Distance threshold for matching.
    :param min_precision: Minimum precision value.
    :param min_recall: Minimum recall value.
    :param savepath: If given, saves the the rendering here instead of displaying.
    """
    # Prepare axis.
    fig, (ax, lax) = plt.subplots(ncols=2, gridspec_kw={"width_ratios": [4, 1]},
                                  figsize=(7.5, 5))
    ax = setup_axis(xlabel='Recall', ylabel='Precision',
                    xlim=1, ylim=1, min_precision=min_precision, min_recall=min_recall, ax=ax)

    # Plot the recall vs. precision curve for each detection class.
    data = md_list.get_dist_data(dist_th)
    for md, detection_name in data:
        md = md_list[(detection_name, dist_th)]
        ap = metrics.get_label_ap(detection_name, dist_th)
        ax.plot(md.recall, md.precision, label='{}: {:.1f}%'.format(PRETTY_DETECTION_NAMES[detection_name], ap * 100),
                color=DETECTION_COLORS[detection_name])
    hx, lx = ax.get_legend_handles_labels()
    lax.legend(hx, lx, borderaxespad=0)
    lax.axis("off")
    plt.tight_layout()
    if savepath is not None:
        plt.savefig(savepath)
        plt.close()


def summary_plot(md_list: DetectionMetricDataList,
                 metrics: DetectionMetrics,
                 min_precision: float,
                 min_recall: float,
                 dist_th_tp: float,
                 savepath: str = None,
                 cohort_analysis: bool = False) -> None:
    """
    Creates a summary plot with PR and TP curves for each class.
    :param md_list: DetectionMetricDataList instance.
    :param metrics: DetectionMetrics instance.
    :param min_precision: Minimum precision value.
    :param min_recall: Minimum recall value.
    :param dist_th_tp: The distance threshold used to determine matches.
    :param savepath: If given, saves the the rendering here instead of displaying.
    """
    n_classes = len(getDetectionNames(cohort_analysis))
    _, axes = plt.subplots(nrows=n_classes, ncols=2, figsize=(15, 5 * n_classes))
    for ind, detection_name in enumerate(getDetectionNames(cohort_analysis)):
        title1, title2 = ('Recall vs Precision', 'Recall vs Error') if ind == 0 else (None, None)
        try:
            ax1 = setup_axis(xlim=1, ylim=1, title=title1, min_precision=min_precision,
                         min_recall=min_recall, ax=axes[ind])
        except:
            ax1 = setup_axis(xlim=1, ylim=1, title=title1, min_precision=min_precision,
                         min_recall=min_recall, ax=axes[ind, 0])

        ax1.set_ylabel('{} \n \n Precision'.format(PRETTY_DETECTION_NAMES[detection_name]), size=20)

        try:
            ax2 = setup_axis(xlim=1, title=title2, min_recall=min_recall, ax=axes[ind])
        except:
            ax2 = setup_axis(xlim=1, title=title2, min_recall=min_recall, ax=axes[ind, 1])

        if ind == n_classes - 1:
            ax1.set_xlabel('Recall', size=20)
            ax2.set_xlabel('Recall', size=20)

        class_pr_curve(md_list, metrics, detection_name, min_precision, min_recall, ax=ax1)
        class_tp_curve(md_list, metrics, detection_name,  min_recall, dist_th_tp=dist_th_tp, ax=ax2)

    plt.tight_layout()

    if savepath is not None:
        plt.savefig(savepath)
        plt.close()

