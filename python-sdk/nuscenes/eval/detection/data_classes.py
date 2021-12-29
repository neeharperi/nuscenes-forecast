# nuScenes dev-kit.
# Code written by Oscar Beijbom, 2019.

from collections import defaultdict
from typing import List, Dict, Tuple

import numpy as np

from nuscenes.eval.common.data_classes import MetricData, EvalBox
from nuscenes.eval.common.utils import center_distance, miss_rate
from nuscenes.eval.detection.constants import ATTRIBUTE_NAMES, TP_METRICS
import pdb

class DetectionConfig:
    """ Data class that specifies the detection evaluation settings. """

    def __init__(self,
                 class_range: Dict[str, int],
                 dist_fcn: str,
                 dist_ths: List[float],
                 dist_th_tp: float,
                 forecast_th : int,
                 min_recall: float,
                 min_precision: float,
                 max_boxes_per_sample: float,
                 mean_ap_weight: int):

        #assert set(class_range.keys()) == set(DETECTION_NAMES), "Class count mismatch."
        assert dist_th_tp in dist_ths, "dist_th_tp must be in set of dist_ths."

        self.class_range = class_range
        self.dist_fcn = dist_fcn
        self.dist_ths = dist_ths
        self.dist_th_tp = dist_th_tp
        self.forecast_th = forecast_th
        self.min_recall = min_recall
        self.min_precision = min_precision
        self.max_boxes_per_sample = max_boxes_per_sample
        self.mean_ap_weight = mean_ap_weight

        self.class_names = self.class_range.keys()

    def __eq__(self, other):
        eq = True
        for key in self.serialize().keys():
            eq = eq and np.array_equal(getattr(self, key), getattr(other, key))
        return eq

    def serialize(self) -> dict:
        """ Serialize instance into json-friendly format. """
        return {
            'class_range': self.class_range,
            'dist_fcn': self.dist_fcn,
            'dist_ths': self.dist_ths,
            'dist_th_tp': self.dist_th_tp,
            'forecast_th': self.forecast_th,
            'min_recall': self.min_recall,
            'min_precision': self.min_precision,
            'max_boxes_per_sample': self.max_boxes_per_sample,
            'mean_ap_weight': self.mean_ap_weight
        }

    @classmethod
    def deserialize(cls, content: dict):
        """ Initialize from serialized dictionary. """
        return cls(content['class_range'],
                   content['dist_fcn'],
                   content['dist_ths'],
                   content['dist_th_tp'],
                   content['forecast_th'],
                   content['min_recall'],
                   content['min_precision'],
                   content['max_boxes_per_sample'],
                   content['mean_ap_weight'])

    @property
    def dist_fcn_callable(self):
        """ Return the distance function corresponding to the dist_fcn string. """
        if self.dist_fcn == 'center_distance':
            return center_distance
        else:
            raise Exception('Error: Unknown distance function %s!' % self.dist_fcn)


class DetectionMetricData(MetricData):
    """ This class holds accumulated and interpolated data required to calculate the detection metrics. """

    nelem = 101

    def __init__(self,
                 recall: np.array,
                 precision: np.array,
                 precision_mr,
                 forecast_precision,
                 confidence: np.array,
                 trans_err: np.array,
                 vel_err: np.array,
                 scale_err: np.array,
                 orient_err: np.array,
                 attr_err: np.array,
                 avg_disp_err: np.array,
                 final_disp_err: np.array,
                 miss_rate: np.array,
                 rec_val = None,
                 rec_fval = None,
                 #reverse_avg_disp_err: np.array,
                 #reverse_final_disp_err: np.array,
                 #reverse_miss_rate: np.array
                 ):

        # Assert lengths.
        assert len(recall) == self.nelem
        assert len(precision) == self.nelem
        assert len(confidence) == self.nelem
        assert len(trans_err) == self.nelem
        assert len(vel_err) == self.nelem
        assert len(scale_err) == self.nelem
        assert len(orient_err) == self.nelem
        assert len(attr_err) == self.nelem
        assert len(avg_disp_err) == self.nelem
        assert len(final_disp_err) == self.nelem
        assert len(miss_rate) == self.nelem
        #assert len(reverse_avg_disp_err) == self.nelem
        #assert len(reverse_final_disp_err) == self.nelem
        #assert len(reverse_miss_rate) == self.nelem

        # Assert ordering.
        assert all(confidence == sorted(confidence, reverse=True))  # Confidences should be descending.
        assert all(recall == sorted(recall))  # Recalls should be ascending.

        # Set attributes explicitly to help IDEs figure out what is going on.
        self.recall = recall
        self.precision = precision
        self.precision_mr = precision_mr
        self.forecast_precision = forecast_precision
        self.confidence = confidence
        self.trans_err = trans_err
        self.vel_err = vel_err
        self.scale_err = scale_err
        self.orient_err = orient_err
        self.attr_err = attr_err
        self.avg_disp_err = avg_disp_err
        self.final_disp_err = final_disp_err
        self.miss_rate = miss_rate
        self.rec_val = rec_val
        self.rec_fval = rec_fval
        #self.reverse_avg_disp_err = reverse_avg_disp_err
        #self.reverse_final_disp_err = reverse_final_disp_err
        #self.reverse_miss_rate = reverse_miss_rate

    def __eq__(self, other):
        eq = True
        for key in self.serialize().keys():
            eq = eq and np.array_equal(getattr(self, key), getattr(other, key))
        return eq

    @property
    def max_recall_ind(self):
        """ Returns index of max recall achieved. """

        # Last instance of confidence > 0 is index of max achieved recall.
        non_zero = np.nonzero(self.confidence)[0]
        if len(non_zero) == 0:  # If there are no matches, all the confidence values will be zero.
            max_recall_ind = 0
        else:
            max_recall_ind = non_zero[-1]

        return max_recall_ind

    @property
    def max_recall(self):
        """ Returns max recall achieved. """

        return self.recall[self.max_recall_ind]

    def serialize(self):
        """ Serialize instance into json-friendly format. """
        return {
            'recall': self.recall.tolist(),
            'precision': self.precision.tolist(),
            'precision_mr': self.precision_mr.tolist(),
            'forecast_precision': [self.forecast_precision[i].tolist() for i in range(len(self.forecast_precision))],
            'confidence': self.confidence.tolist(),
            'trans_err': self.trans_err.tolist(),
            'vel_err': self.vel_err.tolist(),
            'scale_err': self.scale_err.tolist(),
            'orient_err': self.orient_err.tolist(),
            'attr_err': self.attr_err.tolist(),
            'avg_disp_err' : self.avg_disp_err.tolist(),
            'final_disp_err' : self.final_disp_err.tolist(),
            'miss_rate' : self.miss_rate.tolist(),
            #'rec_val' : self.rec_val.tolist(),
            #'rec_fval' : [self.rec_fval[i].tolist() for i in range(len(self.rec_fval))],

            #'reverse_avg_disp_err' : self.reverse_avg_disp_err.tolist(),
            #'reverse_final_disp_err' : self.reverse_final_disp_err.tolist(),
            #'reverse_miss_rate' : self.reverse_miss_rate.tolist(),
        }

    @classmethod
    def deserialize(cls, content: dict):
        """ Initialize from serialized content. """
        return cls(recall=np.array(content['recall']),
                   precision=np.array(content['precision']),
                   precision_mr=np.array(content['precision_mr']),
                   forecast_precision=[np.array(content['forecast_precision'][i]) for i in range(len(content['forecast_precision']))],
                   confidence=np.array(content['confidence']),
                   trans_err=np.array(content['trans_err']),
                   vel_err=np.array(content['vel_err']),
                   scale_err=np.array(content['scale_err']),
                   orient_err=np.array(content['orient_err']),
                   attr_err=np.array(content['attr_err']),
                   avg_disp_err=np.array(content['avg_disp_err']),
                   final_disp_err=np.array(content['final_disp_err']),
                   miss_rate=np.array(content['miss_rate']),
                   #rec_val=np.array(content['rec_val']),
                   #rec_fval=[np.array(content['rec_fval'][i]) for i in range(len(content['rec_fval']))],
                   #reverse_avg_disp_err=np.array(content['reverse_avg_disp_err']),
                   #reverse_final_disp_err=np.array(content['reverse_final_disp_err']),
                   #reverse_miss_rate=np.array(content['reverse_miss_rate']),
                   )

    @classmethod
    def no_predictions(cls, timesteps=7):
        """ Returns a md instance corresponding to having no predictions. """
        return cls(recall=np.linspace(0, 1, cls.nelem),
                   precision=np.zeros(cls.nelem),
                   precision_mr=np.zeros(cls.nelem),
                   forecast_precision=[np.zeros(cls.nelem) for i in range(timesteps)],
                   confidence=np.zeros(cls.nelem),
                   trans_err=np.ones(cls.nelem),
                   vel_err=np.ones(cls.nelem),
                   scale_err=np.ones(cls.nelem),
                   orient_err=np.ones(cls.nelem),
                   attr_err=np.ones(cls.nelem),
                   avg_disp_err=np.ones(cls.nelem),
                   final_disp_err=np.ones(cls.nelem),
                   miss_rate=np.ones(cls.nelem),
                   #rec_val=None,
                   #rec_fval=[None for i in range(timesteps)],
                   #reverse_avg_disp_err=np.ones(cls.nelem),
                   #reverse_final_disp_err=np.ones(cls.nelem),
                   #reverse_miss_rate=np.ones(cls.nelem),
                   )

    @classmethod
    def random_md(cls, timesteps=7):
        """ Returns an md instance corresponding to a random results. """
        return cls(recall=np.linspace(0, 1, cls.nelem),
                   precision=np.random.random(cls.nelem),
                   precision_mr=np.random.random(cls.nelem),
                   forecast_precision=[np.random.random(cls.nelem) for i in range(timesteps)],
                   confidence=np.linspace(0, 1, cls.nelem)[::-1],
                   trans_err=np.random.random(cls.nelem),
                   vel_err=np.random.random(cls.nelem),
                   scale_err=np.random.random(cls.nelem),
                   orient_err=np.random.random(cls.nelem),
                   attr_err=np.random.random(cls.nelem),
                   avg_disp_err=np.random.random(cls.nelem),
                   final_disp_err=np.random.random(cls.nelem),
                   miss_rate=np.random.random(cls.nelem),
                   #rec_val=np.random.random(cls.nelem),
                   #rec_fval=[np.random.random(cls.nelem) for i in range(timesteps)],
                   #reverse_avg_disp_err=np.random.random(cls.nelem),
                   #reverse_final_disp_err=np.random.random(cls.nelem),
                   #reverse_miss_rate=np.random.random(cls.nelem),
                   )


class DetectionMetrics:
    """ Stores average precision and true positive metric results. Provides properties to summarize. """

    def __init__(self, cfg: DetectionConfig):

        self.cfg = cfg
        self._label_aps = defaultdict(lambda: defaultdict(float))
        self._label_faps_mr = defaultdict(lambda: defaultdict(float))
        self._label_ars = defaultdict(lambda: defaultdict(float))
        self._label_faps = defaultdict(lambda: defaultdict(float))
        self._label_fars = defaultdict(lambda: defaultdict(float))
        self._label_aaps = defaultdict(lambda: defaultdict(float))
        self._label_aars = defaultdict(lambda: defaultdict(float))
        self._label_tp_errors = defaultdict(lambda: defaultdict(float))
        self.eval_time = None

    def add_label_ap(self, detection_name: str, dist_th: float, ap: float) -> None:
        self._label_aps[detection_name][dist_th] = ap

    def get_label_ap(self, detection_name: str, dist_th: float) -> float:
        return self._label_aps[detection_name][dist_th]

    def add_label_fap_mr(self, detection_name: str, dist_th: float, fap_mr: float) -> None:
        self._label_faps_mr[detection_name][dist_th] = fap_mr

    def get_label_fap_mr(self, detection_name: str, dist_th: float) -> float:
        return self._label_faps_mr[detection_name][dist_th]

    def add_label_ar(self, detection_name: str, dist_th: float, ar: float) -> None:
        self._label_ars[detection_name][dist_th] = ar
    
    def get_label_ar(self, detection_name: str, dist_th: float) -> float:
        return self._label_ars[detection_name][dist_th]

    def add_label_fap(self, detection_name: str, dist_th: float, fap: float) -> None:
        self._label_faps[detection_name][dist_th] = fap

    def get_label_fap(self, detection_name: str, dist_th: float) -> float:
        return self._label_faps[detection_name][dist_th]

    def add_label_far(self, detection_name: str, dist_th: float, far: float) -> None:
        self._label_fars[detection_name][dist_th] = far

    def get_label_far(self, detection_name: str, dist_th: float) -> float:
        return self._label_fars[detection_name][dist_th]
    
    def add_label_aap(self, detection_name: str, dist_th: float, aap: float) -> None:
        self._label_aaps[detection_name][dist_th] = aap

    def get_label_aap(self, detection_name: str, dist_th: float) -> float:
        return self._label_aaps[detection_name][dist_th]

    def add_label_aar(self, detection_name: str, dist_th: float, aar: float) -> None:
        self._label_aars[detection_name][dist_th] = aar

    def get_label_aar(self, detection_name: str, dist_th: float) -> float:
        return self._label_aars[detection_name][dist_th]
    
    def add_label_tp(self, detection_name: str, metric_name: str, tp: float):
        self._label_tp_errors[detection_name][metric_name] = tp

    def get_label_tp(self, detection_name: str, metric_name: str) -> float:
        return self._label_tp_errors[detection_name][metric_name]

    def add_runtime(self, eval_time: float) -> None:
        self.eval_time = eval_time

    @property
    def mean_dist_aps(self) -> Dict[str, float]:
        """ Calculates the mean over distance thresholds for each label. """
        return {class_name: np.mean(list(d.values())) for class_name, d in self._label_aps.items()}

    @property
    def mean_dist_faps_mr(self) -> Dict[str, float]:
        """ Calculates the mean over distance thresholds for each label. """

        return {class_name: np.mean(list(d.values())) for class_name, d in self._label_faps_mr.items()}
    
    @property
    def mean_dist_ars(self) -> Dict[str, float]:
        """ Calculates the mean over distance thresholds for each label. """
        return {class_name: np.mean(list(d.values())) for class_name, d in self._label_ars.items()}

    @property
    def mean_dist_faps(self) -> Dict[str, float]:
        """ Calculates the mean over distance thresholds for each label. """
        return {class_name: np.mean(list(d.values())) for class_name, d in self._label_faps.items()}

    @property
    def mean_dist_fars(self) -> Dict[str, float]:
        """ Calculates the mean over distance thresholds for each label. """
        return {class_name: np.mean(list(d.values())) for class_name, d in self._label_fars.items()}

    @property
    def mean_dist_aaps(self) -> Dict[str, float]:
        """ Calculates the mean over distance thresholds for each label. """
        return {class_name: np.mean(list(d.values())) for class_name, d in self._label_aaps.items()}

    @property
    def mean_dist_aars(self) -> Dict[str, float]:
        """ Calculates the mean over distance thresholds for each label. """
        return {class_name: np.mean(list(d.values())) for class_name, d in self._label_aars.items()}


    @property
    def mean_ap(self) -> float:
        """ Calculates the mean AP by averaging over distance thresholds and classes. """
        return float(np.mean(list(self.mean_dist_aps.values())))

    @property
    def mean_fap_mr(self) -> float:
        """ Calculates the mean AP by averaging over distance thresholds and classes. """
        return float(np.mean(list(self.mean_dist_faps_mr.values())))

    @property
    def mean_ar(self) -> float:
        """ Calculates the mean AR by averaging over distance thresholds and classes. """
        return float(np.mean(list(self.mean_dist_ars.values())))

    @property
    def mean_fap(self) -> float:
        """ Calculates the mean FAP by averaging over distance thresholds and classes. """
        return float(np.mean(list(self.mean_dist_faps.values())))

    @property
    def mean_far(self) -> float:
        """ Calculates the mean FAR by averaging over distance thresholds and classes. """
        return float(np.mean(list(self.mean_dist_fars.values())))

    @property
    def mean_aap(self) -> float:
        """ Calculates the mean FAP by averaging over distance thresholds and classes. """
        return float(np.mean(list(self.mean_dist_aaps.values())))

    @property
    def mean_aar(self) -> float:
        """ Calculates the mean FAR by averaging over distance thresholds and classes. """
        return float(np.mean(list(self.mean_dist_aars.values())))

    @property
    def tp_errors(self) -> Dict[str, float]:
        """ Calculates the mean true positive error across all classes for each metric. """
        errors = {}
        for metric_name in TP_METRICS:
            class_errors = []
            for detection_name in self.cfg.class_names:
                class_errors.append(self.get_label_tp(detection_name, metric_name))

            errors[metric_name] = float(np.nanmean(class_errors))

        return errors

    @property
    def tp_scores(self) -> Dict[str, float]:
        scores = {}
        tp_errors = self.tp_errors
        for metric_name in TP_METRICS:

            # We convert the true positive errors to "scores" by 1-error.
            score = 1.0 - tp_errors[metric_name]

            # Some of the true positive errors are unbounded, so we bound the scores to min 0.
            score = max(0.0, score)

            scores[metric_name] = score

        return scores

    @property
    def nd_score(self) -> float:
        """
        Compute the nuScenes detection score (NDS, weighted sum of the individual scores).
        :return: The NDS.
        """
        # Summarize.
        total = float(self.cfg.mean_ap_weight * self.mean_ap + np.sum(list(self.tp_scores.values())))

        # Normalize.
        total = total / float(self.cfg.mean_ap_weight + len(self.tp_scores.keys()))

        return total

    def serialize(self):
        return {
            'label_aps': self._label_aps,
            'mean_dist_aps': self.mean_dist_aps,
            'mean_ap': self.mean_ap,
            'label_faps_mr': self._label_faps_mr,
            'mean_dist_faps_mr': self.mean_dist_faps_mr,
            'mean_fap_mr': self.mean_fap_mr,
            'label_ars': self._label_ars,
            'mean_dist_ars': self.mean_dist_ars,
            'mean_ar': self.mean_ar,
            'label_faps': self._label_faps,
            'mean_dist_faps': self.mean_dist_faps,
            'mean_fap': self.mean_fap,
            'label_fars': self._label_fars,
            'mean_dist_fars': self.mean_dist_fars,
            'mean_far': self.mean_far,
            'label_aaps': self._label_aaps,
            'mean_dist_aaps': self.mean_dist_aaps,
            'mean_aap': self.mean_aap,
            'label_aars': self._label_aars,
            'mean_dist_aars': self.mean_dist_aars,
            'mean_aar': self.mean_aar,
            'label_tp_errors': self._label_tp_errors,
            'tp_errors': self.tp_errors,
            'tp_scores': self.tp_scores,
            'nd_score': self.nd_score,
            'eval_time': self.eval_time,
            'cfg': self.cfg.serialize()
        }

    @classmethod
    def deserialize(cls, content: dict):
        """ Initialize from serialized dictionary. """

        cfg = DetectionConfig.deserialize(content['cfg'])

        metrics = cls(cfg=cfg)
        metrics.add_runtime(content['eval_time'])

        for detection_name, label_aps in content['label_aps'].items():
            for dist_th, ap in label_aps.items():
                metrics.add_label_ap(detection_name=detection_name, dist_th=float(dist_th), ap=float(ap))

        for detection_name, label_faps_mr in content['label_faps_mr'].items():
            for dist_th, fap_mr in label_faps_mr.items():
                metrics.add_label_fap_mr(detection_name=detection_name, dist_th=float(dist_th), fap_mr=float(fap_mr))

        for detection_name, label_ars in content['label_ars'].items():
            for dist_th, ar in label_ars.items():
                metrics.add_label_ar(detection_name=detection_name, dist_th=float(dist_th), ar=float(ar))

        for detection_name, label_faps in content['label_faps'].items():
            for dist_th, fap in label_faps.items():
                metrics.add_label_fap(detection_name=detection_name, dist_th=float(dist_th), fap=float(fap))

        for detection_name, label_fars in content['label_fars'].items():
            for dist_th, far in label_fars.items():
                metrics.add_label_far(detection_name=detection_name, dist_th=float(dist_th), far=float(far))

        for detection_name, label_aaps in content['label_aaps'].items():
            for dist_th, aap in label_aaps.items():
                metrics.add_label_aap(detection_name=detection_name, dist_th=float(dist_th), aap=float(aap))

        for detection_name, label_aars in content['label_aars'].items():
            for dist_th, aar in label_aars.items():
                metrics.add_label_aar(detection_name=detection_name, dist_th=float(dist_th), aar=float(aar))

        for detection_name, label_tps in content['label_tp_errors'].items():
            for metric_name, tp in label_tps.items():
                metrics.add_label_tp(detection_name=detection_name, metric_name=metric_name, tp=float(tp))

        return metrics

    def __eq__(self, other):
        eq = True
        eq = eq and self._label_aps == other._label_aps
        eq = eq and self._label_faps_mr == other._label_faps_mr
        eq = eq and self._label_ars == other._label_ars
        eq = eq and self._label_faps == other._label_faps
        eq = eq and self._label_fars == other._label_fars
        eq = eq and self._label_aaps == other._label_aaps
        eq = eq and self._label_aars == other._label_aars
        eq = eq and self._label_tp_errors == other._label_tp_errors
        eq = eq and self.eval_time == other.eval_time
        eq = eq and self.cfg == other.cfg

        return eq

class DetectionBox(EvalBox):
    """ Data class used during detection evaluation. Can be a prediction or ground truth."""

    def __init__(self,
                 sample_token: str = "",
                 translation: Tuple[float, float, float] = (0, 0, 0),
                 size: Tuple[float, float, float] = (0, 0, 0),
                 rotation: Tuple[float, float, float, float] = (0, 0, 0, 0),
                 velocity: Tuple[float, float] = (0, 0),
                 forecast_boxes: list = [],
                 ego_translation: [float, float, float] = (0, 0, 0),  # Translation to ego vehicle in meters.
                 num_pts: int = -1,  # Nbr. LIDAR or RADAR inside the box. Only for gt boxes.
                 detection_name: str = 'car',  # The class name used in the detection challenge.
                 detection_score: float = -1.0,  # GT samples do not have a score.
                 forecast_score: float = -1.0,
                 forecast_id: int = -1,
                 attribute_name: str = ''):  # Box attribute. Each box can have at most 1 attribute.

        super().__init__(sample_token, translation, size, rotation, velocity, ego_translation, num_pts)

        assert detection_name is not None, 'Error: detection_name cannot be empty!'
        #assert detection_name in DETECTION_NAMES, 'Error: Unknown detection_name %s' % detection_name

        assert attribute_name in ATTRIBUTE_NAMES or attribute_name == '', \
            'Error: Unknown attribute_name %s' % attribute_name

        assert type(detection_score) == float, 'Error: detection_score must be a float!'
        assert not np.any(np.isnan(detection_score)), 'Error: detection_score may not be NaN!'

        # Assign.
        self.forecast_boxes = forecast_boxes
        self.detection_name = detection_name
        self.detection_score = detection_score
        self.forecast_score = forecast_score
        self.forecast_id = forecast_id
        self.attribute_name = attribute_name

    def __eq__(self, other):
        return (self.sample_token == other.sample_token and
                self.translation == other.translation and
                self.size == other.size and
                self.rotation == other.rotation and
                self.velocity == other.velocity and
                self.ego_translation == other.ego_translation and
                self.num_pts == other.num_pts and
                self.detection_name == other.detection_name and
                self.detection_score == other.detection_score and
                self.attribute_name == other.attribute_name)

    def serialize(self) -> dict:
        """ Serialize instance into json-friendly format. """
        return {
            'sample_token': self.sample_token,
            'translation': self.translation,
            'size': self.size,
            'rotation': self.rotation,
            'velocity': self.velocity,
            'forecast_boxes': self.forecast_boxes,
            'ego_translation': self.ego_translation,
            'num_pts': self.num_pts,
            'detection_name': self.detection_name,
            'detection_score': self.detection_score,
            'forecast_score': self.forecast_score,
            'forecast_id': self.forecast_id,
            'attribute_name': self.attribute_name
        }

    @classmethod
    def deserialize(cls, content: dict):
        """ Initialize from serialized content. """
        return cls(sample_token=content['sample_token'],
                   translation=tuple(content['translation']),
                   size=tuple(content['size']),
                   rotation=tuple(content['rotation']),
                   velocity=tuple(content['velocity']),
                   forecast_boxes=content['forecast_boxes'],
                   ego_translation=(0.0, 0.0, 0.0) if 'ego_translation' not in content
                   else tuple(content['ego_translation']),
                   num_pts=-1 if 'num_pts' not in content else int(content['num_pts']),
                   detection_name=content['detection_name'],
                   detection_score=-1.0 if 'detection_score' not in content else float(content['detection_score']),
                   forecast_score=-1.0 if 'forecast_score' not in content else float(content['forecast_score']),
                   forecast_id=-1 if 'forecast_id' not in content else int(content['forecast_id']),
                   attribute_name=content['attribute_name'])


class DetectionMetricDataList:
    """ This stores a set of MetricData in a dict indexed by (name, match-distance). """

    def __init__(self):
        self.md = {}

    def __getitem__(self, key):
        return self.md[key]

    def __eq__(self, other):
        eq = True
        for key in self.md.keys():
            eq = eq and self[key] == other[key]
        return eq

    def get_class_data(self, detection_name: str) -> List[Tuple[DetectionMetricData, float]]:
        """ Get all the MetricData entries for a certain detection_name. """
        return [(md, dist_th) for (name, dist_th), md in self.md.items() if name == detection_name]

    def get_dist_data(self, dist_th: float) -> List[Tuple[DetectionMetricData, str]]:
        """ Get all the MetricData entries for a certain match_distance. """
        return [(md, detection_name) for (detection_name, dist), md in self.md.items() if dist == dist_th]

    def set(self, detection_name: str, match_distance: float, data: DetectionMetricData):
        """ Sets the MetricData entry for a certain detection_name and match_distance. """
        self.md[(detection_name, match_distance)] = data

    def serialize(self) -> dict:
        return {key[0] + ':' + str(key[1]): value.serialize() for key, value in self.md.items()}

    @classmethod
    def deserialize(cls, content: dict):
        mdl = cls()
        for key, md in content.items():
            name, distance = key.split(':')
            mdl.set(name, float(distance), DetectionMetricData.deserialize(md))
        return mdl
