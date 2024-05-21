import warnings

from time import time
import numpy as np


# ------------------------------------------------------------------------------
# Adapted from https://github.com/HoBeom/OneEuroFilter-Numpy
# Original licence: Copyright (c)  HoBeom Jeon, under the MIT License.
# ------------------------------------------------------------------------------

def smoothing_factor(t_e, cutoff):
    r = 2 * np.pi * cutoff * t_e
    return r / (r + 1)


def exponential_smoothing(a, x, x_prev):
    return a * x + (1 - a) * x_prev


class OneEuroFilter:
    def __init__(self,
                 x0,
                 dx0=0.0,
                 min_cutoff=1.7,
                 beta=0.3,
                 d_cutoff=30.0,
                 fps=None):
        """One Euro Filter for keypoints smoothing.

        Args:
            x0 (np.ndarray[K, 2]): Initialize keypoints value
            dx0 (float): 0.0
            min_cutoff (float): parameter for one euro filter
            beta (float): parameter for one euro filter
            d_cutoff (float): Input data FPS
            fps (float): Video FPS for video inference
        """
        warnings.warn(
            'OneEuroFilter from '
            '`mmpose/core/post_processing/one_euro_filter.py` will '
            'be deprecated in the future. Please use Smoother'
            '(`mmpose/core/post_processing/smoother.py`) with '
            'OneEuroFilter (`mmpose/core/post_processing/temporal_'
            'filters/one_euro_filter.py`).', DeprecationWarning)

        # The parameters.
        self.data_shape = x0.shape
        self.min_cutoff = np.full(x0.shape, min_cutoff)
        self.beta = np.full(x0.shape, beta)
        self.d_cutoff = np.full(x0.shape, d_cutoff)
        # Previous values.
        self.x_prev = x0.astype(np.float32)
        self.dx_prev = np.full(x0.shape, dx0)
        self.mask_prev = np.ma.masked_where(x0 <= 0, x0)
        self.realtime = True
        if fps is None:
            # Using in realtime inference
            self.t_e = None
            self.skip_frame_factor = d_cutoff
            self.fps = d_cutoff
        else:
            # fps using video inference
            self.realtime = False
            self.fps = float(fps)
            self.d_cutoff = np.full(x0.shape, self.fps)

        self.t_prev = time()

    def __call__(self, x, t_e=1.0):
        """Compute the filtered signal.

        Hyper-parameters (cutoff, beta) are from `VNect
        <http://gvv.mpi-inf.mpg.de/projects/VNect/>`__ .

        Realtime Camera fps (d_cutoff) default 30.0

        Args:
            x (np.ndarray[K, 2]): keypoints results in frame
            t_e (Optional): video skip frame count for posetrack
                evaluation
        """
        assert x.shape == self.data_shape

        t = 0
        if self.realtime:
            t = time()
            t_e = (t - self.t_prev) * self.skip_frame_factor
        t_e = np.full(x.shape, t_e)

        # missing keypoints mask
        mask = np.ma.masked_where(x <= 0, x)

        # The filtered derivative of the signal.
        a_d = smoothing_factor(t_e / self.fps, self.d_cutoff)
        dx = (x - self.x_prev) / t_e
        dx_hat = exponential_smoothing(a_d, dx, self.dx_prev)

        # The filtered signal.
        cutoff = self.min_cutoff + self.beta * np.abs(dx_hat)
        a = smoothing_factor(t_e / self.fps, cutoff)
        x_hat = exponential_smoothing(a, x, self.x_prev)

        # missing keypoints remove
        np.copyto(x_hat, -10, where=mask.mask)

        # Memorize the previous values.
        self.x_prev = x_hat
        self.dx_prev = dx_hat
        self.t_prev = t
        self.mask_prev = mask

        return x_hat


# ------------------------------------------------------------------------------
# Adapted from https://github.com/leoxiaobin/deep-high-resolution-net.pytorch
# Original licence: Copyright (c) Microsoft, under the MIT License.
# ------------------------------------------------------------------------------

def nms(dets, thr):
    """Greedily select boxes with high confidence and overlap <= thr.

    Args:
        dets: [[x1, y1, x2, y2, score]].
        thr: Retain overlap < thr.

    Returns:
         list: Indexes to keep.
    """
    if len(dets) == 0:
        return []

    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while len(order) > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thr)[0]
        order = order[inds + 1]

    return keep


def oks_iou(g, d, a_g, a_d, sigmas=None, vis_thr=None):
    """Calculate oks ious.

    Args:
        g: Ground truth keypoints.
        d: Detected keypoints.
        a_g: Area of the ground truth object.
        a_d: Area of the detected object.
        sigmas: standard deviation of keypoint labelling.
        vis_thr: threshold of the keypoint visibility.

    Returns:
        list: The oks ious.
    """
    if sigmas is None:
        sigmas = np.array([
            .26, .25, .25, .35, .35, .79, .79, .72, .72, .62, .62, 1.07, 1.07,
            .87, .87, .89, .89
        ]) / 10.0
    vars = (sigmas * 2)**2
    xg = g[0::3]
    yg = g[1::3]
    vg = g[2::3]
    ious = np.zeros(len(d), dtype=np.float32)
    for n_d in range(0, len(d)):
        xd = d[n_d, 0::3]
        yd = d[n_d, 1::3]
        vd = d[n_d, 2::3]
        dx = xd - xg
        dy = yd - yg
        e = (dx**2 + dy**2) / vars / ((a_g + a_d[n_d]) / 2 + np.spacing(1)) / 2
        if vis_thr is not None:
            ind = list(vg > vis_thr) and list(vd > vis_thr)
            e = e[ind]
        ious[n_d] = np.sum(np.exp(-e)) / len(e) if len(e) != 0 else 0.0
    return ious


def oks_nms(kpts_db, thr, sigmas=None, vis_thr=None, score_per_joint=False):
    """OKS NMS implementations.

    Args:
        kpts_db: keypoints.
        thr: Retain overlap < thr.
        sigmas: standard deviation of keypoint labelling.
        vis_thr: threshold of the keypoint visibility.
        score_per_joint: the input scores (in kpts_db) are per joint scores

    Returns:
        np.ndarray: indexes to keep.
    """
    if len(kpts_db) == 0:
        return []

    if score_per_joint:
        scores = np.array([k['score'].mean() for k in kpts_db])
    else:
        scores = np.array([k['score'] for k in kpts_db])

    kpts = np.array([k['keypoints'].flatten() for k in kpts_db])
    areas = np.array([k['area'] for k in kpts_db])

    order = scores.argsort()[::-1]

    keep = []
    while len(order) > 0:
        i = order[0]
        keep.append(i)

        oks_ovr = oks_iou(kpts[i], kpts[order[1:]], areas[i], areas[order[1:]],
                          sigmas, vis_thr)

        inds = np.where(oks_ovr <= thr)[0]
        order = order[inds + 1]

    keep = np.array(keep)

    return keep


def _rescore(overlap, scores, thr, type='gaussian'):
    """Rescoring mechanism gaussian or linear.

    Args:
        overlap: calculated ious
        scores: target scores.
        thr: retain oks overlap < thr.
        type: 'gaussian' or 'linear'

    Returns:
        np.ndarray: indexes to keep
    """
    assert len(overlap) == len(scores)
    assert type in ['gaussian', 'linear']

    if type == 'linear':
        inds = np.where(overlap >= thr)[0]
        scores[inds] = scores[inds] * (1 - overlap[inds])
    else:
        scores = scores * np.exp(-overlap**2 / thr)

    return scores


def soft_oks_nms(kpts_db,
                 thr,
                 max_dets=20,
                 sigmas=None,
                 vis_thr=None,
                 score_per_joint=False):
    """Soft OKS NMS implementations.

    Args:
        kpts_db
        thr: retain oks overlap < thr.
        max_dets: max number of detections to keep.
        sigmas: Keypoint labelling uncertainty.
        score_per_joint: the input scores (in kpts_db) are per joint scores

    Returns:
        np.ndarray: indexes to keep.
    """
    if len(kpts_db) == 0:
        return []

    if score_per_joint:
        scores = np.array([k['score'].mean() for k in kpts_db])
    else:
        scores = np.array([k['score'] for k in kpts_db])

    kpts = np.array([k['keypoints'].flatten() for k in kpts_db])
    areas = np.array([k['area'] for k in kpts_db])

    order = scores.argsort()[::-1]
    scores = scores[order]

    keep = np.zeros(max_dets, dtype=np.intp)
    keep_cnt = 0
    while len(order) > 0 and keep_cnt < max_dets:
        i = order[0]

        oks_ovr = oks_iou(kpts[i], kpts[order[1:]], areas[i], areas[order[1:]],
                          sigmas, vis_thr)

        order = order[1:]
        scores = _rescore(oks_ovr, scores[1:], thr)

        tmp = scores.argsort()[::-1]
        order = order[tmp]
        scores = scores[tmp]

        keep[keep_cnt] = i
        keep_cnt += 1

    keep = keep[:keep_cnt]

    return keep


def _compute_iou(bboxA, bboxB):
    """Compute the Intersection over Union (IoU) between two boxes .

    Args:
        bboxA (list): The first bbox info (left, top, right, bottom, score).
        bboxB (list): The second bbox info (left, top, right, bottom, score).

    Returns:
        float: The IoU value.
    """

    x1 = max(bboxA[0], bboxB[0])
    y1 = max(bboxA[1], bboxB[1])
    x2 = min(bboxA[2], bboxB[2])
    y2 = min(bboxA[3], bboxB[3])

    inter_area = max(0, x2 - x1) * max(0, y2 - y1)

    bboxA_area = (bboxA[2] - bboxA[0]) * (bboxA[3] - bboxA[1])
    bboxB_area = (bboxB[2] - bboxB[0]) * (bboxB[3] - bboxB[1])
    union_area = float(bboxA_area + bboxB_area - inter_area)
    if union_area == 0:
        union_area = 1e-5
        warnings.warn('union_area=0 is unexpected')

    iou = inter_area / union_area

    return iou


def _track_by_iou(res, results_last, thr):
    """Get track id using IoU tracking greedily.

    Args:
        res (dict): The bbox & pose results of the person instance.
        results_last (list[dict]): The bbox & pose & track_id info of the
            last frame (bbox_result, pose_result, track_id).
        thr (float): The threshold for iou tracking.

    Returns:
        int: The track id for the new person instance.
        list[dict]: The bbox & pose & track_id info of the persons
            that have not been matched on the last frame.
        dict: The matched person instance on the last frame.
    """

    bbox = list(res['bbox'])

    max_iou_score = -1
    max_index = -1
    match_result = {}
    for index, res_last in enumerate(results_last):
        bbox_last = list(res_last['bbox'])

        iou_score = _compute_iou(bbox, bbox_last)
        if iou_score > max_iou_score:
            max_iou_score = iou_score
            max_index = index

    if max_iou_score > thr:
        track_id = results_last[max_index]['track_id']
        match_result = results_last[max_index]
        del results_last[max_index]
    else:
        track_id = -1

    return track_id, results_last, match_result


def _track_by_oks(res, results_last, thr):
    """Get track id using OKS tracking greedily.

    Args:
        res (dict): The pose results of the person instance.
        results_last (list[dict]): The pose & track_id info of the
            last frame (pose_result, track_id).
        thr (float): The threshold for oks tracking.

    Returns:
        int: The track id for the new person instance.
        list[dict]: The pose & track_id info of the persons
            that have not been matched on the last frame.
        dict: The matched person instance on the last frame.
    """
    pose = res['keypoints'].reshape((-1))
    area = res['area']
    max_index = -1
    match_result = {}

    if len(results_last) == 0:
        return -1, results_last, match_result

    pose_last = np.array(
        [res_last['keypoints'].reshape((-1)) for res_last in results_last])
    area_last = np.array([res_last['area'] for res_last in results_last])

    oks_score = oks_iou(pose, pose_last, area, area_last)

    max_index = np.argmax(oks_score)

    if oks_score[max_index] > thr:
        track_id = results_last[max_index]['track_id']
        match_result = results_last[max_index]
        del results_last[max_index]
    else:
        track_id = -1

    return track_id, results_last, match_result


def _get_area(results):
    """Get bbox for each person instance on the current frame.

    Args:
        results (list[dict]): The pose results of the current frame
            (pose_result).
    Returns:
        list[dict]: The bbox & pose info of the current frame
            (bbox_result, pose_result, area).
    """
    for result in results:
        if 'bbox' in result:
            result['area'] = ((result['bbox'][2] - result['bbox'][0]) *
                              (result['bbox'][3] - result['bbox'][1]))
        else:
            xmin = np.min(
                result['keypoints'][:, 0][result['keypoints'][:, 0] > 0],
                initial=1e10)
            xmax = np.max(result['keypoints'][:, 0])
            ymin = np.min(
                result['keypoints'][:, 1][result['keypoints'][:, 1] > 0],
                initial=1e10)
            ymax = np.max(result['keypoints'][:, 1])
            result['area'] = (xmax - xmin) * (ymax - ymin)
            result['bbox'] = np.array([xmin, ymin, xmax, ymax])
    return results


def _temporal_refine(result, match_result, fps=None):
    """Refine koypoints using tracked person instance on last frame.

    Args:
        results (dict): The pose results of the current frame
                (pose_result).
        match_result (dict): The pose results of the last frame
                (match_result)
    Returns:
        (array): The person keypoints after refine.
    """
    if 'one_euro' in match_result:
        result['keypoints'][:, :2] = match_result['one_euro'](
            result['keypoints'][:, :2])
        result['one_euro'] = match_result['one_euro']
    else:
        result['one_euro'] = OneEuroFilter(result['keypoints'][:, :2], fps=fps)
    return result['keypoints']


def get_track_id(results,
                 results_last,
                 next_id,
                 min_keypoints=3,
                 use_oks=False,
                 tracking_thr=0.3,
                 use_one_euro=False,
                 fps=None):
    """Get track id for each person instance on the current frame.

    Args:
        results (list[dict]): The bbox & pose results of the current frame
            (bbox_result, pose_result).
        results_last (list[dict], optional): The bbox & pose & track_id info
            of the last frame (bbox_result, pose_result, track_id). None is
            equivalent to an empty result list. Default: None
        next_id (int): The track id for the new person instance.
        min_keypoints (int): Minimum number of keypoints recognized as person.
            0 means no minimum threshold required. Default: 3.
        use_oks (bool): Flag to using oks tracking. default: False.
        tracking_thr (float): The threshold for tracking.
        use_one_euro (bool): Option to use one-euro-filter. default: False.
        fps (optional): Parameters that d_cutoff
            when one-euro-filter is used as a video input

    Returns:
        tuple:
        - results (list[dict]): The bbox & pose & track_id info of the \
            current frame (bbox_result, pose_result, track_id).
        - next_id (int): The track id for the new person instance.
    """
    if use_one_euro:
        warnings.warn(
            'In the future, get_track_id() will no longer perform '
            'temporal refinement and the arguments `use_one_euro` and '
            '`fps` will be deprecated. This part of function has been '
            'migrated to Smoother (mmpose.core.Smoother). See '
            'demo/top_down_pose_trackign_demo_with_mmdet.py for an '
            'example.', DeprecationWarning)

    if results_last is None:
        results_last = []

    results = _get_area(results)

    if use_oks:
        _track = _track_by_oks
    else:
        _track = _track_by_iou

    for result in results:
        track_id, results_last, match_result = _track(result, results_last,
                                                      tracking_thr)
        if track_id == -1:
            if np.count_nonzero(result['keypoints'][:, 1]) >= min_keypoints:
                result['track_id'] = next_id
                next_id += 1
            else:
                # If the number of keypoints detected is small,
                # delete that person instance.
                result['keypoints'][:, 1] = -10
                result['bbox'] *= 0
                result['track_id'] = -1
        else:
            result['track_id'] = track_id

        if use_one_euro:
            result['keypoints'] = _temporal_refine(
                result, match_result, fps=fps)
        del match_result

    return results, next_id


def vis_pose_tracking_result(model,
                             img,
                             result,
                             radius=4,
                             thickness=1,
                             kpt_score_thr=0.3,
                             dataset='TopDownCocoDataset',
                             dataset_info=None,
                             show=False,
                             out_file=None):
    """Visualize the pose tracking results on the image.

    Args:
        model (nn.Module): The loaded detector.
        img (str | np.ndarray): Image filename or loaded image.
        result (list[dict]): The results to draw over `img`
            (bbox_result, pose_result).
        radius (int): Radius of circles.
        thickness (int): Thickness of lines.
        kpt_score_thr (float): The threshold to visualize the keypoints.
        skeleton (list[tuple]): Default None.
        show (bool):  Whether to show the image. Default True.
        out_file (str|None): The filename of the output visualization image.
    """
    if hasattr(model, 'module'):
        model = model.module

    palette = np.array([[255, 128, 0], [255, 153, 51], [255, 178, 102],
                        [230, 230, 0], [255, 153, 255], [153, 204, 255],
                        [255, 102, 255], [255, 51, 255], [102, 178, 255],
                        [51, 153, 255], [255, 153, 153], [255, 102, 102],
                        [255, 51, 51], [153, 255, 153], [102, 255, 102],
                        [51, 255, 51], [0, 255, 0], [0, 0, 255], [255, 0, 0],
                        [255, 255, 255]])

    if dataset_info is None and dataset is not None:
        warnings.warn(
            'dataset is deprecated.'
            'Please set `dataset_info` in the config.'
            'Check https://github.com/open-mmlab/mmpose/pull/663 for details.',
            DeprecationWarning)
        # TODO: These will be removed in the later versions.
        if dataset in ('TopDownCocoDataset', 'BottomUpCocoDataset',
                       'TopDownOCHumanDataset'):
            kpt_num = 17
            skeleton = [[15, 13], [13, 11], [16, 14], [14, 12], [11, 12],
                        [5, 11], [6, 12], [5, 6], [5, 7], [6, 8], [7, 9],
                        [8, 10], [1, 2], [0, 1], [0, 2], [1, 3], [2, 4],
                        [3, 5], [4, 6]]

        elif dataset == 'TopDownCocoWholeBodyDataset':
            kpt_num = 133
            skeleton = [[15, 13], [13, 11], [16, 14], [14, 12], [11, 12],
                        [5, 11], [6, 12], [5, 6], [5, 7], [6, 8], [7, 9],
                        [8, 10], [1, 2], [0, 1], [0, 2],
                        [1, 3], [2, 4], [3, 5], [4, 6], [15, 17], [15, 18],
                        [15, 19], [16, 20], [16, 21], [16, 22], [91, 92],
                        [92, 93], [93, 94], [94, 95], [91, 96], [96, 97],
                        [97, 98], [98, 99], [91, 100], [100, 101], [101, 102],
                        [102, 103], [91, 104], [104, 105], [105, 106],
                        [106, 107], [91, 108], [108, 109], [109, 110],
                        [110, 111], [112, 113], [113, 114], [114, 115],
                        [115, 116], [112, 117], [117, 118], [118, 119],
                        [119, 120], [112, 121], [121, 122], [122, 123],
                        [123, 124], [112, 125], [125, 126], [126, 127],
                        [127, 128], [112, 129], [129, 130], [130, 131],
                        [131, 132]]
            radius = 1

        elif dataset == 'TopDownAicDataset':
            kpt_num = 14
            skeleton = [[2, 1], [1, 0], [0, 13], [13, 3], [3, 4], [4, 5],
                        [8, 7], [7, 6], [6, 9], [9, 10], [10, 11], [12, 13],
                        [0, 6], [3, 9]]

        elif dataset == 'TopDownMpiiDataset':
            kpt_num = 16
            skeleton = [[0, 1], [1, 2], [2, 6], [6, 3], [3, 4], [4, 5], [6, 7],
                        [7, 8], [8, 9], [8, 12], [12, 11], [11, 10], [8, 13],
                        [13, 14], [14, 15]]

        elif dataset in ('OneHand10KDataset', 'FreiHandDataset',
                         'PanopticDataset'):
            kpt_num = 21
            skeleton = [[0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7],
                        [7, 8], [0, 9], [9, 10], [10, 11], [11, 12], [0, 13],
                        [13, 14], [14, 15], [15, 16], [0, 17], [17, 18],
                        [18, 19], [19, 20]]

        elif dataset == 'InterHand2DDataset':
            kpt_num = 21
            skeleton = [[0, 1], [1, 2], [2, 3], [4, 5], [5, 6], [6, 7], [8, 9],
                        [9, 10], [10, 11], [12, 13], [13, 14], [14, 15],
                        [16, 17], [17, 18], [18, 19], [3, 20], [7, 20],
                        [11, 20], [15, 20], [19, 20]]

        else:
            raise NotImplementedError()

    elif dataset_info is not None:
        kpt_num = dataset_info.keypoint_num
        skeleton = dataset_info.skeleton

    for res in result:
        track_id = res['track_id']
        bbox_color = palette[track_id % len(palette)]
        pose_kpt_color = palette[[track_id % len(palette)] * kpt_num]
        pose_link_color = palette[[track_id % len(palette)] * len(skeleton)]
        img = model.show_result(
            img, [res],
            skeleton,
            radius=radius,
            thickness=thickness,
            pose_kpt_color=pose_kpt_color,
            pose_link_color=pose_link_color,
            bbox_color=tuple(bbox_color.tolist()),
            kpt_score_thr=kpt_score_thr,
            show=show,
            out_file=out_file)

    return img
