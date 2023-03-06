import torch
import numpy as np
from ...ops.iou3d_nms import iou3d_nms_utils

def limit(ang):
    ang = ang % (2 * np.pi)

    ang[ang > np.pi] = ang[ang > np.pi] - 2 * np.pi

    ang[ang < -np.pi] = ang[ang < -np.pi] + 2 * np.pi

    return ang

def compute_WBF(det_names,
                det_scores,
                det_boxes,
                iou_thresh=0.85,
                iou_thresh2=0.1,
                type='mean',
                retain_low=False,
                score_thresh=0.4):
    if len(det_names) == 0:
        return det_names, det_scores, det_boxes

    sort = det_scores.argsort()
    sort = sort[::-1]
    det_scores = det_scores[sort]
    det_names = det_names[sort]
    det_boxes = det_boxes[sort]

    cluster_id = -1
    cluster_box_dict = {}
    cluster_score_dict = {}

    cluster_merged_dict = {}
    cluster_name_dict = {}
    det_boxes[:, 6] = limit(det_boxes[:, 6])

    out_boxes = []
    out_scores = []
    out_name = []

    for i, box in enumerate(det_boxes):

        score = det_scores[i]
        name = det_names[i]
        if i == 0:
            cluster_id += 1
            cluster_box_dict[cluster_id] = [box]
            cluster_score_dict[cluster_id] = [score]
            cluster_merged_dict[cluster_id] = box
            cluster_name_dict[cluster_id] = name
            continue

        valid_clusters = []
        keys = list(cluster_merged_dict)
        keys.sort()
        for key in keys:
            valid_clusters.append(cluster_merged_dict[key])

        valid_clusters = np.array(valid_clusters).reshape((-1, 7))
        ious = iou3d_nms_utils.boxes_bev_iou_cpu(np.array([box[:7]]), valid_clusters)

        argmax = np.argmax(ious, -1)[0]
        max_iou = np.max(ious, -1)[0]

        if max_iou >= iou_thresh:
            cluster_box_dict[argmax].append(box)
            cluster_score_dict[argmax].append(score)
        elif iou_thresh2 <= max_iou < iou_thresh and score > score_thresh and retain_low:
            if np.max(cluster_score_dict[argmax])-score < 0.1:
                out_scores.append(score_thresh)
                out_boxes.append(box)
                out_name.append(name)
        elif 0.03 <= max_iou < iou_thresh2 and retain_low:
            continue
        elif (not retain_low) and 0.03 <= max_iou < iou_thresh:
            continue
        else:
            cluster_id += 1
            cluster_box_dict[cluster_id] = [box]
            cluster_score_dict[cluster_id] = [score]
            cluster_merged_dict[cluster_id] = box
            cluster_name_dict[cluster_id] = name

    for i in cluster_merged_dict.keys():
        if type == 'mean':
            score_sum = 0
            box_sum = np.zeros(shape=(7,))

            angles = []

            for j, sub_score in enumerate(cluster_score_dict[i]):
                box_sum += cluster_box_dict[i][j]
                score_sum += sub_score
                angles.append(cluster_box_dict[i][j][6])
            box_sum /= len(cluster_score_dict[i])
            score_sum /= len(cluster_score_dict[i])

            cluster_merged_dict[i][:6] = box_sum[:6]

            angles = np.array(angles)
            angles = limit(angles)
            res = angles - cluster_merged_dict[i][6]
            res = limit(res)
            res = res[np.abs(res) < 1.5]
            res = res.mean()
            b = cluster_merged_dict[i][6] + res
            cluster_merged_dict[i][6] = b

            out_scores.append(np.max(cluster_score_dict[i]))
            out_boxes.append(cluster_merged_dict[i])
            out_name.append(cluster_name_dict[i])
        else:
            out_scores.append(np.max(cluster_score_dict[i]))
            out_boxes.append(cluster_merged_dict[i])
            out_name.append(cluster_name_dict[i])

    out_boxes = np.array(out_boxes)
    out_scores = np.array(out_scores)
    out_names = np.array(out_name)

    return out_names, out_scores, out_boxes

def class_agnostic_nms(box_scores, box_preds, nms_config, score_thresh=None):
    src_box_scores = box_scores
    if score_thresh is not None:
        scores_mask = (box_scores >= score_thresh)
        box_scores = box_scores[scores_mask]
        box_preds = box_preds[scores_mask]

    selected = []
    if box_scores.shape[0] > 0:
        box_scores_nms, indices = torch.topk(box_scores, k=min(nms_config.NMS_PRE_MAXSIZE, box_scores.shape[0]))
        boxes_for_nms = box_preds[indices]
        keep_idx, selected_scores = getattr(iou3d_nms_utils, nms_config.NMS_TYPE)(
                boxes_for_nms[:, 0:7], box_scores_nms, nms_config.NMS_THRESH, **nms_config
        )
        selected = indices[keep_idx[:nms_config.NMS_POST_MAXSIZE]]

    if score_thresh is not None:
        original_idxs = scores_mask.nonzero().view(-1)
        selected = original_idxs[selected]
    return selected, src_box_scores[selected]


def multi_classes_nms(cls_scores, box_preds, nms_config, score_thresh=None):
    """
    Args:
        cls_scores: (N, num_class)
        box_preds: (N, 7 + C)
        nms_config:
        score_thresh:

    Returns:

    """
    pred_scores, pred_labels, pred_boxes = [], [], []
    for k in range(cls_scores.shape[1]):
        if score_thresh is not None:
            scores_mask = (cls_scores[:, k] >= score_thresh)
            box_scores = cls_scores[scores_mask, k]
            cur_box_preds = box_preds[scores_mask]
        else:
            box_scores = cls_scores[:, k]

        selected = []
        if box_scores.shape[0] > 0:
            box_scores_nms, indices = torch.topk(box_scores, k=min(nms_config.NMS_PRE_MAXSIZE, box_scores.shape[0]))
            boxes_for_nms = cur_box_preds[indices]
            keep_idx, selected_scores = getattr(iou3d_nms_utils, nms_config.NMS_TYPE)(
                    boxes_for_nms[:, 0:7], box_scores_nms, nms_config.NMS_THRESH, **nms_config
            )
            selected = indices[keep_idx[:nms_config.NMS_POST_MAXSIZE]]

        pred_scores.append(box_scores[selected])
        pred_labels.append(box_scores.new_ones(len(selected)).long() * k)
        pred_boxes.append(cur_box_preds[selected])

    pred_scores = torch.cat(pred_scores, dim=0)
    pred_labels = torch.cat(pred_labels, dim=0)
    pred_boxes = torch.cat(pred_boxes, dim=0)

    return pred_scores, pred_labels, pred_boxes
