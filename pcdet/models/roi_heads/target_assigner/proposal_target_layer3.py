import numpy as np
import torch
import torch.nn as nn

from ....ops.iou3d_nms import iou3d_nms_utils

class ProposalTargetLayer(nn.Module):
    def __init__(self, roi_sampler_cfg):
        super().__init__()
        self.roi_sampler_cfg = roi_sampler_cfg

    def limit(self,ang):
        ang = ang % (2 * np.pi)

        ang[ang > np.pi] = ang[ang > np.pi] - 2 * np.pi

        ang[ang < -np.pi] = ang[ang < -np.pi] + 2 * np.pi

        return ang

    def ang_weight(self,pred, gt):

        a = torch.abs(pred - gt)
        b = 2 * np.pi - torch.abs(pred - gt)

        res = torch.stack([a, b])

        res = torch.min(res, 0)[0]

        return 1 - res / np.pi

    def forward(self, batch_dict,ind=''):
        """
        Args:
            batch_dict:
                batch_size:
                rois: (B, num_rois, 7 + C)
                roi_scores: (B, num_rois)
                gt_boxes: (B, N, 7 + C + 1)
                roi_labels: (B, num_rois)
        Returns:
            batch_dict:
                rois: (B, M, 7 + C)
                gt_of_rois: (B, M, 7 + C)
                gt_iou_of_rois: (B, M)
                roi_scores: (B, M)
                roi_labels: (B, M)
                reg_valid_mask: (B, M)
                rcnn_cls_labels: (B, M)
        """
        batch_rois, batch_gt_of_rois, batch_roi_ious, batch_roi_scores, batch_roi_labels = self.sample_rois_for_rcnn(
            batch_dict=batch_dict,ind=ind,
        )
        # regression valid mask
        if self.roi_sampler_cfg.CLS_SCORE_TYPE in ['roi_ioud_x','roi_ioud']:

            roi_angle = batch_rois[..., 6].clone()
            gt_angle = batch_gt_of_rois[..., 6].clone()
            roi_angle = self.limit(roi_angle)
            gt_angle = self.limit(gt_angle)
            ang_target = self.ang_weight(roi_angle, gt_angle)
            direction_constraint = self.roi_sampler_cfg.DIRECTION_C

            ang_target = (torch.clamp(ang_target, direction_constraint, 1) - direction_constraint) / (
                        1 - direction_constraint)
            batch_roi_ious*=ang_target

        if self.roi_sampler_cfg.CLS_SCORE_TYPE in ['roi_iou_x','roi_ioud_x']:
            reg_valid_mask = batch_roi_ious.new_zeros(batch_roi_ious.shape).long()
            for cls_i in range(len(self.roi_sampler_cfg.REG_FG_THRESH)):
                reg_fg_thresh = self.roi_sampler_cfg.REG_FG_THRESH[cls_i]
                cls_mask = batch_gt_of_rois[...,-1] == (cls_i+1)
                this_reg_valid_mask = ((batch_roi_ious > reg_fg_thresh) & cls_mask).long()
                reg_valid_mask += this_reg_valid_mask
        else:
            reg_valid_mask = (batch_roi_ious > self.roi_sampler_cfg.REG_FG_THRESH).long()

        # classification label
        if self.roi_sampler_cfg.CLS_SCORE_TYPE == 'cls':
            batch_cls_labels = (batch_roi_ious > self.roi_sampler_cfg.CLS_FG_THRESH).long()
            ignore_mask = (batch_roi_ious > self.roi_sampler_cfg.CLS_BG_THRESH) & \
                          (batch_roi_ious < self.roi_sampler_cfg.CLS_FG_THRESH)
            batch_cls_labels[ignore_mask > 0] = -1
        elif self.roi_sampler_cfg.CLS_SCORE_TYPE in ['roi_iou','roi_ioud']:
            iou_bg_thresh = self.roi_sampler_cfg.CLS_BG_THRESH
            iou_fg_thresh = self.roi_sampler_cfg.CLS_FG_THRESH
            fg_mask = batch_roi_ious > iou_fg_thresh
            bg_mask = batch_roi_ious < iou_bg_thresh
            interval_mask = (fg_mask == 0) & (bg_mask == 0)

            batch_cls_labels = (fg_mask > 0).float()
            batch_cls_labels[interval_mask] = \
                (batch_roi_ious[interval_mask] - iou_bg_thresh) / (iou_fg_thresh - iou_bg_thresh)


        elif self.roi_sampler_cfg.CLS_SCORE_TYPE in ['roi_iou_x','roi_ioud_x']:
            all_iou_bg_thresh = self.roi_sampler_cfg.CLS_BG_THRESH
            all_iou_fg_thresh = self.roi_sampler_cfg.CLS_FG_THRESH
            batch_cls_labels = batch_roi_ious.new_zeros(batch_roi_ious.shape)
            for cls_id in range(len(all_iou_bg_thresh)):
                gt_cls = batch_gt_of_rois[..., -1]
                iou_fg_thresh = all_iou_fg_thresh[cls_id]
                iou_bg_thresh = all_iou_bg_thresh[cls_id]

                cls_mask = gt_cls == (cls_id+1)

                fg_mask = batch_roi_ious > iou_fg_thresh
                bg_mask = batch_roi_ious < iou_bg_thresh
                interval_mask = (fg_mask == 0) & (bg_mask == 0)

                cls_labels = (fg_mask > 0).float()
                cls_labels[interval_mask] = \
                    (batch_roi_ious[interval_mask] - iou_bg_thresh) / (iou_fg_thresh - iou_bg_thresh)

                batch_cls_labels[cls_mask] = cls_labels[cls_mask]

        else:
            raise NotImplementedError

        targets_dict = {'rois'+ind: batch_rois, 'gt_of_rois'+ind: batch_gt_of_rois, 'gt_iou_of_rois'+ind: batch_roi_ious,
                        'roi_scores'+ind: batch_roi_scores, 'roi_labels'+ind: batch_roi_labels,
                        'reg_valid_mask'+ind: reg_valid_mask,
                        'rcnn_cls_labels'+ind: batch_cls_labels}

        return targets_dict

    def sample_rois_for_rcnn(self, batch_dict, ind=''):
        """
        Args:
            batch_dict:
                batch_size:
                rois: (B, num_rois, 7 + C)
                roi_scores: (B, num_rois)
                gt_boxes: (B, N, 7 + C + 1)
                roi_labels: (B, num_rois)
        Returns:

        """
        batch_size = batch_dict['batch_size']
        rois = batch_dict['rois'+ind]
        roi_scores = batch_dict['roi_scores'+ind]
        roi_labels = batch_dict['roi_labels']
        gt_boxes = batch_dict['gt_boxes']

        gt_code_size = gt_boxes.shape[-1]
        roi_code_size = rois.shape[-1]
        batch_rois = rois.new_zeros(batch_size, self.roi_sampler_cfg.ROI_PER_IMAGE, roi_code_size)
        batch_gt_of_rois = rois.new_zeros(batch_size, self.roi_sampler_cfg.ROI_PER_IMAGE, gt_code_size )
        batch_roi_ious = rois.new_zeros(batch_size, self.roi_sampler_cfg.ROI_PER_IMAGE)
        batch_roi_scores = rois.new_zeros(batch_size, self.roi_sampler_cfg.ROI_PER_IMAGE)
        batch_roi_labels = rois.new_zeros((batch_size, self.roi_sampler_cfg.ROI_PER_IMAGE), dtype=torch.long)

        for index in range(batch_size):
            cur_roi, cur_gt, cur_roi_labels, cur_roi_scores = \
                rois[index], gt_boxes[index], roi_labels[index], roi_scores[index]
            k = cur_gt.__len__() - 1
            while k > 0 and cur_gt[k].sum() == 0:
                k -= 1
            cur_gt = cur_gt[:k + 1]
            cur_gt = cur_gt.new_zeros((1, cur_gt.shape[1])) if len(cur_gt) == 0 else cur_gt

            if self.roi_sampler_cfg.get('SAMPLE_ROI_BY_EACH_CLASS', False):
                max_overlaps, gt_assignment = self.get_max_iou_with_same_class(
                    rois=cur_roi, roi_labels=cur_roi_labels,
                    gt_boxes=cur_gt[:, 0:7], gt_labels=cur_gt[:, -1].long()
                )
            else:
                iou3d = iou3d_nms_utils.boxes_iou3d_gpu(cur_roi, cur_gt[:, 0:7])  # (M, N)
                max_overlaps, gt_assignment = torch.max(iou3d, dim=1)

            if self.roi_sampler_cfg.CLS_SCORE_TYPE in ['roi_iou_x','roi_ioud_x']:
                sampled_inds = self.subsample_rois(max_overlaps=max_overlaps,gts = cur_gt[gt_assignment])
            else:
                sampled_inds = self.subsample_rois(max_overlaps=max_overlaps)

            batch_rois[index] = cur_roi[sampled_inds]
            batch_roi_labels[index] = cur_roi_labels[sampled_inds]
            batch_roi_ious[index] = max_overlaps[sampled_inds]
            batch_roi_scores[index] = cur_roi_scores[sampled_inds]
            batch_gt_of_rois[index] = cur_gt[gt_assignment[sampled_inds]]

        return batch_rois, batch_gt_of_rois, batch_roi_ious, batch_roi_scores, batch_roi_labels

    def subsample_rois(self, max_overlaps, gts=None):
        # sample fg, easy_bg, hard_bg
        fg_rois_per_image = int(np.round(self.roi_sampler_cfg.FG_RATIO * self.roi_sampler_cfg.ROI_PER_IMAGE))

        if gts is None:
            fg_thresh = min(self.roi_sampler_cfg.REG_FG_THRESH, self.roi_sampler_cfg.CLS_FG_THRESH)
            fg_inds = ((max_overlaps >= fg_thresh)).nonzero().view(-1)
        else:
            fg_inds = max_overlaps.new_zeros(max_overlaps.shape).long()
            for i in range(len(self.roi_sampler_cfg.CLS_FG_THRESH)):
                cls_mask = gts[...,-1] == (i+1)
                this_fg_thresh = min(self.roi_sampler_cfg.REG_FG_THRESH[i], self.roi_sampler_cfg.CLS_FG_THRESH[i])
                this_fg_inds = (max_overlaps >= this_fg_thresh) & cls_mask
                fg_inds+=this_fg_inds
            fg_inds = fg_inds.nonzero().view(-1)


        easy_bg_inds = ((max_overlaps < self.roi_sampler_cfg.CLS_BG_THRESH_LO)).nonzero().view(-1)

        if gts is None:
            hard_bg_inds = ((max_overlaps < self.roi_sampler_cfg.REG_FG_THRESH) &
                            (max_overlaps >= self.roi_sampler_cfg.CLS_BG_THRESH_LO)).nonzero().view(-1)
        else:
            hard_bg_inds = max_overlaps.new_zeros(max_overlaps.shape).long()
            for i in range(len(self.roi_sampler_cfg.REG_FG_THRESH)):
                cls_mask = gts[...,-1] == (i+1)
                this_hard_bg_inds = ((max_overlaps < self.roi_sampler_cfg.REG_FG_THRESH[i]) &
                                (max_overlaps >= self.roi_sampler_cfg.CLS_BG_THRESH_LO) & cls_mask)
                hard_bg_inds+=this_hard_bg_inds
            hard_bg_inds = hard_bg_inds.nonzero().view(-1)


        fg_num_rois = fg_inds.numel()
        bg_num_rois = hard_bg_inds.numel() + easy_bg_inds.numel()

        if fg_num_rois > 0 and bg_num_rois > 0:
            # sampling fg
            fg_rois_per_this_image = min(fg_rois_per_image, fg_num_rois)

            rand_num = torch.from_numpy(np.random.permutation(fg_num_rois)).type_as(max_overlaps).long()
            fg_inds = fg_inds[rand_num[:fg_rois_per_this_image]]

            # sampling bg
            bg_rois_per_this_image = self.roi_sampler_cfg.ROI_PER_IMAGE - fg_rois_per_this_image
            bg_inds = self.sample_bg_inds(
                hard_bg_inds, easy_bg_inds, bg_rois_per_this_image, self.roi_sampler_cfg.HARD_BG_RATIO
            )

        elif fg_num_rois > 0 and bg_num_rois == 0:
            # sampling fg
            rand_num = np.floor(np.random.rand(self.roi_sampler_cfg.ROI_PER_IMAGE) * fg_num_rois)
            rand_num = torch.from_numpy(rand_num).type_as(max_overlaps).long()
            fg_inds = fg_inds[rand_num]
            bg_inds = []

        elif bg_num_rois > 0 and fg_num_rois == 0:
            # sampling bg
            bg_rois_per_this_image = self.roi_sampler_cfg.ROI_PER_IMAGE
            bg_inds = self.sample_bg_inds(
                hard_bg_inds, easy_bg_inds, bg_rois_per_this_image, self.roi_sampler_cfg.HARD_BG_RATIO
            )
        else:
            print('maxoverlaps:(min=%f, max=%f)' % (max_overlaps.min().item(), max_overlaps.max().item()))
            print('ERROR: FG=%d, BG=%d' % (fg_num_rois, bg_num_rois))
            raise NotImplementedError

        sampled_inds = torch.cat((fg_inds, bg_inds), dim=0)
        return sampled_inds

    @staticmethod
    def sample_bg_inds(hard_bg_inds, easy_bg_inds, bg_rois_per_this_image, hard_bg_ratio):
        if hard_bg_inds.numel() > 0 and easy_bg_inds.numel() > 0:
            hard_bg_rois_num = min(int(bg_rois_per_this_image * hard_bg_ratio), len(hard_bg_inds))
            easy_bg_rois_num = bg_rois_per_this_image - hard_bg_rois_num

            # sampling hard bg
            rand_idx = torch.randint(low=0, high=hard_bg_inds.numel(), size=(hard_bg_rois_num,)).long()
            hard_bg_inds = hard_bg_inds[rand_idx]

            # sampling easy bg
            rand_idx = torch.randint(low=0, high=easy_bg_inds.numel(), size=(easy_bg_rois_num,)).long()
            easy_bg_inds = easy_bg_inds[rand_idx]

            bg_inds = torch.cat([hard_bg_inds, easy_bg_inds], dim=0)
        elif hard_bg_inds.numel() > 0 and easy_bg_inds.numel() == 0:
            hard_bg_rois_num = bg_rois_per_this_image
            # sampling hard bg
            rand_idx = torch.randint(low=0, high=hard_bg_inds.numel(), size=(hard_bg_rois_num,)).long()
            bg_inds = hard_bg_inds[rand_idx]
        elif hard_bg_inds.numel() == 0 and easy_bg_inds.numel() > 0:
            easy_bg_rois_num = bg_rois_per_this_image
            # sampling easy bg
            rand_idx = torch.randint(low=0, high=easy_bg_inds.numel(), size=(easy_bg_rois_num,)).long()
            bg_inds = easy_bg_inds[rand_idx]
        else:
            raise NotImplementedError

        return bg_inds

    @staticmethod
    def get_max_iou_with_same_class(rois, roi_labels, gt_boxes, gt_labels):
        """
        Args:
            rois: (N, 7)
            roi_labels: (N)
            gt_boxes: (N, )
            gt_labels:

        Returns:

        """
        """
        :param rois: (N, 7)
        :param roi_labels: (N)
        :param gt_boxes: (N, 8)
        :return:
        """
        max_overlaps = rois.new_zeros(rois.shape[0])
        gt_assignment = roi_labels.new_zeros(roi_labels.shape[0])

        for k in range(gt_labels.min().item(), gt_labels.max().item() + 1):
            roi_mask = (roi_labels == k)
            gt_mask = (gt_labels == k)
            if roi_mask.sum() > 0 and gt_mask.sum() > 0:
                cur_roi = rois[roi_mask]
                cur_gt = gt_boxes[gt_mask]
                original_gt_assignment = gt_mask.nonzero().view(-1)

                iou3d = iou3d_nms_utils.boxes_iou3d_gpu(cur_roi[:,0:7], cur_gt[:,0:7])  # (M, N)
                cur_max_overlaps, cur_gt_assignment = torch.max(iou3d, dim=1)
                max_overlaps[roi_mask] = cur_max_overlaps
                gt_assignment[roi_mask] = original_gt_assignment[cur_gt_assignment]

        return max_overlaps, gt_assignment

class ProposalTargetLayerT(nn.Module):
    def __init__(self, roi_sampler_cfg):
        super().__init__()
        self.roi_sampler_cfg = roi_sampler_cfg

    def limit(self,ang):
        ang = ang % (2 * np.pi)

        ang[ang > np.pi] = ang[ang > np.pi] - 2 * np.pi

        ang[ang < -np.pi] = ang[ang < -np.pi] + 2 * np.pi

        return ang

    def ang_weight(self,pred, gt):

        a = torch.abs(pred - gt)
        b = 2 * np.pi - torch.abs(pred - gt)

        res = torch.stack([a, b])

        res = torch.min(res, 0)[0]

        return 1 - res / np.pi

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size:
                rois: (B, num_rois, 7 + C)
                roi_scores: (B, num_rois)
                gt_boxes: (B, N, 7 + C + 1)
                roi_labels: (B, num_rois)
        Returns:
            batch_dict:
                rois: (B, M, 7 + C)
                gt_of_rois: (B, M, 7 + C)
                gt_iou_of_rois: (B, M)
                roi_scores: (B, M)
                roi_labels: (B, M)
                reg_valid_mask: (B, M)
                rcnn_cls_labels: (B, M)
        """
        batch_rois, batch_gt_of_rois, batch_roi_ious, batch_roi_mious, batch_roi_scores, batch_roi_labels, batch_gt_bbs_mask\
            = self.sample_rois_for_rcnn(
            batch_dict=batch_dict,
        )

        if self.roi_sampler_cfg.CLS_SCORE_TYPE == 'roi_mious' or self.roi_sampler_cfg.CLS_SCORE_TYPE == 'mcls':
            batch_roi_ious = batch_roi_mious

        # regression valid mask
        reg_valid_mask = (batch_roi_ious > self.roi_sampler_cfg.REG_FG_THRESH).long()

        # classification label
        if self.roi_sampler_cfg.CLS_SCORE_TYPE == 'cls' or self.roi_sampler_cfg.CLS_SCORE_TYPE == 'mcls':
            batch_cls_labels = (batch_roi_ious > self.roi_sampler_cfg.CLS_FG_THRESH).long()
            ignore_mask = (batch_roi_ious > self.roi_sampler_cfg.CLS_BG_THRESH) & \
                          (batch_roi_ious < self.roi_sampler_cfg.CLS_FG_THRESH)
            batch_cls_labels[ignore_mask > 0] = -1
        elif self.roi_sampler_cfg.CLS_SCORE_TYPE == 'roi_ious' or self.roi_sampler_cfg.CLS_SCORE_TYPE == 'roi_mious':
            iou_bg_thresh = self.roi_sampler_cfg.CLS_BG_THRESH
            iou_fg_thresh = self.roi_sampler_cfg.CLS_FG_THRESH
            fg_mask = batch_roi_ious > iou_fg_thresh
            bg_mask = batch_roi_ious < iou_bg_thresh
            interval_mask = (fg_mask == 0) & (bg_mask == 0)

            batch_cls_labels = (fg_mask > 0).float()
            batch_cls_labels[interval_mask] = \
                (batch_roi_ious[interval_mask] - iou_bg_thresh) / (iou_fg_thresh - iou_bg_thresh)

        elif self.roi_sampler_cfg.CLS_SCORE_TYPE == 'roi_ioud':
            iou_bg_thresh = self.roi_sampler_cfg.CLS_BG_THRESH
            iou_fg_thresh = self.roi_sampler_cfg.CLS_FG_THRESH
            fg_mask = batch_roi_ious > iou_fg_thresh
            bg_mask = batch_roi_ious < iou_bg_thresh
            interval_mask = (fg_mask == 0) & (bg_mask == 0)

            batch_cls_labels = (fg_mask > 0).float()
            batch_cls_labels[interval_mask] = \
                (batch_roi_ious[interval_mask] - iou_bg_thresh) / (iou_fg_thresh - iou_bg_thresh)

            ang_roi = batch_rois[...,6]
            ang_gt = batch_gt_of_rois[...,6]

            ang_roi = self.limit(ang_roi)
            ang_gt = self.limit(ang_gt)

            ang_target = self.ang_weight(ang_roi,ang_gt)

            ang_target = torch.clamp(ang_target,0.0,0.8)/0.8

            batch_cls_labels*=ang_target


        else:
            raise NotImplementedError

        targets_dict = {'rois': batch_rois, 'gt_of_rois': batch_gt_of_rois, 'gt_iou_of_rois': batch_roi_ious,
                        'roi_scores': batch_roi_scores, 'roi_labels': batch_roi_labels,
                        'reg_valid_mask': reg_valid_mask, 'gt_bbs_mask': batch_gt_bbs_mask,
                        'rcnn_cls_labels': batch_cls_labels}

        return targets_dict


    def sample_rois_for_rcnn(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size:
                rois: (B, num_rois, 7 + C)
                roi_scores: (B, num_rois)
                gt_boxes: (B, N, 7 + C + 1)
                roi_labels: (B, num_rois)
        Returns:

        """
        batch_size = batch_dict['batch_size']
        rois = batch_dict['rois']
        roi_scores = batch_dict['roi_scores']
        roi_labels = batch_dict['roi_labels']
        gt_tracklets = batch_dict['gt_tracklets']

        num_frame = gt_tracklets.shape[-1]//7

        gt_bbs_mask = batch_dict['gt_bbs_mask']


        gt_code_size = gt_tracklets.shape[-1]
        roi_code_size = rois.shape[-1]
        batch_rois = rois.new_zeros(batch_size, self.roi_sampler_cfg.ROI_PER_IMAGE, roi_code_size)
        batch_gt_of_rois = rois.new_zeros(batch_size, self.roi_sampler_cfg.ROI_PER_IMAGE, gt_code_size )
        batch_roi_scores = rois.new_zeros(batch_size, self.roi_sampler_cfg.ROI_PER_IMAGE)
        batch_roi_labels = rois.new_zeros((batch_size, self.roi_sampler_cfg.ROI_PER_IMAGE), dtype=torch.long)

        batch_all_roi_ious = rois.new_zeros(batch_size, self.roi_sampler_cfg.ROI_PER_IMAGE,num_frame)
        batch_gt_bbs_mask = rois.new_zeros(batch_size, self.roi_sampler_cfg.ROI_PER_IMAGE, num_frame)

        for index in range(batch_size):
            cur_roi, cur_gt, cur_roi_labels, cur_roi_scores,cur_bbs_mask = \
                rois[index], gt_tracklets[index], roi_labels[index], roi_scores[index],gt_bbs_mask[index]
            k = cur_gt.__len__() - 1
            while k > 0 and cur_gt[k].sum() == 0:
                k -= 1
            cur_gt = cur_gt[:k + 1]
            cur_gt = cur_gt.new_zeros((1, cur_gt.shape[1])) if len(cur_gt) == 0 else cur_gt
            cur_bbs_mask = cur_bbs_mask.new_zeros((1, cur_bbs_mask.shape[1])) if len(cur_bbs_mask) == 0 else cur_bbs_mask

            if self.roi_sampler_cfg.get('SAMPLE_ROI_BY_EACH_CLASS', False):
                max_overlaps, gt_assignment = self.get_max_iou_with_same_class(
                    rois=cur_roi[:,0:7], roi_labels=cur_roi_labels,
                    gt_boxes=cur_gt[:, 0:7], gt_labels=cur_gt[:, -1].long()
                )
            else:
                iou3d = iou3d_nms_utils.boxes_iou3d_gpu(cur_roi[:, 0:7], cur_gt[:, 0:7])  # (M, N)
                max_overlaps, gt_assignment = torch.max(iou3d, dim=1)

            sampled_inds = self.subsample_rois(max_overlaps=max_overlaps)

            batch_rois[index] = cur_roi[sampled_inds]
            batch_roi_labels[index] = cur_roi_labels[sampled_inds]
            batch_roi_scores[index] = cur_roi_scores[sampled_inds]
            batch_gt_of_rois[index] = cur_gt[gt_assignment[sampled_inds]]

            batch_all_roi_ious[index,:,0] = max_overlaps[sampled_inds]

            batch_gt_bbs_mask[index] = cur_bbs_mask[gt_assignment[sampled_inds]]

        for i in range(1, num_frame):
            for j in range(batch_size):

                this_roi = batch_rois[j,:,i*7:i*7+7]
                this_gt_of_roi = batch_gt_of_rois[j,:,i*7:i*7+7]

                all_ious = iou3d_nms_utils.boxes_iou3d_gpu(this_roi[:, 0:7], this_gt_of_roi[:, 0:7])
                box_num = this_roi.shape[0]

                ious = all_ious[range(box_num),range(box_num)]

                batch_all_roi_ious[j,:,i] = ious


        tracks_mean_ious = batch_all_roi_ious.sum(-1)/(batch_gt_bbs_mask.sum(-1)+0.00001)

        return batch_rois, batch_gt_of_rois,batch_all_roi_ious[...,0], tracks_mean_ious, batch_roi_scores, batch_roi_labels, batch_gt_bbs_mask

    def subsample_rois(self, max_overlaps):
        # sample fg, easy_bg, hard_bg
        fg_rois_per_image = int(np.round(self.roi_sampler_cfg.FG_RATIO * self.roi_sampler_cfg.ROI_PER_IMAGE))
        fg_thresh = min(self.roi_sampler_cfg.REG_FG_THRESH, self.roi_sampler_cfg.CLS_FG_THRESH)

        fg_inds = ((max_overlaps >= fg_thresh)).nonzero().view(-1)
        easy_bg_inds = ((max_overlaps < self.roi_sampler_cfg.CLS_BG_THRESH_LO)).nonzero().view(-1)
        hard_bg_inds = ((max_overlaps < self.roi_sampler_cfg.REG_FG_THRESH) &
                (max_overlaps >= self.roi_sampler_cfg.CLS_BG_THRESH_LO)).nonzero().view(-1)

        fg_num_rois = fg_inds.numel()
        bg_num_rois = hard_bg_inds.numel() + easy_bg_inds.numel()

        if fg_num_rois > 0 and bg_num_rois > 0:
            # sampling fg
            fg_rois_per_this_image = min(fg_rois_per_image, fg_num_rois)

            rand_num = torch.from_numpy(np.random.permutation(fg_num_rois)).type_as(max_overlaps).long()
            fg_inds = fg_inds[rand_num[:fg_rois_per_this_image]]

            # sampling bg
            bg_rois_per_this_image = self.roi_sampler_cfg.ROI_PER_IMAGE - fg_rois_per_this_image
            bg_inds = self.sample_bg_inds(
                hard_bg_inds, easy_bg_inds, bg_rois_per_this_image, self.roi_sampler_cfg.HARD_BG_RATIO
            )

        elif fg_num_rois > 0 and bg_num_rois == 0:
            # sampling fg
            rand_num = np.floor(np.random.rand(self.roi_sampler_cfg.ROI_PER_IMAGE) * fg_num_rois)
            rand_num = torch.from_numpy(rand_num).type_as(max_overlaps).long()
            fg_inds = fg_inds[rand_num]
            bg_inds = []

        elif bg_num_rois > 0 and fg_num_rois == 0:
            # sampling bg
            bg_rois_per_this_image = self.roi_sampler_cfg.ROI_PER_IMAGE
            bg_inds = self.sample_bg_inds(
                hard_bg_inds, easy_bg_inds, bg_rois_per_this_image, self.roi_sampler_cfg.HARD_BG_RATIO
            )
        else:
            print('maxoverlaps:(min=%f, max=%f)' % (max_overlaps.min().item(), max_overlaps.max().item()))
            print('ERROR: FG=%d, BG=%d' % (fg_num_rois, bg_num_rois))
            raise NotImplementedError

        sampled_inds = torch.cat((fg_inds, bg_inds), dim=0)
        return sampled_inds

    @staticmethod
    def sample_bg_inds(hard_bg_inds, easy_bg_inds, bg_rois_per_this_image, hard_bg_ratio):
        if hard_bg_inds.numel() > 0 and easy_bg_inds.numel() > 0:
            hard_bg_rois_num = min(int(bg_rois_per_this_image * hard_bg_ratio), len(hard_bg_inds))
            easy_bg_rois_num = bg_rois_per_this_image - hard_bg_rois_num

            # sampling hard bg
            rand_idx = torch.randint(low=0, high=hard_bg_inds.numel(), size=(hard_bg_rois_num,)).long()
            hard_bg_inds = hard_bg_inds[rand_idx]

            # sampling easy bg
            rand_idx = torch.randint(low=0, high=easy_bg_inds.numel(), size=(easy_bg_rois_num,)).long()
            easy_bg_inds = easy_bg_inds[rand_idx]

            bg_inds = torch.cat([hard_bg_inds, easy_bg_inds], dim=0)
        elif hard_bg_inds.numel() > 0 and easy_bg_inds.numel() == 0:
            hard_bg_rois_num = bg_rois_per_this_image
            # sampling hard bg
            rand_idx = torch.randint(low=0, high=hard_bg_inds.numel(), size=(hard_bg_rois_num,)).long()
            bg_inds = hard_bg_inds[rand_idx]
        elif hard_bg_inds.numel() == 0 and easy_bg_inds.numel() > 0:
            easy_bg_rois_num = bg_rois_per_this_image
            # sampling easy bg
            rand_idx = torch.randint(low=0, high=easy_bg_inds.numel(), size=(easy_bg_rois_num,)).long()
            bg_inds = easy_bg_inds[rand_idx]
        else:
            raise NotImplementedError

        return bg_inds

    @staticmethod
    def get_max_iou_with_same_class(rois, roi_labels, gt_boxes, gt_labels):
        """
        Args:
            rois: (N, 7)
            roi_labels: (N)
            gt_boxes: (N, )
            gt_labels:

        Returns:

        """
        """
        :param rois: (N, 7)
        :param roi_labels: (N)
        :param gt_boxes: (N, 8)
        :return:
        """
        max_overlaps = rois.new_zeros(rois.shape[0])
        gt_assignment = roi_labels.new_zeros(roi_labels.shape[0])

        for k in range(gt_labels.min().item(), gt_labels.max().item() + 1):
            roi_mask = (roi_labels == k)
            gt_mask = (gt_labels == k)
            if roi_mask.sum() > 0 and gt_mask.sum() > 0:
                cur_roi = rois[roi_mask]
                cur_gt = gt_boxes[gt_mask]
                original_gt_assignment = gt_mask.nonzero().view(-1)

                iou3d = iou3d_nms_utils.boxes_iou3d_gpu(cur_roi[:,0:7], cur_gt[:,0:7])  # (M, N)
                cur_max_overlaps, cur_gt_assignment = torch.max(iou3d, dim=1)
                max_overlaps[roi_mask] = cur_max_overlaps
                gt_assignment[roi_mask] = original_gt_assignment[cur_gt_assignment]

        return max_overlaps, gt_assignment
