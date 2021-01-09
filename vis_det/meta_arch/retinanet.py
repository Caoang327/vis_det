from fvcore.nn import giou_loss, sigmoid_focal_loss_jit
import copy
import types
import torch
from detectron2.modeling.meta_arch.retinanet import permute_to_N_HWA_K
from detectron2.modeling.matcher import Matcher
from detectron2.structures import Boxes, pairwise_iou
from detectron2.layers import cat


def permute_all_cls_and_box_to_N_HWA_K_and_concat(box_cls, box_delta, num_classes=80):
    """
    Rearrange the tensor layout from the network output, i.e.: list[Tensor]: #lvl tensors of shape (N, A x K, Hi, Wi)
    to per-image predictions, i.e.:Tensor: of shape (N x sum(Hi x Wi x A), K)
    """
    # for each feature level, permute the outputs to make them be in the
    # same format as the labels. Note that the labels are computed for
    # all feature levels concatenated, so we keep the same representation
    # for the objectness and the box_delta
    box_cls_flattened = [permute_to_N_HWA_K(x, num_classes) for x in box_cls]
    box_delta_flattened = [permute_to_N_HWA_K(x, 4) for x in box_delta]
    # concatenate on the first dimension (representing the feature levels), to
    # take into account the way the labels were generated (with all feature maps
    # being concatenated as well)
    box_cls = cat(box_cls_flattened, dim=1).view(-1, num_classes)
    box_delta = cat(box_delta_flattened, dim=1).view(-1, 4)
    return box_cls, box_delta


@torch.no_grad()
def anchor_matching_nms(self, anchors, targets, box_cls):
    """
    This function matches each anchor with ground-truth labels based on the IoU; also returns a bool array to indicate anchors with highest probability for each ground-truth instances.

    Args:
    -- self: retinanet model
    -- anchors (list[Boxes]): A list of #feature level Boxes. The Boxes contains anchors of this image on the specific feature level.
    -- targets (list[Instances]): a list of N `Instances`s. The i-th`Instances` contains the ground-truth per-instance annotations for the i-th input image.  Specify `targets` during training only.
    -- box_cls: cls results.

    Returns:
    -- gt_classes (Tensor): An integer tensor of shape (N, R) storing ground-truth labels for each anchor.
        R is the total number of anchors, i.e. the sum of Hi x Wi x A for all levels.
        Anchors with an IoU with some target higher than the foreground threshold
        are assigned their corresponding label in the [0, K-1] range.
        Anchors whose IoU are below the background threshold are assigned
        the label "K". Anchors whose IoU are between the foreground and background
        thresholds are assigned a label "-1", i.e. ignore.
    -- gt_anchors_deltas (Tensor): Shape (N, R, 4).
        The last dimension represents ground-truth box2box transform
        targets (dx, dy, dw, dh) that map each anchor to its matched ground-truth box.
        The values in the tensor are meaningful only when the corresponding
        anchor is labeled as foreground.
    -- keep_nms (Tensor): maximum activated anchors.
    """
    gt_classes = []
    gt_anchors_deltas = []
    keep_nms_list = []
    anchors = Boxes.cat(anchors)  # Rx4

    box_cls = [permute_to_N_HWA_K(x, self.num_classes) for x in box_cls]

    for img_idx, targets_per_image in enumerate(targets):
        match_quality_matrix = pairwise_iou(targets_per_image.gt_boxes, anchors)
        gt_matched_idxs, anchor_labels = self.matcher(match_quality_matrix)

        box_cls_per_image = [box_cls_per_level[img_idx] for box_cls_per_level in box_cls]
        box_cls_per_image = torch.cat(box_cls_per_image, dim=0)
        keep_nms = torch.zeros_like(box_cls_per_image).sum(dim=1)
        has_gt = len(targets_per_image) > 0
        if has_gt:
            # ground truth box regression
            matched_gt_boxes = targets_per_image.gt_boxes[gt_matched_idxs]
            gt_anchors_reg_deltas_i = self.box2box_transform.get_deltas(
                anchors.tensor, matched_gt_boxes.tensor
            )

            gt_classes_i = targets_per_image.gt_classes[gt_matched_idxs]
            # Anchors with label 0 are treated as background.
            gt_classes_i[anchor_labels == 0] = self.num_classes
            # Anchors with label -1 are ignored.
            gt_classes_i[anchor_labels == -1] = -1

            for instance_idxs in range(len(targets_per_image.gt_classes)):
                valid_idx = ((gt_matched_idxs == instance_idxs) & (anchor_labels == 1))
                if len(box_cls_per_image[valid_idx, gt_classes_i[valid_idx]]) == 0:
                    continue
                max_id = torch.argmax(box_cls_per_image[valid_idx, gt_classes_i[valid_idx]])
                keep_id = torch.where(valid_idx)[0]
                keep_id = keep_id[max_id]
                keep_nms[keep_id] = 1
            keep_nms = (keep_nms == 1)
        else:
            gt_classes_i = torch.zeros_like(gt_matched_idxs) + self.num_classes
            gt_anchors_reg_deltas_i = torch.zeros_like(anchors.tensor)

        gt_classes.append(gt_classes_i)
        gt_anchors_deltas.append(gt_anchors_reg_deltas_i)
        keep_nms_list.append(keep_nms)

    return torch.stack(gt_classes), torch.stack(gt_anchors_deltas), torch.stack(keep_nms_list)


def pre_sigmoid_loss(self, box_cls, box_delta, selected_anchor, selected_anchor_class, scale_weight):
    """
    Calculate the pre-sigmoid loss for classification.

    Args:
    -- self: RetinaNet model.
    -- box_cls: classification results from RetinaNet.
    -- box_delta: box regression delta results from RetinaNet.
    -- seletecd_anchor: bool list, True indicates the anchor is selected.
    -- selected_anchor_class: the ground-truth class for selected anchor.
    -- scale_weight: scale weight for all anchors.

    Returns:
    -- fore_loss: the pre-sigmoid loss.
    """
    weight_flatten = [torch.ones((permute_to_N_HWA_K(x, self.num_classes)).shape[0:2]).to(selected_anchor_class.device)*scale_weight[i] for i, x in enumerate(box_cls)]
    weight_flatten = torch.cat(weight_flatten, dim=1).view(-1, 1)
    pred_class_logits, pred_anchor_deltas = permute_all_cls_and_box_to_N_HWA_K_and_concat(box_cls, box_delta, self.num_classes)  # Shapes: (N x R, K) and (N x R, 4), respectively.
    selected_anchor = selected_anchor.flatten()

    fore_loss = torch.sum((pred_class_logits * weight_flatten)[selected_anchor, selected_anchor_class]) / max(1, self.loss_normalizer)
    return fore_loss


def get_pseudo_label_simple(self, anchors, box_cls_batch, box_delta_batch, gt_instances, scale_weight, enforce_back=False, back_thre=0.3, fore_thre=0.7, IOU_thre=0.5):
    pass


def get_pseudo_label(self, anchors, box_cls_batch, box_delta_batch, gt_instances, scale_weight, enforce_back=False, back_thre=0.3, fore_thre=0.7, IOU_thre=0.5):
    """
    Calculate the pseudo-label based on current predictions results.
    This function contains 3 steps:

    Args:
    -- self: Retinanet model.
    -- anchors (list[Boxes]): A list of #feature level Boxes. The Boxes contains anchors of this image on the specific feature level.
    -- box_cls_batch: cls results from RetinaNet.
    -- box_delta_batch: box regression delta results from RetinaNet.
    -- gt_instances: ground-truth instances.
    -- scale_weight: list, the weight for loss from different scale.
    -- enforce_back: bool, whether apply extra constraints on background detections.
    -- back_thre: IoU threshold for background detections. If IoU of detected regions and gt-instances is smaller than this threshold, it is regarded as background.
    -- fore_thre: IoU threshold for foreground detections.
    -- IoU_thre: NMS IoU threshold.

    Returns:
    -- pred_logits: classification prediction results from selected detections.
    -- pred_boxes: box regression prediction results from selected detections.
    -- pseudo_target_logits: classification predictions pseudo targets.
    -- pseudo_target_boxes: boxes regression pseudo targets.
    -- weight_logits: logits prediction scale weights.
    -- weight_boxes: boxes prediction scale weights.
    """
    with torch.no_grad():
        anchors = type(anchors[0]).cat(anchors).tensor
        device = anchors.device
        N = len(gt_instances)
        weight_flatten = [torch.ones((permute_to_N_HWA_K(x, self.num_classes)).shape[0:2]).to(device)*scale_weight[i] for i, x in enumerate(box_cls_batch)]
        weight_flatten = torch.cat(weight_flatten, dim=1).view(-1)
        pred_logits_collect = []
        pred_boxes_collect = []
        pseudo_target_logits_collect = []
        pseudo_target_boxes_collect = []
        weight_logits_collect = []
        weight_boxes_collect = []
    # For each image in the batch:
    for i in range(N):
        # Aggregate box_cls and box_delta for each scale.
        box_cls = [box_cls[i:i+1] for box_cls in box_cls_batch]
        box_delta = [box_delta[i:i+1] for box_delta in box_delta_batch]
        pred_class_logits, pred_anchor_deltas = permute_all_cls_and_box_to_N_HWA_K_and_concat(box_cls, box_delta, self.num_classes)  # Shapes: (N x R, K) and (N x R, 4), respectively.
        pred_box = self.box2box_transform.apply_deltas(pred_anchor_deltas, anchors)
        gt_boxes = gt_instances[i].gt_boxes
        gt_labels = gt_instances[i].gt_classes
        # Initial the pseudo_targets
        with torch.no_grad():
            pseudo_target_logits = pred_class_logits.clone().to(pred_class_logits.device)
            pseudo_target_logits = pseudo_target_logits.sigmoid()
            pseudo_target_boxes = pred_box.clone().to(pred_box.device)
        # Step 1: For each object, assgin groud-truth to the predicted boxes of the highest IoU, to prevent the case that there are missing detections.
        # For convenience, we use Matcher provided by D2 to achieve this step. We use a high fore_thre to get the highest IoU match.
        matcher = Matcher([back_thre, fore_thre], [-1, 0, 1], allow_low_quality_matches=True)
        with torch.no_grad():
            match_quality_matrix = pairwise_iou(gt_boxes, Boxes(anchors))
            matched_idxs, anchor_labels = matcher(match_quality_matrix)
            del match_quality_matrix
            # Assign groud-truth predictions to the selected anchors.
            selected_anchor = anchor_labels == 1
            pseudo_target_logits[selected_anchor] = 0
            pseudo_target_logits[selected_anchor, gt_labels[matched_idxs[selected_anchor]]] = 1
            pseudo_target_boxes[selected_anchor] = gt_boxes.tensor[matched_idxs[selected_anchor]]
            # If enforce_back is enabled, background-anchors are also included in the pseudo-labels.
            # background-anchors are anchors which are far away from any objects.
            # By enableing enforce_back, we enforce the background-anchors to detect nothing.
            if enforce_back:
                background_idxs = anchor_labels == -1
                pseudo_target_logits[background_idxs] = 0
                pseudo_back_logits = pseudo_target_logits[background_idxs].clone().to(pseudo_target_logits.device)
                pred_class_back_logits = pred_class_logits[background_idxs]
                weight_back_logits = weight_flatten[background_idxs]

        # Step 2: Conduct NMS process, filter out eliminated dectections.
        # Only apply constraints on detections kept after NMS.
        logits_sigmoid = pseudo_target_logits.flatten()
        num_topk = min(self.topk_candidates, pseudo_target_boxes.size(0))
        predicted_prob, topk_idxs = logits_sigmoid.sort(descending=True)
        predicted_prob = predicted_prob[: num_topk]
        topk_idxs = topk_idxs[:num_topk]
        keep_idxs = predicted_prob > self.score_threshold
        predicted_prob = predicted_prob[keep_idxs]
        topk_idxs = topk_idxs[keep_idxs]
        anchor_idxs = topk_idxs // self.num_classes

        pseudo_target_logits = pseudo_target_logits[anchor_idxs]
        pseudo_target_boxes = pseudo_target_boxes[anchor_idxs]
        pred_box = pred_box[anchor_idxs]
        pred_class_logits = pred_class_logits[anchor_idxs]
        weight_logits = weight_flatten[anchor_idxs]
        weight_boxes = weight_flatten[anchor_idxs]
        gt_labels = gt_instances[i].gt_classes

        # Step 3: Match the rest detections with the ground-truth objects and assign pseudo-targets based on the matching.
        # If IoU > IOU_thre, assign ground-truth cls and box as the target.
        # Else, assign background as targets.
        matcher = Matcher([IOU_thre], [0, 1], allow_low_quality_matches=False)

        match_quality_matrix = pairwise_iou(gt_boxes, Boxes(pseudo_target_boxes))
        matched_idxs, anchor_labels = matcher(match_quality_matrix)
        del match_quality_matrix

        target = torch.zeros(((anchor_labels == 1).sum(), 80), dtype=pred_box.dtype, device=pred_box.device)
        target[torch.arange((anchor_labels == 1).sum()), gt_labels[matched_idxs[anchor_labels == 1]]] = 1.0
        pseudo_target_logits[anchor_labels == 1] = target
        pseudo_target_boxes[anchor_labels == 1] = gt_boxes.tensor[matched_idxs[anchor_labels == 1]]
        pseudo_target_boxes = pseudo_target_boxes[anchor_labels == 1]
        pred_box = pred_box[anchor_labels == 1]
        pseudo_target_logits[anchor_labels == 0] = 0
        weight_boxes = weight_boxes[anchor_labels == 1]
        if enforce_back:
            pseudo_target_logits = torch.cat([pseudo_back_logits, pseudo_target_logits], dim=0)
            pred_class_logits = torch.cat([pred_class_back_logits, pred_class_logits], dim=0)
            weight_logits = torch.cat([weight_back_logits, weight_logits], dim=0)
        pseudo_target_boxes_collect.append(pseudo_target_boxes)
        pseudo_target_logits_collect.append(pseudo_target_logits)
        pred_boxes_collect.append(pred_box)
        pred_logits_collect.append(pred_class_logits)
        weight_logits_collect.append(weight_logits)
        weight_boxes_collect.append(weight_boxes)
    return torch.cat(pred_logits_collect), torch.cat(pred_boxes_collect), torch.cat(pseudo_target_logits_collect), torch.cat(pseudo_target_boxes_collect), torch.cat(weight_logits_collect), torch.cat(weight_boxes_collect)


def retinanet_vis_loss(self, anchors, gt_instances, scale_weight, box_cls, box_delta, **kwargs):
    """
    Calculate the visualization loss for retinaNet.

    Args:
    -- self: RetinaNet model.
    -- anchors (list[Boxes]): A list of #feature level Boxes. The Boxes contains anchors of this image on the specific feature level.
    -- gt_instances: ground-truth instances.
    -- scale_weight: list, the weight for loss from different scale.
    -- box_cls, box_delta: outputs of retinanet head.

    Returns:
    -- losses: dict.
    """
    with torch.no_grad():
        anchor_label, _, keep_index = self.anchor_matching_nms(anchors, gt_instances, box_cls)
    pred_logits, pred_boxes, pseudo_logits, pseudo_boxes, weight_logits, weight_boxes = self.get_pseudo_label(anchors, box_cls, box_delta, gt_instances, scale_weight, **kwargs)

    cls_loss = sigmoid_focal_loss_jit(pred_logits, pseudo_logits, alpha=self.focal_loss_alpha, gamma=self.focal_loss_gamma, reduction='None')
    cls_loss = torch.sum(cls_loss * weight_logits.unsqueeze(1))/pred_logits.shape[0]
    reg_loss = giou_loss(pred_boxes, pseudo_boxes, reduction='None')
    reg_loss = torch.sum(reg_loss * weight_boxes)/pred_boxes.shape[0]
    fore_loss = self.pre_sigmoid_loss(box_cls, box_delta, keep_index, anchor_label[keep_index], scale_weight)
    return {'cls_loss': cls_loss, 'reg_loss': reg_loss, 'fore_loss': fore_loss}


def retinanet_vis(self, x, gt_instances, scale_weight=None, **kwargs):
    """
    Calculate the loss in retinaNet

    Args:
    -- self: retinanet model.
    -- x: input tensor, the same size as images.tensor.
    -- gt_instances: list[Instances], layout information.
    -- scale_weight: list, the weight for loss from different scale.

    Returns:
    -- losses: dict.
    """

    features = self.backbone(x)
    features = [features[f] for f in self.in_features]

    box_cls, box_delta = self.head(features)
    anchors = self.anchor_generator(features)
    return self.vis_loss(anchors, gt_instances, scale_weight, box_cls, box_delta, **kwargs)


def version_check(self):
    """
    Check if there are attributions missing due to different version.
    """
    # anchor_matcher --> matcher
    if hasattr(self, "anchor_matcher"):
        self.matcher = self.anchor_matcher
    if hasattr(self, "head_in_features"):
        self.in_features = self.head_in_features
    if hasattr(self, "test_topk_candidates"):
        self.topk_candidates = self.test_topk_candidates
    if hasattr(self, "test_score_thresh"):
        self.score_threshold = self.test_score_thresh


def warp_retina(model):
    """
    Warp the RetinaNet with visualization functions.

    Args:
    -- model: retinaNet model from D2.

    Returns:
    -- warp_model: model with visualization function.
    """

    warp_model = copy.deepcopy(model)
    warp_model.version_check = types.MethodType(version_check, warp_model)
    warp_model.anchor_matching_nms = types.MethodType(anchor_matching_nms, warp_model)
    warp_model.get_pseudo_label = types.MethodType(get_pseudo_label, warp_model)
    warp_model.pre_sigmoid_loss = types.MethodType(pre_sigmoid_loss, warp_model)
    warp_model.vis = types.MethodType(retinanet_vis, warp_model)
    warp_model.vis_loss = types.MethodType(retinanet_vis_loss, warp_model)
    warp_model.version_check()
    return warp_model
