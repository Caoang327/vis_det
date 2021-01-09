import types
from fvcore.nn import smooth_l1_loss
import torch
import torch.nn.functional as F
from detectron2.layers import cat
from detectron2.structures import pairwise_iou
from detectron2.modeling.roi_heads.fast_rcnn import FastRCNNOutputs
from detectron2.modeling.roi_heads import select_foreground_proposals
from detectron2.modeling.proposal_generator.proposal_utils import add_ground_truth_to_proposals
from .rcnn_helper import warp_FastRCNNOutputs, get_level_assignments, get_gt_proposals, mask_rcnn_loss_vis
# from detectron2.structures import Boxes, ImageList, Instances, pairwise_iou


def rpn_vis_losses(
    self,
    anchors,
    pred_objectness_logits,
    gt_labels,
    pred_anchor_deltas,
    gt_boxes,
    scale_weight
):
    """
    Return the visualization losses from a set of RPN predictions and their associated ground-truth. Warp to RPN model.

    Args:
    -- self: rpn model.
    -- anchors: (list[Boxes or RotatedBoxes]), anchors for each feature map, each has shape (Hi*Wi*A, B), where B is box dimension (4 or 5).
    -- pred_objectness_logits: (list[Tensor]), A list of L elements. Element i is a tensor of shape (N, Hi*Wi*A) representing the predicted objectness logits for all anchors.
    -- gt_labels: (list[Tensor]), Output of :meth:`label_and_sample_anchors`. pred_anchor_deltas (list[Tensor]): A list of L elements. Element i is a tensor of shape
        (N, Hi*Wi*A, 4 or 5) representing the predicted "deltas" used to transform anchors to proposals.
    -- gt_boxes (list[Boxes or RotatedBoxes]): Output of :meth:`label_and_sample_anchors`.
    -- scale_weight: weights for losses of each anchor in different scales, with the same shape as pred_objectness_logits

    Returns:
    -- dict[loss name -> loss value]: A dict mapping from loss name to loss value. Loss names are: `loss_rpn_cls` for objectness classification and `loss_rpn_loc` for proposal localization.
    """

    num_images = len(gt_labels)
    gt_labels = torch.stack(gt_labels)  # (N, sum(Hi*Wi*Ai))

    anchors = type(anchors[0]).cat(anchors).tensor  # Ax(4 or 5)
    gt_anchor_deltas = [self.box2box_transform.get_deltas(anchors, k) for k in gt_boxes]
    gt_anchor_deltas = torch.stack(gt_anchor_deltas)  # (N, sum(Hi*Wi*Ai), 4 or 5)

    neg_mask = gt_labels == 0
    valid_mask = gt_labels >= 0
    pos_mask = gt_labels == 1
    weight = cat(scale_weight, dim=1)

    localization_loss = smooth_l1_loss(
        cat(pred_anchor_deltas, dim=1)[pos_mask],
        gt_anchor_deltas[pos_mask],
        self.smooth_l1_beta,
        reduction="none",
    )
    localization_loss = torch.sum(localization_loss * weight[pos_mask].unsqueeze(-1))
    objectness_loss = F.binary_cross_entropy_with_logits(
        cat(pred_objectness_logits, dim=1)[valid_mask],
        gt_labels[valid_mask].to(torch.float32),
        reduction="none",
    )
    objectness_loss = torch.sum(weight[valid_mask] * objectness_loss)
    normalizer = self.batch_size_per_image * num_images

    loss_pos = torch.sum(cat(pred_objectness_logits, dim=1)[pos_mask] * weight[pos_mask])
    loss_neg = torch.sum(cat(pred_objectness_logits, dim=1)[neg_mask] * weight[neg_mask])
    return {
        "loss_rpn_cls": objectness_loss / normalizer,
        "loss_rpn_loc": localization_loss / normalizer,
        "loss_rpn_pos": loss_pos / normalizer,
        "loss_rpn_neg": loss_neg / normalizer
    }


def rpn_vis_forward(self, images, features, gt_instances, scale_weight=None):
    """
    Calculate the vis losses of rpn given features and annotations. Warp to RPN model.

    Args:
    -- self: rpn model.
    -- images: ImageList, input images of length `N`.
    -- features: (dict[str, Tensor]): input data as a mapping from feature map name to tensor. Axis 0 represents the number of images `N` in
        the input data; axes 1-3 are channels, height, and width, which may vary between feature maps (e.g., if a feature pyramid is used).
    -- gt_instances: list[Instances], layout information.
    -- scale_weight: list, the weight for loss from different scale.

    Returns:
    -- proposals: rpn proposals.
    -- losses: rpn loss.
    """

    features = [features[f] for f in self.in_features]
    anchors = self.anchor_generator(features)

    pred_objectness_logits, pred_anchor_deltas = self.rpn_head(features)
    # Transpose the Hi*Wi*A dimension to the middle:
    pred_objectness_logits = [
        # (N, A, Hi, Wi) -> (N, Hi, Wi, A) -> (N, Hi*Wi*A)
        score.permute(0, 2, 3, 1).flatten(1)
        for score in pred_objectness_logits
    ]
    pred_anchor_deltas = [
        # (N, A*B, Hi, Wi) -> (N, A, B, Hi, Wi) -> (N, Hi, Wi, A, B) -> (N, Hi*Wi*A, B)
        x.view(x.shape[0], -1, self.anchor_generator.box_dim, x.shape[-2], x.shape[-1])
        .permute(0, 3, 4, 1, 2)
        .flatten(1, -2)
        for x in pred_anchor_deltas
    ]

    weight = [torch.ones_like(x) * scale_weight[i] for i, x in enumerate(pred_objectness_logits)]
    gt_labels, gt_boxes = self.label_and_sample_anchors(anchors, gt_instances)
    losses = self.rpn_vis_losses(anchors, pred_objectness_logits, gt_labels, pred_anchor_deltas, gt_boxes, weight)

    losses = {k: v * self.loss_weight.get(k, 1.0) for k, v in losses.items()}
    proposals = self.predict_proposals(
        anchors, pred_objectness_logits, pred_anchor_deltas, images.image_sizes
    )
    return losses, proposals


def rpn_version_check(self):
    """
    Check if there are version conflicts and missing attributions. Warp this function to rpn model.
    """


@torch.no_grad()
def label_proposal_pseudo_targets(self, proposals, targets):
    """
    Label the pseudo-targets for some proposals.
    It performs box matching between `proposals` and `targets`, and assigns training labels to the proposals.
    If sample is True, it returns ``self.batch_size_per_image`` random samples with a fraction of positives that is no larger than
    ``self.positive_fraction``.
    Modified from the original D2 code.

    Args:
    -- self: RoI head model.
    -- proposals: RPN proposals.
    -- targets: ground-truth instances.
    -- sample: whether sample proposals.

    Returns:
    - proposal_boxes: the proposal boxes
    - gt_boxes: the ground-truth box that the proposal is assigned to.
    """
    gt_boxes = [x.gt_boxes for x in targets]

    if self.proposal_append_gt:
        proposals = add_ground_truth_to_proposals(gt_boxes, proposals)

    proposals_with_gt = []

    num_fg_samples = []
    num_bg_samples = []
    for proposals_per_image, targets_per_image in zip(proposals, targets):
        has_gt = len(targets_per_image) > 0
        match_quality_matrix = pairwise_iou(
            targets_per_image.gt_boxes, proposals_per_image.proposal_boxes
        )
        matched_idxs, matched_labels = self.proposal_matcher(match_quality_matrix)
        sampled_idxs, gt_classes = self._sample_proposals(
            matched_idxs, matched_labels, targets_per_image.gt_classes
        )

        # Set target attributes of the sampled proposals:
        proposals_per_image = proposals_per_image[sampled_idxs]
        proposals_per_image.gt_classes = gt_classes

        if has_gt:
            sampled_targets = matched_idxs[sampled_idxs]

            for (trg_name, trg_value) in targets_per_image.get_fields().items():
                if trg_name.startswith("gt_") and not proposals_per_image.has(trg_name):
                    proposals_per_image.set(trg_name, trg_value[sampled_targets])

        num_bg_samples.append((gt_classes == self.num_classes).sum().item())
        num_fg_samples.append(gt_classes.numel() - num_bg_samples[-1])
        proposals_with_gt.append(proposals_per_image)

    return proposals_with_gt


def roi_vis_box_loss(self, features, gt_instances, proposals, scale_weight=None):
    """
    Calculate the regression loss and classification loss for visualization. Warp this function to roi heads.

    Args:
    -- self: RoI head of RCNN.
    -- features: (dict[str, Tensor]), input data as a mapping from feature map name to tensor. Axis 0 represents the number of images `N` in
        the input data; axes 1-3 are channels, height, and width, which may vary between feature maps (e.g., if a feature pyramid is used).
    -- gt_instances: list[Instances], layout information.
    -- proposals: proposed results from rpn of rcnn.
    -- scale_weight: list, the weight for loss from different scale.

    Returns:
    -- losses.
    """
    # proposals = self.label_proposal_pseudo_targets(proposals, gt_instances)

    proposals = self.label_proposal_pseudo_targets(proposals, gt_instances)
    features = [features[f] for f in self.box_in_features]
    box_features = self.box_pooler(features, [x.proposal_boxes for x in proposals])
    box_features = self.box_head(box_features)
    level_assign = get_level_assignments(self.box_pooler, features, [x.proposal_boxes for x in proposals])
    assign_weight = torch.gather(torch.tensor(scale_weight), dim=0, index=level_assign.type(torch.LongTensor)).to(features[0].device)
    pred_class_logits, pred_proposal_deltas = self.box_predictor(box_features)
    del box_features

    fastrcnn = FastRCNNOutputs(
        self.box_predictor.box2box_transform,
        pred_class_logits,
        pred_proposal_deltas,
        proposals,
        self.box_predictor.smooth_l1_beta
    )
    fastrcnn = warp_FastRCNNOutputs(fastrcnn)  # warp the loss function to FastRCNNOutputs.
    losses = fastrcnn.vis_FasterRCNN_loss(assign_weight)
    return losses


def roi_vis_box_C4_loss(self, features, gt_instances, proposals):
    """
    Calculate the regression loss and classification loss for visualization. Warp this function to roi heads.

    Args:
    -- self: RoI head of RCNN.
    -- features: (dict[str, Tensor]), input data as a mapping from feature map name to tensor. Axis 0 represents the number of images `N` in
        the input data; axes 1-3 are channels, height, and width, which may vary between feature maps (e.g., if a feature pyramid is used).
    -- gt_instances: list[Instances], layout information.
    -- proposals: proposed results from rpn of rcnn.

    Returns:
    -- losses.
    """
    proposals = self.label_proposal_pseudo_targets(proposals, gt_instances)

    proposal_boxes = [x.proposal_boxes for x in proposals]
    box_features = self._shared_roi_transform(
            [features[f] for f in self.in_features], proposal_boxes)
    pred_class_logits, pred_proposal_deltas = self.box_predictor(box_features.mean(dim=[2, 3]))
    del box_features

    fastrcnn = FastRCNNOutputs(
        self.box_predictor.box2box_transform,
        pred_class_logits,
        pred_proposal_deltas,
        proposals,
        self.box_predictor.smooth_l1_beta
    )
    fastrcnn = warp_FastRCNNOutputs(fastrcnn)  # warp the loss function to FastRCNNOutputs.
    losses = fastrcnn.losses()
    return losses


def roi_vis_mask_loss(self, features, instances, scale_weight=None):
    """
    Calculate the mask loss for visualization. Warp this function to roi heads.

    Args:
    -- self: RoI head of RCNN.
    -- instances: per-image proposals to compute the mask predictions.
    -- scale_weight: list, the weight for loss from different scale.

    Returns:
    -- losses. loss_mask.
    """

    if not self.mask_on:
        return {}
    features = [features[f] for f in self.mask_in_features]

    proposals, _ = select_foreground_proposals(instances, self.num_classes)

    proposal_boxes = [x.proposal_boxes for x in proposals]
    level_assign = get_level_assignments(self.mask_pooler, features, proposal_boxes)
    assign_weight = torch.gather(torch.tensor(scale_weight), dim=0, index=level_assign.type(torch.LongTensor)).to(features[0].device)
    mask_features = self.mask_pooler(features, proposal_boxes)
    x = self.mask_head.layers(mask_features)
    return {"loss_mask": mask_rcnn_loss_vis(x, instances, assign_weight)}


def roi_vis_pre_sigmoid(self, features, proposals, scale_weight=None):
    """
    Calculate the pre-sigmoid classification loss for visualization. Warp this function to roi heads.

    Args:
    -- self: RoI head of RCNN.
    -- features: (dict[str, Tensor]): input data as a mapping from feature map name to tensor. Axis 0 represents the number of images `N` in
        the input data; axes 1-3 are channels, height, and width, which may vary between feature maps (e.g., if a feature pyramid is used).
    -- proposals: proposed results from rpn of rcnn.
    -- scale_weight: list, the weight for loss from different scale.

    Returns:
    -- losses. loss_pos_scores, loss_neg_scores.
    """

    features = [features[f] for f in self.box_in_features]
    box_features = self.box_pooler(features, [x.proposal_boxes for x in proposals])
    box_features = self.box_head(box_features)
    level_assign = get_level_assignments(self.box_pooler, features, [x.proposal_boxes for x in proposals])
    assign_weight = torch.gather(torch.tensor(scale_weight), dim=0, index=level_assign.type(torch.LongTensor)).to(features[0].device)
    pred_class_logits, _ = self.box_predictor(box_features)

    gt_classes = cat([p.gt_classes for p in proposals], dim=0)
    neg_idx = gt_classes == self.num_classes
    pos_idx = gt_classes < self.num_classes
    pos_score = pred_class_logits[pos_idx, gt_classes[pos_idx]] * assign_weight[pos_idx]
    neg_score = pred_class_logits[neg_idx, self.num_classes] * assign_weight[neg_idx]
    losses = {'loss_pos_scores': torch.sum(pos_score)/100.0, 'loss_neg_scores': torch.sum(neg_score)/100.0}
    return losses


def roi_vis_pre_sigmoid_C4(self, features, proposals):
    """
    Calculate the pre-sigmoid classification loss for visualization. Warp this function to roi heads.

    Args:
    -- self: RoI head of RCNN.
    -- features: (dict[str, Tensor]): input data as a mapping from feature map name to tensor. Axis 0 represents the number of images `N` in
        the input data; axes 1-3 are channels, height, and width, which may vary between feature maps (e.g., if a feature pyramid is used).
    -- proposals: proposed results from rpn of rcnn.
    -- scale_weight: list, the weight for loss from different scale.

    Returns:
    -- losses. loss_pos_scores, loss_neg_scores.
    """

    proposal_boxes = [x.proposal_boxes for x in proposals]
    box_features = self._shared_roi_transform(
            [features[f] for f in self.in_features], proposal_boxes)
    pred_class_logits, pred_proposal_deltas = self.box_predictor(box_features.mean(dim=[2, 3]))
    del box_features

    gt_classes = cat([p.gt_classes for p in proposals], dim=0)
    neg_idx = gt_classes == self.num_classes
    pos_idx = gt_classes < self.num_classes
    pos_score = pred_class_logits[pos_idx, gt_classes[pos_idx]]
    neg_score = pred_class_logits[neg_idx, self.num_classes]
    losses = {'loss_pos_scores': torch.sum(pos_score)/100.0, 'loss_neg_scores': torch.sum(neg_score)/100.0}
    return losses


def roi_vis_forward(self, features, gt_instances, proposals, scale_weight=None):
    """
    Calculate the vis losses of roi head. warped to RoI submodule of RCNN. Warp this to roi heads.

    Args:
    -- self: RoI head of RCNN.
    -- features: (dict[str, Tensor]): input data as a mapping from feature map name to tensor. Axis 0 represents the number of images `N` in
        the input data; axes 1-3 are channels, height, and width, which may vary between feature maps (e.g., if a feature pyramid is used).
    -- gt_instances: list[Instances], layout information.
    -- proposals: proposed results from rpn of rcnn.
    -- scale_weight: list, the weight for loss from different scale.

    Returns:
    -- losses.
    """

    # compute the box loss
    # print('loss')
    losses = self.roi_vis_box_loss(features, gt_instances, proposals, scale_weight)
    # print("vis_box_loss")

    # we compute the pre-sigmoid loss and mask loss for gt proposals only
    gt_proposals = get_gt_proposals(self.mask_on, gt_instances)
    # print("gt_proposals")
    losses.update(self.roi_vis_mask_loss(features, gt_proposals, scale_weight))
    # print("Mask Loss")
    losses.update(self.roi_vis_pre_sigmoid(features, gt_proposals, scale_weight))
    # print("pre_sigmoid")
    return losses


def roi_vis_forward_C4(self, features, gt_instances, proposals):
    """
    Calculate the vis losses of roi head. warped to RoI submodule of RCNN. Warp this to roi heads.

    Args:
    -- self: RoI head of RCNN.
    -- features: (dict[str, Tensor]): input data as a mapping from feature map name to tensor. Axis 0 represents the number of images `N` in
        the input data; axes 1-3 are channels, height, and width, which may vary between feature maps (e.g., if a feature pyramid is used).
    -- gt_instances: list[Instances], layout information.
    -- proposals: proposed results from rpn of rcnn.
    -- scale_weight: list, the weight for loss from different scale.

    Returns:
    -- losses.
    """

    # compute the box loss
    losses = self.roi_vis_box_c4(features, gt_instances, proposals)

    # we compute the pre-sigmoid loss and mask loss for gt proposals only
    gt_proposals = get_gt_proposals(self.mask_on, gt_instances)

    losses.update(self.roi_vis_pre_sigmoid_C4(features, gt_proposals))

    return losses


def roi_version_check(self):
    """
    Check if there are version conflicts and missing attributions.
    """


def rcnn_vis(self, x, gt_instances, images, scale_weight=None):
    """
    rcnn visualization.Warp this to RCNN model.

    Args:
    -- self: RCNN model.
    -- x: input tensor, the same size as images.tensor.
    -- gt_instances: list[Instances], layout information.
    -- images: Image list after preprocessing
    -- scale_weight: list, the weight for loss from different scale.

    Returns:
    -- losses:
    """

    features = self.backbone(x)

    losses, proposals = self.proposal_generator.rpn_vis_forward(images, features, gt_instances, scale_weight)

    losses.update(self.roi_heads.roi_vis_forward(features, gt_instances, proposals, scale_weight))

    return losses


def rcnn_version_check(self):
    """
    Check if there are version conflicts and missing attributions.
    """


def warp_rpn(model):
    """
    Warp visualize model to rpn.

    Args:
    -- model: RPN model.

    Returns:
    -- model: warped model.
    """

    model.version_check = types.MethodType(rpn_version_check, model)
    model.rpn_vis_losses = types.MethodType(rpn_vis_losses, model)
    model.rpn_vis_forward = types.MethodType(rpn_vis_forward, model)
    model.version_check()
    return model


def warp_roi(model):
    """
    Warp visualize model to roi.

    Args:
    -- model: RoI model.

    Returns:
    -- model: warped model.
    """

    model.version_check = types.MethodType(roi_version_check, model)
    model.label_proposal_pseudo_targets = types.MethodType(label_proposal_pseudo_targets, model)
    model.roi_vis_mask_loss = types.MethodType(roi_vis_mask_loss, model)
    model.roi_vis_box_loss = types.MethodType(roi_vis_box_loss, model)
    model.roi_vis_pre_sigmoid = types.MethodType(roi_vis_pre_sigmoid, model)
    model.roi_vis_forward = types.MethodType(roi_vis_forward, model)
    model.roi_vis_box_c4 = types.MethodType(roi_vis_box_C4_loss, model)
    model.roi_vis_pre_sigmoid_C4 = types.MethodType(roi_vis_pre_sigmoid_C4, model)
    model.roi_vis_forward_C4 = types.MethodType(roi_vis_forward_C4, model)
    model.version_check()
    return model


def warp_rcnn(model):
    """
    Warp visualize model to rcnn.

    Args:
    -- model: rcnn model.

    Returns:
    -- model. warped model.
    """
    model.version_check = types.MethodType(rcnn_version_check, model)
    model.proposal_generator = warp_rpn(model.proposal_generator)
    model.roi_heads = warp_roi(model.roi_heads)
    model.vis = types.MethodType(rcnn_vis, model)
    return model
