import math
import types
from fvcore.nn import smooth_l1_loss
import torch
import torch.nn.functional as F
from detectron2.structures import Instances
from detectron2.modeling.poolers import assign_boxes_to_levels
from detectron2.layers import nonzero_tuple, cat

__all__ = ["get_gt_proposals", "get_level_assignments", "warp_FastRCNNOutputs", "mask_rcnn_loss_vis"]


def get_gt_proposals_single(mask_on, gt_instances):
    """
    Helper function, get the ground-truth instances as the proposals from each gt_instance.

    Args:
    -- mask_on: if mask_on in this model.
    -- gt_instances: ground truth object instances.

    Returns:
    -- gt_proposal: ground-truth proposals.

    """
    device = gt_instances[0].gt_classes.device
    gt_proposal = Instances(gt_instances.image_size)
    gt_logit_value = math.log((1.0 - 1e-10) / (1 - (1.0 - 1e-10)))
    gt_logits = gt_logit_value * torch.ones(len(gt_instances), device=device)
    gt_proposal.proposal_boxes = gt_instances.gt_boxes
    gt_proposal.objectness_logits = gt_logits
    gt_proposal.gt_boxes = gt_instances.gt_boxes
    gt_proposal.gt_classes = gt_instances.gt_classes
    if mask_on:
        gt_proposal.gt_masks = gt_instances.gt_masks
    return gt_proposal


def get_gt_proposals(mask_on, gt_instances):
    """
    Helper function, get the ground-truth instances as the proposals from gt_instances list.

    Args:
    -- mask_on: if mask_on in this model.
    -- gt_instances: ground truth object instances.

    Returns:
    -- gt_proposal: list.
    """
    return [get_gt_proposals_single(mask_on, gt_instance) for gt_instance in gt_instances]


def get_level_assignments(box_pooler, x, box_lists):
    """
    Helper function, get the level assignments for each proposals.

    Args:
    -- box_pooler: RoI pooler.
    -- x: list[Tensor], a list of feature maps of NCHW shape, with scales matching those used to construct this module.
    -- box_lists: list[Boxes] | list[RotatedBoxes], a list of N Boxes or N RotatedBoxes, where N is the number of images in the batch.
        The box coordinates are defined on the original image and will be scaled by the `scales` argument of :class:`ROIPooler`.
    Returns:
    -- Tensor, level assignments.
    """

    num_level_assignments = len(box_pooler.level_poolers)

    assert isinstance(x, list) and isinstance(
            box_lists, list
        ), "Arguments to pooler must be lists"
    assert (
            len(x) == num_level_assignments
        ), "unequal value, num_level_assignments={}, but x is list of {} Tensors".format(
            num_level_assignments, len(x)
        )

    assert len(box_lists) == x[0].size(
        0
    ), "unequal value, x[0] batch dim 0 is {}, but box_list has length {}".format(
        x[0].size(0), len(box_lists)
    )

    level_assignments = assign_boxes_to_levels(
        box_lists, box_pooler.min_level, box_pooler.max_level, box_pooler.canonical_box_size, box_pooler.canonical_level)
    return level_assignments


def vis_softmax_cross_entropy_loss_(self, scale_weight):
    """
    Calculate the softmax cross entropy loss for the box classification with scale weight. Warp this model to FastRCNNOutputs.

    Args:
    -- self: FastRCNNOutputs.
    -- scale_weight: the weight for loss from different scale.

    Returns:
    -- losses: scalar.
    """

    if self._no_instances:
        return 0.0 * self.pred_class_logits.sum()
    else:
        losses = F.cross_entropy(self.pred_class_logits, self.gt_classes, reduction="none")
        losses = torch.mean(losses * scale_weight)
    return losses


def vis_smooth_l1_loss_(self, scale_weight):
    """
    Calculate the smooth l1 loss for box regression with scale weight. Warp this model to FastRCNNOutputs model.

    Args:
    -- self: FastRCNNOutputs.
    -- scale_weight: the weight for loss from different scale.

    Returns:
    -- losses: scalar.
    """
    if self._no_instances:
        return 0.0 * self.pred_proposal_deltas.sum()

    box_dim = self.gt_boxes.tensor.size(1)  # 4 or 5
    cls_agnostic_bbox_reg = self.pred_proposal_deltas.size(1) == box_dim
    device = self.pred_proposal_deltas.device

    bg_class_ind = self.pred_class_logits.shape[1] - 1

    # Box delta loss is only computed between the prediction for the gt class k
    # (if 0 <= k < bg_class_ind) and the target; there is no loss defined on predictions
    # for non-gt classes and background.
    # Empty fg_inds produces a valid loss of zero as long as the size_average
    # arg to smooth_l1_loss is False (otherwise it uses torch.mean internally
    # and would produce a nan loss).
    fg_inds = nonzero_tuple((self.gt_classes >= 0) & (self.gt_classes < bg_class_ind))[0]
    if cls_agnostic_bbox_reg:
        # pred_proposal_deltas only corresponds to foreground class for agnostic
        gt_class_cols = torch.arange(box_dim, device=device)
    else:
        fg_gt_classes = self.gt_classes[fg_inds]
        # pred_proposal_deltas for class k are located in columns [b * k : b * k + b],
        # where b is the dimension of box representation (4 or 5)
        # Note that compared to Detectron1,
        # we do not perform bounding box regression for background classes.
        gt_class_cols = box_dim * fg_gt_classes[:, None] + torch.arange(box_dim, device=device)

    gt_proposal_deltas = self.box2box_transform.get_deltas(self.proposals.tensor, self.gt_boxes.tensor)

    loss_box_reg = smooth_l1_loss(
        self.pred_proposal_deltas[fg_inds[:, None], gt_class_cols],
        gt_proposal_deltas[fg_inds],
        self.smooth_l1_beta,
        reduction="none",
    )
    # The loss is normalized using the total number of regions (R), not the number
    # of foreground regions even though the box regression loss is only defined on
    # foreground regions. Why? Because doing so gives equal training influence to
    # each foreground example. To see how, consider two different minibatches:
    #  (1) Contains a single foreground region
    #  (2) Contains 100 foreground regions
    # If we normalize by the number of foreground regions, the single example in
    # minibatch (1) will be given 100 times as much influence as each foreground
    # example in minibatch (2). Normalizing by the total number of regions, R,
    # means that the single example in minibatch (1) and each of the 100 examples
    # in minibatch (2) are given equal influence.

    loss_box_reg = torch.sum(loss_box_reg * scale_weight[fg_inds].unsqueeze(-1)) / self.gt_classes.numel()
    return loss_box_reg


def vis_FasterRCNN_loss(self, scale_weight):
    """
    Calculate the roi losses for faster rcnn.

    Args:
    -- self: FastRCNNOutputs.
    -- scale_weight: the weight for loss from different scale.

    Returns:
    -- losses.
    """
    return{
        "loss_cls": self.vis_softmax_cross_entropy_loss_(scale_weight),
        "loss_box_reg": self.vis_smooth_l1_loss_(scale_weight)
    }


def warp_FastRCNNOutputs(fastrcnnoutput):
    """
    Warp vis_FasterRCNN_loss,vis_softmax_cross_entropy_loss_ and vis_smooth_l1_loss_ to FastRCNNOutputs instance.

    Args:
    -- fastrcnnoutput: FasterRCNNOutputs instance.
    """
    fastrcnnoutput.vis_FasterRCNN_loss = types.MethodType(vis_FasterRCNN_loss, fastrcnnoutput)
    fastrcnnoutput.vis_smooth_l1_loss_ = types.MethodType(vis_smooth_l1_loss_, fastrcnnoutput)
    fastrcnnoutput.vis_softmax_cross_entropy_loss_ = types.MethodType(vis_softmax_cross_entropy_loss_, fastrcnnoutput)
    return fastrcnnoutput


def mask_rcnn_loss_vis(pred_mask_logits, instances, scale_weight):
    """
    Compute the mask loss for mask rcnn with scale weight. reference: mask_rcnn_loss in D2.

    Inputs:
    -- pred_mask_logits: A tensor of shape (B, C, Hmask, Wmask) or (B, 1, Hmask, Wmask) for class-specific or class-agnostic, where B is the total number of predicted masks
       in all images, C is the number of foreground classes, and Hmask, Wmask are the height and width of the mask predictions. The values are logits.
    -- instance: A list of N Instances, where N is the number of images in the batch. These instances are in 1:1 correspondence with the pred_mask_logits. The ground-truth labels (class, box, mask,
        ...) associated with each instance are stored in fields.
    -- scale_weight: the weight for loss from different scale.

    Returns:
    -- mask_loss: Tensor, scalar.
    """
    cls_agnostic_mask = pred_mask_logits.size(1) == 1
    total_num_masks = pred_mask_logits.size(0)
    mask_side_len = pred_mask_logits.size(2)
    assert pred_mask_logits.size(2) == pred_mask_logits.size(3), "Mask prediction must be square!"

    gt_classes = []
    gt_masks = []
    for instances_per_image in instances:
        if len(instances_per_image) == 0:
            continue
        if not cls_agnostic_mask:
            gt_classes_per_image = instances_per_image.gt_classes.to(dtype=torch.int64)
            gt_classes.append(gt_classes_per_image)

        gt_masks_per_image = instances_per_image.gt_masks.crop_and_resize(
            instances_per_image.proposal_boxes.tensor, mask_side_len
        ).to(device=pred_mask_logits.device)
        # A tensor of shape (N, M, M), N=#instances in the image; M=mask_side_len
        gt_masks.append(gt_masks_per_image)

    if len(gt_masks) == 0:
        return pred_mask_logits.sum() * 0

    gt_masks = cat(gt_masks, dim=0)

    if cls_agnostic_mask:
        pred_mask_logits = pred_mask_logits[:, 0]
    else:
        indices = torch.arange(total_num_masks)
        gt_classes = cat(gt_classes, dim=0)
        pred_mask_logits = pred_mask_logits[indices, gt_classes]

    gt_masks = gt_masks.to(dtype=torch.float32)

    mask_loss = F.binary_cross_entropy_with_logits(pred_mask_logits, gt_masks, reduction="none")
    mask_loss = torch.mean(mask_loss * scale_weight.view(pred_mask_logits.shape[0], 1, 1))
    return mask_loss
