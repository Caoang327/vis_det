import torch


def tv_loss(img, alpha=2):
    """
    Compute total variation loss.

    Args:
    -- img: PyTorch Variable of shape (N, 3, H, W) holding an input image.
    -- alpha: alpha norm.

    Returns:
    -- loss: PyTorch Variable holding a scalar giving the total variation loss for img.
    """
    N = img.shape[0]
    w_variance = torch.sum(torch.pow(img[:, :, :, :-1] - img[:, :, :, 1:], alpha))
    h_variance = torch.sum(torch.pow(img[:, :, :-1, :] - img[:, :, 1:, :], alpha))
    loss = (h_variance + w_variance) / N
    return loss


class layout_vis_loss(object):
    def __init__(self, args):
        super().__init__()
        self.arch = args.arch

        # alpha norm value (alpha), alpha norm loss weight (lamb_norm), tv_loss alpha (tv_alpha), tv loss weight (lamb_tv).
        self.alpha = args.alpha
        self.lamb_norm = args.lamb_norm
        self.lamb_tv = args.lamb_tv
        self.tv_alpha = args.tv_alpha
        if self.arch == "retina":
            # focal classification loss weight (cls_wei), regresssion loss weight (reg_wei) and pre-sigmoid loss weight (pos_wei).
            self.cls_wei = args.cls_wei
            self.reg_wei = args.reg_wei
            self.fore_wei = args.fore_wei
        elif self.arch == "fasterrcnn":
            # rpn classification loss weight (rpn_cls_wei) and regression loss weight (rpn_reg_wei).
            self.rpn_cls_wei = args.rpn_cls_wei
            self.rpn_reg_wei = args.rpn_reg_wei

            # roi classification loss weight (roi_cls_wei), regression loss weight (roi_reg_wei), pre-sigmoid loss weight for positive anchors (roi_pos_wei)
            # and mask loss weight (roi_mask_wei).
            self.roi_cls_wei = args.roi_cls_wei
            self.roi_reg_wei = args.roi_cls_wei
            self.roi_pos_wei = args.roi_pos_wei
        elif self.arch == "maskrcnn":
            # rpn classification loss weight (rpn_cls_wei) and regression loss weight (rpn_reg_wei).
            self.rpn_cls_wei = args.rpn_cls_wei
            self.rpn_reg_wei = args.rpn_reg_wei

            # roi classification loss weight (roi_cls_wei), regression loss weight (roi_reg_wei), pre-sigmoid loss weight for positive anchors (roi_pos_wei)
            # and mask loss weight (roi_mask_wei).
            self.roi_cls_wei = args.roi_cls_wei
            self.roi_reg_wei = args.roi_cls_wei
            self.roi_pos_wei = args.roi_pos_wei
            self.roi_mask_wei = args.roi_mask_wei
        else:
            raise NotImplementedError("Not implemented meta arch")

    def forward(self, x, losses, alter=False, stage=0):
        """
        Calculate the optimization loss in visualization.

        Args:
        -- x (tensor): the input tensor of model.
        -- losses (dict): loss dict calculated by model.
        -- alter (bool, optional): if alternative optimizing. If use alternative optimization, different stage has different loss. Defaults to False.
        -- stage (int, optional): Stage of alternative optimization if alter sets to True. Defaults to 0.
        """
        if self.arch == "retina":
            loss = -1 * self.fore_wei * losses['fore_loss'] + self.cls_wei * losses['cls_loss'] + self.reg_wei * losses['reg_loss'] + self.lamb_norm * torch.sum(x**self.alpha) + self.lamb_tv * tv_loss(x, self.tv_alpha)
        elif self.arch == 'fasterrcnn':
            if alter is True:
                if stage == 0:
                    loss = self.rpn_cls_wei * losses['loss_rpn_cls'] + self.rpn_reg_wei * losses['loss_rpn_loc'] + self.lamb_norm * torch.sum(x**self.alpha) + self.lamb_tv * tv_loss(x, self.tv_alpha)
                elif stage == 1:
                    loss = -1 * self.roi_pos_wei * losses['loss_pos_scores'] + self.roi_cls_wei * losses['loss_cls'] + self.roi_reg_wei * losses['loss_box_reg']
            else:
                loss = -1 * self.roi_pos_wei * losses['loss_pos_scores'] + self.roi_cls_wei * losses['loss_cls'] + self.roi_reg_wei * losses['loss_box_reg'] + self.rpn_cls_wei * losses['loss_rpn_cls'] +\
                 self.rpn_reg_wei * losses['loss_rpn_loc'] + self.lamb_norm * torch.sum(x**self.alpha) + self.lamb_tv * tv_loss(x, self.tv_alpha)
        elif self.arch == 'maskrcnn':
            if alter is True:
                if stage == 0:
                    loss = self.rpn_cls_wei * losses['loss_rpn_cls'] + self.rpn_reg_wei * losses['loss_rpn_loc'] + self.lamb_norm * torch.sum(x**self.alpha) + self.lamb_tv * tv_loss(x, self.tv_alpha)
                elif stage == 1:
                    loss = -1 * self.roi_pos_wei * losses['loss_pos_scores'] + self.roi_cls_wei * losses['loss_cls'] + self.roi_reg_wei * losses['loss_box_reg'] + self.roi_mask_wei * losses['loss_mask']
            else:
                loss = -1 * self.roi_pos_wei * losses['loss_pos_scores'] + self.roi_cls_wei * losses['loss_cls'] + self.roi_reg_wei * losses['loss_box_reg'] + self.rpn_cls_wei * losses['loss_rpn_cls'] +\
                 self.rpn_reg_wei * losses['loss_rpn_loc'] + self.roi_mask_wei * losses['loss_mask'] + self.lamb_norm * torch.sum(x**self.alpha) + self.lamb_tv * tv_loss(x, self.tv_alpha)

        return loss
