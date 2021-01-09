import torch
from vis_det.optimize.util import clipping, calculate_clipping, blur_image, jitter
import numpy as np
from vis_det.optimize.loss import tv_loss
from detectron2.structures import Instances, Boxes
import math 


class layout_optimize(object):
    def __init__(self, args, cfg):
        super().__init__()
        self.arch = args.arch
        self.lr = args.lr
        self.optimization = args.optimization
        self.niter = args.niter
        self.lr_decay = args.lr_decay
        self.lr_decay_time = args.lr_decay_time
        self.blur_every = args.blur_every
        self.blur_start = args.blur_start
        self.blur_decay = args.blur_decay
        self.blur_end = args.blur_end
        self.if_jitter = args.if_jitter
        self.jitter_every = args.jitter_every
        self.jitter_x = args.jitter_x
        self.jitter_y = args.jitter_y
        self.momentum = args.momentum
        self.alter = args.alter
        self.scale_weight = args.scale_weight
        lo, hi = calculate_clipping(cfg, 1/255.0)
        self.std = torch.Tensor(cfg.MODEL.PIXEL_STD).view(3, 1, 1)
        self.mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).view(3, 1, 1)
        self.LO = lo
        self.HI = hi

    def retina_invert(self, images, gt_instances, model, loss_fun):
        lo = self.LO
        hi = self.HI

        # initialize input tensor
        with torch.no_grad():
            x = torch.randn(images.tensor.shape).to(images.device)
            x = x * 0.2
            x = clipping(x, lo, hi)

        x = x.requires_grad_()
        if self.optimization == "SGD":
            optimizer = torch.optim.SGD([x], self.lr, momentum=self.momentum)
        else:
            raise NotImplementedError("Not implemented optimizer")
        lr = self.lr
        sigma = self.blur_start
        for i in range(self.niter):
            print(1)
            losses = model.vis(x * 255, gt_instances, scale_weight=self.scale_weight)
            loss = loss_fun.forward(x, losses, alter=False)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            lr_ = lr * (self.lr_decay ** (i//self.lr_decay_time))
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_
            with torch.no_grad():

                x = clipping(x, lo, hi)

                if i % self.blur_every == 0:
                    sigma = max(sigma * self.blur_decay, self.blur_end)
                    blur_image(x.data, sigma=sigma)

                if self.if_jitter is True:

                    if i % self.jitter_every == 0:
                        jitter(x.data, self.jitter_x, self.jitter_y)
        return x

    def rcnn_invert(self, images, gt_instances, model, loss_fun):
        lo = self.LO
        hi = self.HI

        # initialize input tensor
        with torch.no_grad():
            x = torch.randn(images.tensor.shape).to(images.device)
            x = x * 0.2
            x = clipping(x, lo, hi)

        x = x.requires_grad_()
        if self.optimization == "SGD":
            optimizer = torch.optim.SGD([x], self.lr, momentum=self.momentum)
        else:
            raise NotImplementedError("Not implemented optimizer")

        sigma = self.blur_start
        lr = self.lr
        print("Initialization ")
        for i in range(self.niter):
            # if i % 10 == 0:
            #     print("iteration %d" % (i))
            losses = model.vis(x*255, gt_instances, images, scale_weight=self.scale_weight)
            if self.alter is True:
                loss = loss_fun.forward(x, losses, alter=True, stage=0)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses = model.vis(x*255, gt_instances, images, scale_weight=self.scale_weight)
                loss = loss_fun.forward(x, losses, alter=True, stage=1)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            else:
                loss = loss_fun.forward(x, losses, alter=False)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            with torch.no_grad():

                x = clipping(x, lo, hi)

                if i % self.blur_every == 0:
                    sigma = max(sigma * self.blur_decay, self.blur_end)
                    blur_image(x.data, sigma=sigma)

                if self.if_jitter is True:

                    if i % self.jitter_every == 0 and (self.niter - 1) >= 100:
                        jitter(x.data, self.jitter_x, self.jitter_y)

                lr_ = lr * (self.lr_decay ** (i//self.lr_decay_time))
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_
            if i % 100 == 0:
                print("iteration %d" % (i))
        return x

    def invert(self, images, gt_instances, model, loss_fun):
        print("Into Invert")
        if self.arch == 'retina':
            print("Retina Invert")
            return self.retina_invert(images, gt_instances, model, loss_fun)
        elif self.arch == 'fasterrcnn':
            print("Faster Invert")
            return self.rcnn_invert(images, gt_instances, model, loss_fun)
        elif self.arch == 'maskrcnn':
            print("Mask Invert")
            return self.rcnn_invert(images, gt_instances, model, loss_fun)

    def post_processing(self, x):
        """
        Transfer the inverted tensor to numpy image.
        """
        with torch.no_grad():
            x = clipping(x, self.LO, self.HI)
            inverse_img = (x * 255) * self.std.to(x.device) + self.mean.to(x.device)
            inverse_img = inverse_img.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
        return inverse_img


class neuron_optimize(object):
    """
    An abstract optimization object for visualizing individual objects.
    """
    def __init__(self, args, cfg):
        super().__init__()
        self.arch = args.arch
        self.lr = args.lr
        self.optimization = args.optimization
        self.lamb_norm = args.lamb_norm
        self.alpha = args.alpha
        self.lamb_tv = args.lamb_tv
        self.tv_alpha = args.tv_alpha
        self.niter = args.niter
        self.lr_decay = args.lr_decay
        self.lr_decay_time = args.lr_decay_time
        self.blur_every = args.blur_every
        self.blur_start = args.blur_start
        self.blur_decay = args.blur_decay
        self.blur_end = args.blur_end
        self.if_jitter = args.if_jitter
        self.jitter_every = args.jitter_every
        self.jitter_x = args.jitter_x
        self.jitter_y = args.jitter_y
        self.momentum = args.momentum
        self.alter = args.alter
        self.scale_weight = args.scale_weight
        lo, hi = calculate_clipping(cfg, 1/255.0)
        self.std = torch.Tensor(cfg.MODEL.PIXEL_STD).view(3, 1, 1)
        self.mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).view(3, 1, 1)
        self.LO = lo
        self.HI = hi

    def Retina_Neuron(self, model, category, scale, anchor_type, scale_weight, p_x=0.5, p_y=0.5, batch=3, H=672, W=672):
        lo = self.LO
        hi = self.HI

        # initialize input tensor
        with torch.no_grad():
            x = torch.randn((batch, 3, H, W)).to(model.device)
            x = x * 0.2
            x = clipping(x, lo, hi)
        x = x.requires_grad_()

        if self.optimization == "SGD":
            optimizer = torch.optim.SGD([x], self.lr, momentum=self.momentum)
        else:
            raise NotImplementedError("Not implemented optimizer")
        lr = self.lr
        sigma = self.blur_start
        for i in range(self.niter):
            features = model.backbone(x*255)
            features = [features[f] for f in model.in_features]
            box_cls, box_deltas = model.head(features)
            box_cls = box_cls[scale]
            N, _, H, W = box_cls.shape
            box_cls = box_cls.view(N, -1, 80, H, W)
            pH = int(p_x * (H-1))
            pW = int(p_y * (W-1))

            loss = -1 * torch.sum(box_cls[:, anchor_type, category, pH, pW]) * scale_weight[scale] + batch * self.lamb_norm * torch.sum(x**self.alpha) + 3 * self.lamb_tv * tv_loss(x, self.tv_alpha)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            lr_ = lr * (self.lr_decay ** (i//self.lr_decay_time))
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            with torch.no_grad():
                x = clipping(x, lo, hi)
                if i % self.blur_every == 0:
                    sigma = max(sigma * self.blur_decay, self.blur_end)
                    blur_image(x.data, sigma=sigma)
                if self.if_jitter is True:
                    if i % self.jitter_every == 0:
                        jitter(x.data, self.jitter_x, self.jitter_y)

    def RCNN_Neuron(self, model, category, scale, anchor_type, scale_weight, p_x=0.5, p_y=0.5, batch=3, H=672, W=672):
        lo = self.LO
        hi = self.HI
        gt_proposals = Instances((672, 672))
        scale_size = [32.0, 64.0, 128.0, 256.0, 512.0]
        if anchor_type == 0:
            ly = math.sqrt(scale_size[scale]**2/2)/2
            lx = 2 * ly
        elif anchor_type == 1:
            ly = scale_size[scale]/2
            lx = ly
        elif anchor_type == 2:
            lx = math.sqrt(scale_size[scale]**2/2)/2
            ly = lx * 2
        # initialize input tensor
        with torch.no_grad():
            x = torch.randn((batch, 3, H, W)).to(model.device)
            x = x * 0.2
            x = clipping(x, lo, hi)
        x = x.requires_grad_()

        if self.optimization == "SGD":
            optimizer = torch.optim.SGD([x], self.lr, momentum=self.momentum)
        else:
            raise NotImplementedError("Not implemented optimizer")
        lr = self.lr
        sigma = self.blur_start
        for i in range(self.niter):
            features = model.backbone(x*255)
            features = [features[f] for f in model.in_features]
            box_cls, box_deltas = model.head(features)
            box_cls = box_cls[scale]
            N, _, H, W = box_cls.shape
            box_cls = box_cls.view(N, -1, 80, H, W)
            pH = int(p_x * (H-1))
            pW = int(p_y * (W-1))

            loss = -1 * torch.sum(box_cls[:, anchor_type, category, pH, pW]) * scale_weight[scale] + batch * self.lamb_norm * torch.sum(x**self.alpha) + 3 * self.lamb_tv * tv_loss(x, self.tv_alpha)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            lr_ = lr * (self.lr_decay ** (i//self.lr_decay_time))
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            with torch.no_grad():
                x = clipping(x, lo, hi)
                if i % self.blur_every == 0:
                    sigma = max(sigma * self.blur_decay, self.blur_end)
                    blur_image(x.data, sigma=sigma)
                if self.if_jitter is True:
                    if i % self.jitter_every == 0:
                        jitter(x.data, self.jitter_x, self.jitter_y)