import torch
from scipy.ndimage.filters import gaussian_filter1d


def jitter(X, ox, oy):
    """
    Helper function to randomly jitter an image.

    Args
    -- X: PyTorch Tensor of shape (N, C, H, W)
    -- ox, oy: Integers giving number of pixels to jitter along W and H axes

    Returns:
    -- X: A new PyTorch Tensor of shape (N, C, H, W)
    """
    if ox != 0:
        left = X[:, :, :, :-ox]
        right = X[:, :, :, -ox:]
        X = torch.cat([right, left], dim=3)
    if oy != 0:
        top = X[:, :, :-oy]
        bottom = X[:, :, -oy:]
        X = torch.cat([bottom, top], dim=2)
    return X


def blur_image(X, sigma=1):
    """
    Helper function to blur an image.

    Args:
    -- X: Pytorch tensor of shape (N, C, H, W).
    -- sigma: sigma of gaussian blur.

    Returns
    -- A new pytorch tensor of shape (N, C, H, W).
    """
    X_np = X.cpu().clone().numpy()
    X_np = gaussian_filter1d(X_np, sigma, axis=2)
    X_np = gaussian_filter1d(X_np, sigma, axis=3)
    X.copy_(torch.Tensor(X_np).type_as(X))
    return X


def clipping(X, lo, hi):
    """
    Helper function for clipping.

    Args:
    -- X: Pytorch tensor of shape (N, C, H, W)
    -- lo: list, lower bound for each channel
    -- hi: list, upper bound for each channel

    Returns:
    -- X: Pytorch tensor of shape (N, C, H, W)
    """
    for c in range(3):
        X.data[:, c].clamp_(min=lo[c], max=hi[c])
    return X


def calculate_clipping(cfg, scale):
    """
    Helper function for calculating lo and hi.

    Args:
    -- cfg: configuration file for model
    -- scale: scale for lo, hi. If lo, hi in [0,255], scale is 1. If lo, hi in [0,1], scale is 1/255

    Returns:
    -- LO, HI: list, lower bound and upper bound of each channel
    """
    LO = []
    HI = []
    for c in range(3):
        lo = float(-cfg.MODEL.PIXEL_MEAN[c] / cfg.MODEL.PIXEL_STD[c])*scale
        hi = float((255.0 - cfg.MODEL.PIXEL_MEAN[c]) / cfg.MODEL.PIXEL_STD[c])*scale
        LO.append(lo)
        HI.append(hi)
    return LO, HI
