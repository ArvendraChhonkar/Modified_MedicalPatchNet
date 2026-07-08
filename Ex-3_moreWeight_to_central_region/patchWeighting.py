import torch
import torch.nn.functional as F


def normalize_img(img):
    # img: [B,1,H,W]
    minv = img.amin(dim=(2, 3), keepdim=True)
    maxv = img.amax(dim=(2, 3), keepdim=True)
    img = (img - minv) / (maxv - minv + 1e-6)
    return img


def make_center_prior(batch, height, width, device):
    yy = torch.linspace(-1.0, 1.0, height, device=device).view(1, 1, height, 1)
    xx = torch.linspace(-1.0, 1.0, width, device=device).view(1, 1, 1, width)
    rr = torch.sqrt(xx * xx + yy * yy)
    center_prior = 1.0 - torch.clamp(rr, 0.0, 1.0)
    center_prior = center_prior.expand(batch, 1, height, width)
    return center_prior


def blur_map(x, kernel_size=31):
    pad = kernel_size // 2
    return F.avg_pool2d(F.pad(x, (pad, pad, pad, pad), mode="reflect"), kernel_size=kernel_size, stride=1)


def get_foreground_weight_map(img, min_weight=0.15):
    # img: [B,1,H,W], grayscale tensor in [0,1] or near that
    img = normalize_img(img)

    local_mean = blur_map(img, kernel_size=31)
    contrast = torch.abs(img - local_mean)
    contrast = normalize_img(contrast)

    center_prior = make_center_prior(img.size(0), img.size(2), img.size(3), img.device)

    soft_map = 0.55 * img + 0.25 * contrast + 0.20 * center_prior
    soft_map = normalize_img(soft_map)

    soft_map = min_weight + (1.0 - min_weight) * soft_map
    soft_map = torch.clamp(soft_map, min=min_weight, max=1.0)
    return soft_map


def patch_weights_from_map(weight_map, patch_size):
    # weight_map: [B,1,H,W]
    B, C, H, W = weight_map.shape
    assert C == 1
    assert H % patch_size == 0 and W % patch_size == 0

    pw = weight_map.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
    pw = pw.contiguous().view(B, 1, -1, patch_size, patch_size)
    pw = pw.mean(dim=(-1, -2)).squeeze(1)  # [B,P]
    pw = torch.clamp(pw, min=0.0, max=1.0)
    return pw


def patch_weights_to_grid(patch_weights, img_size, patch_size):
    # [B,P] -> [B,1,H,W] only for visualization
    B, P = patch_weights.shape
    patch_count = img_size // patch_size
    assert P == patch_count * patch_count
    grid = patch_weights.view(B, 1, patch_count, patch_count)
    grid = grid.repeat_interleave(patch_size, dim=2).repeat_interleave(patch_size, dim=3)
    return grid