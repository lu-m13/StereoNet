import torch
import torch.nn as nn
import torch.nn.functional as F


def convbn(c1, c2, k, s, p, d):
    """conv + bn"""
    return nn.Sequential(nn.Conv2d(c1, c2, k, s, padding=d if d > 1 else p, dilation=d, bias=False),
                         nn.BatchNorm2d(c2))


def convbn_3d(c1, c2, k, s, p):
    """conv3d + bn"""
    return nn.Sequential(nn.Conv3d(c1, c2, k, s, padding=p, bias=False),
                         nn.BatchNorm3d(c2))


class BasicBlock(nn.Module):
    """ResNet BasicBlock"""
    expansion = 1

    def __init__(self, c1, c2, s, downsample, p, d):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Sequential(convbn(c1, c2, 3, s, p, d), nn.ReLU(inplace=True))
        self.conv2 = convbn(c2, c2, 3, 1, p, d)
        self.downsample = downsample
        self.stride = s

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample is not None:
            x = self.downsample(x)
        out += x
        return out


def disparity_regression(x, maxdisp):
    """disparity regression"""
    assert len(x.shape) == 4
    disp_values = torch.arange(0, maxdisp, dtype=x.dtype, device=x.device)
    disp_values = disp_values.view(1, maxdisp, 1, 1)
    return torch.sum(x * disp_values, 1, keepdim=False)


def build_image_pyramid(img, k):
    image_pyramid = [F.interpolate(img,scale_factor=(1./(2**i)),mode='bilinear')
                     for i in range(k)]
    return list(reversed(image_pyramid))


def build_concat_volume(left_ft, right_ft, maxdisp):
    B, C, H, W = left_ft.shape
    volume = left_ft.new_zeros([B, 2 * C, maxdisp, H, W])

    for i in range(maxdisp):
        if i > 0:
            volume[:, :C, i, :, i:] = left_ft[:, :, :, i:]
            volume[:, C:, i, :, i:] = right_ft[:, :, :, :-i]
        else:
            volume[:, :C, i, :, :] = left_ft
            volume[:, C:, i, :, :] = right_ft

    volume = volume.contiguous()

    return volume


def build_minus_volume(left_ft, right_ft, maxdisp):
    B, C, H, W = left_ft.shape
    volume = left_ft.new_zeros([B, C, maxdisp, H, W])
    for i in range(maxdisp):
        if i > 0:
            volume[:, :, i, :, i:] = left_ft[:, :, :, i:] - right_ft[:, :, :, :-i]
        else:
            volume[:, :, i, :, :] = left_ft - right_ft

    volume = volume.contiguous()
    return volume


def groupwise_correlation(fea1, fea2, num_groups):
    B, C, H, W = fea1.shape
    assert C % num_groups == 0
    channels_per_group = C // num_groups
    cost = (fea1 * fea2).view([B, num_groups, channels_per_group, H, W]).mean(dim=2)
    assert cost.shape == (B, num_groups, H, W)
    return cost


def build_gwc_volume(left_ft, right_ft, maxdisp, num_groups):
    B, C, H, W = left_ft.shape
    volume = left_ft.new_zeros([B, num_groups, maxdisp, H, W])
    for i in range(maxdisp):
        if i > 0:
            volume[:, :, i, :, i:] = groupwise_correlation(left_ft[:, :, :, i:],
                                                           right_ft[:, :, :, :-i],
                                                           num_groups)
        else:
            volume[:, :, i, :, :] = groupwise_correlation(left_ft, right_ft, num_groups)
    volume = volume.contiguous()
    return volume


if __name__ == '__main__':
    print(__file__)

