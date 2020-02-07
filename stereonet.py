import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

from models.submodule import *


class FeatureExtraction(nn.Module):
    def __init__(self, num_stages):
        super().__init__()
        self.num_stages = num_stages
        self.conv2d_down = nn.ModuleList()
        c1, c2 = 3, 32
        for _ in range(num_stages):
            self.conv2d_down.append(nn.Conv2d(c1, c2, kernel_size=5, stride=2, padding=2))
            c1 = c2 = 32

        self.conv2d_blocks = nn.ModuleList()
        for _ in range(6):
            self.conv2d_blocks.append(BasicBlock(32, 32, 1, None, 1, 1))

        self.conv2d_out = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        output = x

        for k in range(self.num_stages):
            output = self.conv2d_down[k](output)

        for block in self.conv2d_blocks:
            output = block(output)

        return self.conv2d_out(output)


class EdgeAwareRefinement(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2d_ft = nn.Sequential(convbn(4, 32, 3, 1, 1, 1), nn.LeakyReLU(negative_slope=0.2, inplace=True))

        dilation_modules = [BasicBlock(32, 32, 1, None, 1, di) for di in [1, 2, 4, 8, 1, 1]]
        self.conv2d_dilated = nn.Sequential(*dilation_modules)

        self.conv2d_out = nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1)

    def forward(self, lr_d, hr_g):
        out_d = torch.unsqueeze(lr_d, dim=1)  # b*h*w -> b*1*h*w
        sr_d = F.interpolate(out_d, scale_factor=2, mode='bilinear', align_corners=False)
        sr_d *= 2
        out_d = self.conv2d_ft(torch.cat([sr_d, hr_g], dim=1))
        out_d = self.conv2d_dilated(out_d)
        return nn.ReLU(inplace=True)(torch.squeeze(sr_d + self.conv2d_out(out_d), dim=1))


class StereoNet(nn.Module):
    def __init__(self, maxdisp=192, num_stages=3):
        super().__init__()
        self.maxdisp = maxdisp
        self.num_stages = num_stages
        # feature extraction
        self.feature_extraction = FeatureExtraction(num_stages)
        # cost aggregation
        self.cost_agg = nn.ModuleList()
        for _ in range(4):
            self.cost_agg.append(nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                               nn.LeakyReLU(negative_slope=0.2, inplace=True)))
        self.cost_agg_out = nn.Conv3d(32, 1, kernel_size=3, stride=1, padding=1)
        # edge aware refinement
        self.edge_aware_refinements = nn.ModuleList()
        for _ in range(self.num_stages):
            self.edge_aware_refinements.append(EdgeAwareRefinement())

    def forward(self, left, right):
        disp_range = (self.maxdisp + 1) // pow(2, self.num_stages)
        # extract feature
        left_ft = self.feature_extraction(left)
        right_ft = self.feature_extraction(right)
        # build cost volume
        cost = build_minus_volume(left_ft, right_ft, disp_range)
        # cost aggregation
        for f in self.cost_agg:
            cost = f(cost)
        # predict
        cost = self.cost_agg_out(cost)
        cost = torch.squeeze(cost, dim=1)  # b,1,d,h,w -> b,d,h,w
        pred = F.softmax(cost, dim=1)
        pred = disparity_regression(pred, disp_range)
        # refinements
        pred_pyramid_list = [pred]  # /8 or /16
        img_pyramid_list = build_image_pyramid(left, self.num_stages)  # /4, /2, /1 or /8
        for k in range(self.num_stages):
            pred_pyramid_list.append(self.edge_aware_refinements[k](pred_pyramid_list[-1], img_pyramid_list[k]))
        # upsampling
        for k in range(len(pred_pyramid_list)):
            pred_pyramid_list[k] = pred_pyramid_list[k] * (left.size()[-1]) / pred_pyramid_list[k].size()[-1]
            pred_pyramid_list[k] = torch.squeeze(
                F.interpolate(
                    torch.unsqueeze(pred_pyramid_list[k], dim=1),
                    size=left.size()[-2:], mode='bilinear', align_corners=False),
                dim=1
            )

        if self.training:
            return pred_pyramid_list
        else:
            return pred_pyramid_list[-1]


if __name__ == '__main__':
    model = StereoNet(maxdisp=192, num_stages=3)
    model.eval()
    print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

    import time

    cudnn.benchmark = True
    input = torch.FloatTensor(6, 3, 540, 960).zero_().cuda()
    start = time.time()
    for i in range(100):
        out = model(input, input)
        for o in out:
            print(o.size())
        exit(0)
    elapsed = time.time() - start
    print('elapsed time', elapsed)
