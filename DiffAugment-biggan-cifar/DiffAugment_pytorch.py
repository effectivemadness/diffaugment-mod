# Differentiable Augmentation for Data-Efficient GAN Training
# Shengyu Zhao, Zhijian Liu, Ji Lin, Jun-Yan Zhu, and Song Han
# https://arxiv.org/pdf/2006.10738

import torch
import torch.nn.functional as F
import numpy as np
import cv2
import torchvision



def DiffAugment(x, policy='', channels_first=True):
    
    if policy:
        if not channels_first:
            x = x.permute(0, 3, 1, 2)
        # torchvision.utils.save_image(x.cpu()[0], 'before.png',normalize=True)
        for p in policy.split(','):
            for f in AUGMENT_FNS[p]:
                x = f(x)
        # torchvision.utils.save_image(x.cpu()[0], 'after.png',normalize=True)
        if not channels_first:
            x = x.permute(0, 2, 3, 1)
        x = x.contiguous()
    return x


def rand_brightness(x):
    x = x + (torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device) - 0.5)
    # print("brightness")
    return x


def rand_saturation(x):
    x_mean = x.mean(dim=1, keepdim=True)
    x = (x - x_mean) * (torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device) * 2) + x_mean
    # print("sat")
    return x


def rand_contrast(x):
    x_mean = x.mean(dim=[1, 2, 3], keepdim=True)
    x = (x - x_mean) * (torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device) + 0.5) + x_mean
    # print("contrast")
    return x


def rand_translation(x, ratio=0.125):
    shift_x, shift_y = int(x.size(2) * ratio + 0.5), int(x.size(3) * ratio + 0.5)
    translation_x = torch.randint(-shift_x, shift_x + 1, size=[x.size(0), 1, 1], device=x.device)
    translation_y = torch.randint(-shift_y, shift_y + 1, size=[x.size(0), 1, 1], device=x.device)
    grid_batch, grid_x, grid_y = torch.meshgrid(
        torch.arange(x.size(0), dtype=torch.long, device=x.device),
        torch.arange(x.size(2), dtype=torch.long, device=x.device),
        torch.arange(x.size(3), dtype=torch.long, device=x.device),
    )
    grid_x = torch.clamp(grid_x + translation_x + 1, 0, x.size(2) + 1)
    grid_y = torch.clamp(grid_y + translation_y + 1, 0, x.size(3) + 1)
    x_pad = F.pad(x, [1, 1, 1, 1, 0, 0, 0, 0])
    x = x_pad.permute(0, 2, 3, 1).contiguous()[grid_batch, grid_x, grid_y].permute(0, 3, 1, 2).contiguous()
    # print("translation")
    return x


def rand_cutout(x, ratio=0.5):
    # print(x.size());
    cutout_size = int(x.size(2) * ratio + 0.5), int(x.size(3) * ratio + 0.5)
    offset_x = torch.randint(0, x.size(2) + (1 - cutout_size[0] % 2), size=[x.size(0), 1, 1], device=x.device)
    offset_y = torch.randint(0, x.size(3) + (1 - cutout_size[1] % 2), size=[x.size(0), 1, 1], device=x.device)
    grid_batch, grid_x, grid_y = torch.meshgrid(
        torch.arange(x.size(0), dtype=torch.long, device=x.device),
        torch.arange(cutout_size[0], dtype=torch.long, device=x.device),
        torch.arange(cutout_size[1], dtype=torch.long, device=x.device),
    )
    grid_x = torch.clamp(grid_x + offset_x - cutout_size[0] // 2, min=0, max=x.size(2) - 1)
    grid_y = torch.clamp(grid_y + offset_y - cutout_size[1] // 2, min=0, max=x.size(3) - 1)
    mask = torch.ones(x.size(0), x.size(2), x.size(3), dtype=x.dtype, device=x.device)
    mask[grid_batch, grid_x, grid_y] = 0
    x = x * mask.unsqueeze(1)
    # print("cutout");
    return x

def get_rot_mat(theta):
    # theta = torch.tensor(theta, device=theta.device) # 수정사항.
    return torch.tensor([[torch.cos(theta), -torch.sin(theta), 0],
                         [torch.sin(theta), torch.cos(theta), 0]], device=theta.device)


def rand_rotation(x, max_theta=25):
    # print(x.size())
    # before_img = x.cpu()[0].permute(1, 2, 0).detach().numpy()
    # print(before_img.shape)
    # cv2.imwrite('before.jpg', before_img)
    # torchvision.utils.save_image(x.cpu()[0], 'before.png',normalize=True)
    theta = (torch.rand(1, device=x.device)[0] * 2 - 1) * np.pi * (max_theta / 180)
    # print(theta)
    rot_mat = get_rot_mat(theta)[None, ...].type(torch.cuda.FloatTensor).repeat(x.shape[0],1,1)
    grid = F.affine_grid(rot_mat, x.size(),align_corners=True).type(torch.cuda.FloatTensor) # align_corners default = false. modded.
    x = F.grid_sample(x, grid,align_corners=True) # align_corners default = false. modded.
    # print("rotation")
    # after_img = x.cpu()[0].permute(1, 2, 0).numpy()
    # cv2.imwrite('after.png', after_img)
    # torchvision.utils.save_image(x.cpu()[0], 'after.png',normalize=True)
    return x

def get_flip_mat(x, horizontal, vertical):
    width = x.size()[2]
    height = x.size()[3]
    # print(width, height)
    if horizontal:
        # horizontal_mat = torch.tensor([[-1, 0, width-1], -1~1 의 값을 가진다고 하는데, 그래서 translation이 따로 필요 없는 듯.
        horizontal_mat = torch.tensor([[-1, 0, 0],
                                       [0, 1, 0]], device=x.device)
        return horizontal_mat
    elif vertical:
        vertical_mat = torch.tensor([[1, 0, 0],
                                     [0, -1, 0]], device=x.device)
        return vertical_mat

def rand_flip(x, horizontal=True, vertical=False):
    if horizontal:
        horizontal_rand = torch.rand(1, device=x.device)[0]
        if horizontal_rand < 0.5:
            # print(x.size())
            # before_img = x.cpu()[0].permute(1, 2, 0).detach().numpy()
            # print(before_img.shape)
            # cv2.imwrite('before.jpg', before_img)
            # torchvision.utils.save_image(x.cpu()[0], 'before.png',normalize=True)
            flip_mat = get_flip_mat(x, True, False)[None, ...].type(torch.cuda.FloatTensor).repeat(x.shape[0],1,1)
            grid = F.affine_grid(flip_mat, x.size(),align_corners=True).type(torch.cuda.FloatTensor)
            x = F.grid_sample(x, grid,align_corners=True)
            # print("rotation")
            # after_img = x.cpu()[0].permute(1, 2, 0).numpy()
            # cv2.imwrite('after.png', after_img)
            # torchvision.utils.save_image(x.cpu()[0], 'after.png',normalize=True)
    if vertical:
        vertical_rand = torch.rand(1, device=x.device)[0]
        if vertical_rand < 0.5:
            # torchvision.utils.save_image(x.cpu()[0], 'before.png',normalize=True)
            flip_mat = get_flip_mat(x, False, True)[None, ...].type(torch.cuda.FloatTensor).repeat(x.shape[0],1,1)
            grid = F.affine_grid(flip_mat, x.size(),align_corners=True).type(torch.cuda.FloatTensor)
            x = F.grid_sample(x, grid,align_corners=True)
            # torchvision.utils.save_image(x.cpu()[0], 'after.png',normalize=True)
    return x






AUGMENT_FNS = {
    'color': [rand_brightness, rand_saturation, rand_contrast],
    'translation': [rand_translation],
    'cutout': [rand_cutout],
    'rotation': [rand_rotation],
    'flip' : [rand_flip]
}
