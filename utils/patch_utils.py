import numpy as np
import torch

def patch_initialization(args, patch_type='rectangle'):
    # if args has not noise_percentage, then use 0.1 as default
    if not hasattr(args, 'noise_percentage'):
        args.noise_percentage = 0.03
    noise_percentage = args.noise_percentage
    image_size = (3, 224, 224)
    if patch_type == 'rectangle':
        mask_length = int((noise_percentage * image_size[1] * image_size[2])**0.5)
        patch = np.random.rand(image_size[0], mask_length, mask_length)
    return patch

def mask_generation(args, patch):
    image_size = (3, 224, 224)
    applied_patch = np.zeros(image_size)
    x_location = image_size[1] - 14 - patch.shape[1]
    y_location = image_size[1] - 14 - patch.shape[2]
    applied_patch[:, x_location: x_location + patch.shape[1], y_location: y_location + patch.shape[2]] = patch
    mask = applied_patch.copy()
    mask[mask != 0] = 1.0
    return mask, applied_patch ,x_location, y_location

def clamp_patch(args, patch):
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    min_in = np.array([0, 0, 0])
    max_in = np.array([1, 1, 1])
    min_out, max_out = np.min((min_in - mean) / std), np.max((max_in - mean) / std)
    patch = torch.clamp(patch, min=min_out, max=max_out)
    return patch

from torchvision.utils import save_image

def de_normalize(img, mean=None, std=None, debug = False):
    if mean is None:
        mean = (0.485, 0.456, 0.406) 
    if std is None:
        std = (0.229, 0.224, 0.225)
    if isinstance(img, torch.Tensor):
        mean = torch.tensor(mean).unsqueeze(1).unsqueeze(2)
        std = torch.tensor(std).unsqueeze(1).unsqueeze(2)
    mean = mean.to(img.device)
    std = std.to(img.device)
    img = img * std + mean
    if debug:
        img_to_save = img.squeeze(1)    
        save_image(img_to_save, 'test.png', n_row=4, normalize=True)
    return img