import torch
from torch import nn 
from torch.utils.data import Dataset, DataLoader



def to_torch(x):
    if isinstance(x,torch.Tensor):
         return x
    else:
        return torch.tensor(x, dtype=torch.float32)

def meshgrid(h, w):
        y = torch.arange(0, h).view(h, 1, 1).repeat(1, w, 1)
        x = torch.arange(0, w).view(1, w, 1).repeat(h, 1, 1)

        arr = torch.cat([y, x], dim=-1)
        return arr

def delta_border( h, w):
    """
    :param h: height
    :param w: width
    :return: normalized distance to image border,
        wtith min distance = 0 at border and max dist = 0.5 at image center
    """
    lower_right_corner = torch.tensor([h - 1, w - 1]).view(1, 1, 2)
    arr = meshgrid(h, w) / lower_right_corner
    dist_left_up = torch.min(arr, dim=-1, keepdims=True)[0]
    dist_right_down = torch.min(1 - arr, dim=-1, keepdims=True)[0]
    edge_dist = torch.min(torch.cat([dist_left_up, dist_right_down], dim=-1), dim=-1)[0]
    return edge_dist

def get_weighting(h, w, Ly, Lx, device):
    weighting = delta_border(h, w)
    # weighting = torch.clip(weighting, self.split_input_params["clip_min_weight"],
    #                        self.split_input_params["clip_max_weight"], )
    weighting = torch.clip(weighting, 0.001,  1)
    weighting = weighting.view(1, h * w, 1).repeat(1, 1, Ly * Lx).to(device)
    # weighting = torch.ones((1, h * w, 1)).repeat(1, 1, Ly * Lx).to(device)
    # if self.split_input_params["tie_braker"]:
    #     L_weighting = self.delta_border(Ly, Lx)
    #     L_weighting = torch.clip(L_weighting,
    #                              self.split_input_params["clip_min_tie_weight"],
    #                              self.split_input_params["clip_max_tie_weight"])

    #     L_weighting = L_weighting.view(1, 1, Ly * Lx).to(device)
    #     weighting = weighting * L_weighting
    return weighting


def get_unfold(x, kernel_size, overlap,):
    '''
        input: x: (N, C, H, W)
        output: x (N, L, C, kernel_size, kernel_size)
    '''

    if not isinstance(x, torch.Tensor):
        x = to_torch(x)
    bs, nc, h, w = x.shape

    stride = kernel_size-overlap
    padding_h = kernel_size - (h - kernel_size ) % stride
    padding_w = kernel_size - (w - kernel_size ) % stride
    # padding_h =  (h - kernel_size ) % stride
    # padding_w =  (w - kernel_size ) % stride
    pad = nn.ReplicationPad2d(padding=(0, padding_w, 0, padding_h))
    x_pad = pad(x)
    fold_params = dict(kernel_size=kernel_size, dilation=1, padding=0, stride=stride)
    unfold = torch.nn.Unfold(**fold_params)
    fold = torch.nn.Fold(output_size=(h+padding_h, w+padding_w),**fold_params)

    Ly = (h + padding_h - fold_params['kernel_size']) // fold_params['stride'] + 1
    Lx = (w + padding_w - fold_params['kernel_size']) // fold_params['stride'] + 1
    x_patches = unfold(x_pad).reshape((bs, nc, kernel_size, kernel_size, Ly*Lx))
    x_patches = x_patches.permute(0,4, 1,2,3)

    weighting =  get_weighting(kernel_size, kernel_size, Ly, Lx, x.device).to(x.dtype)
    normalization = fold(weighting).view(1, 1, h+padding_h, w+padding_w)  # normalizes the overlap
    weighting = weighting.view((1, 1, kernel_size, kernel_size, Ly * Lx))

    return x_patches,fold, normalization, weighting

def get_fold(x_patches, output_size,  kernel_size, padding=0):
    '''
    input:  x: (N*L, C, kernel_size, kernel_size)
    output: x: (N, C, H, W)
    '''
    _, C, _, _ = x_patches.shape
    if not isinstance(x_patches, torch.Tensor):
        x_patches = to_torch(x_patches)
    L = output_size[0]//kernel_size * output_size[1]//kernel_size
    fold_params = dict(output_size=output_size, kernel_size=kernel_size, dilation=1, padding=padding, stride=kernel_size)
    
    x_patches = x_patches.reshape((-1, L, C*kernel_size*kernel_size)).permute(0,2,1)
    fold = torch.nn.Fold(**fold_params)
    x = fold(x_patches)
    return x


import importlib
def instantiate_from_config(config):
    if not "target" in config:
        raise KeyError("Expected key `class_path` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)

