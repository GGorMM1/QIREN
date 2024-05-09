from PIL import Image
from torchvision.transforms import Resize, Compose, ToTensor, Normalize, CenterCrop

import skimage
from torch.utils.data import Dataset
import torch


def get_mgrid(sidelen, dim=2):
    '''Generates a flattened grid of (x,y,...) coordinates in a range of -1 to 1.
    sidelen: int
    dim: int'''
    tensors = tuple(dim * [torch.linspace(-1, 1, steps=sidelen)])
    mgrid = torch.stack(torch.meshgrid(*tensors), dim=-1)
    mgrid = mgrid.reshape(-1, dim)
    return mgrid


def get_cameraman_tensor(sidelength):
    #img = Image.fromarray(skimage.data.camera())
    img = Image.fromarray(skimage.color.rgb2gray(skimage.data.coffee()))
    #img = Image.fromarray(skimage.color.rgb2gray(skimage.data.astronaut()))

    transform = Compose([
        Resize(sidelength),
        CenterCrop(sidelength),
        ToTensor(),
        Normalize(torch.Tensor([0.5]), torch.Tensor([0.5]))
    ])
    img = transform(img)
    return img


class ImageFitting(Dataset):
    def __init__(self, sidelength):
        super().__init__()
        img = get_cameraman_tensor(sidelength)
        self.pixels = img.permute(1, 2, 0).view(-1, 1)
        self.coords = get_mgrid(sidelength, 2)

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        if idx > 0: raise IndexError

        return self.coords, self.pixels
