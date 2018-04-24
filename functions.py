from PIL import Image
import numpy as np
from network import Network
import torch
import json


def process_image(filename):
    im = Image.open(filename)
    im = im.resize((256, 256))
    im = im.crop((16, 16, 240, 240))
    im = np.array(im) / 255

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    im = (im - mean) / std

    return im.transpose(2, 0, 1)


def hydrate(checkpoint):
    state = torch.load(checkpoint + '/checkpoint.pth')
    n = Network(state['arch'], state['units'], state['lr'])
    n.optimizer.load_state_dict(state['optimizer_state'])
    n.model.load_state_dict(state['model_state'])
    n.model.class_to_idx = state['class_to_idx']
    return n


def cat_to_name(filename):
    with open(filename) as f:
        return json.load(f)
