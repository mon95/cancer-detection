
import torch
import argparse
import torch.nn as nn
from torch.utils import data
from torchvision.models import vgg19
from torchvision.models import densenet201
from torchvision import transforms
from torchvision import datasets
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os

class DenseNet(nn.Module):
    def __init__(self, model):
        super(DenseNet, self).__init__()
        self.densenet = model
        self.features_conv = self.densenet.features
        self.global_avg_pool = nn.AvgPool2d(kernel_size=7, stride=1)
        self.classifier = self.densenet.classifier
        self.gradients = None
    

    def activations_hook(self, grad):
        self.gradients = grad
        

    def forward(self, x):
        x = self.features_conv(x)
        h = x.register_hook(self.activations_hook)
        x = self.global_avg_pool(x)
        x = x.view((1, 1024))
        x = self.classifier(x)
        return x
    

    def get_activations_gradient(self):
        return self.gradients
    

    def get_activations(self, x):
        return self.features_conv(x)


def run_inference(model, dataloader):
    ds = DenseNet(model)
    ds.eval()
    img, _ = next(iter(dataloader))
    scores = ds(img)
    label = torch.argmax(scores)
    return ds, img, scores, label


def get_grad_cam(ds, img, scores, label):
    scores[:, label].backward(retain_graph=True)
    gradients = ds.get_activations_gradient()
    pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])
    activations = ds.get_activations(img).detach()
    for i in range(1024):
        activations[:, i, :, :] *= pooled_gradients[i]
    heatmap = torch.mean(activations, dim=1).squeeze()
    heatmap = np.maximum(heatmap, 0)
    heatmap /= torch.max(heatmap)
    #plt.matshow(heatmap.squeeze())
    return heatmap

def render_superimposition(root_dir, heatmap, image):
    print(os.path.join(root_dir, 'images', image))
    img = cv2.imread(os.path.join(root_dir, 'images', image))
    heatmap = heatmap.numpy()
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = heatmap * 0.4 + img
    cv2.imwrite(root_dir + '/superimposed_' + image, superimposed_img)
    cv2.imshow('output', superimposed_img)
