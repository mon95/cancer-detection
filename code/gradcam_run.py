import torch
import torch.nn as nn
from torch.utils import data
from torchvision.models import vgg19
from torchvision.models import densenet201
from torchvision.models import densenet121
from torchvision import transforms
from torchvision import datasets
import matplotlib.pyplot as plt
import argparse
import numpy as np
import cv2
import gradcam_utils
import os

def fetch_and_fit_model():
    model_ft = densenet121(pretrained=True)
    num_ftrs = model_ft.classifier.in_features
    model_ft.classifier = nn.Linear(num_ftrs, 2)
    state = torch.load('../trained_models/densenetFeatureExtraction_l_0.001_m_0.9.pt')
    model_ft.load_state_dict(state['model_state_dict'])
    return model_ft


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', dest='input_directory', action='store', default=os.getcwd(), required=False, help='The root directory containing the image. Use "./" for current directory')
    args = parser.parse_args()

    input_directory = args.input_directory
    transform = transforms.Compose([transforms.Resize((224, 224)), 
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    dataset = datasets.ImageFolder(root=os.getcwd(), transform=transform)
    dataloader = data.DataLoader(dataset=dataset, shuffle=False, batch_size=1)
    images = []
    for filename in os.listdir(os.path.join(input_directory, 'images')):
        if filename.endswith('.jpg'):
            images.append(filename)

    model_gradcam = fetch_and_fit_model()
    ds, img, scores, label = gradcam_utils.run_inference(model_gradcam, dataloader)
    heatmap = gradcam_utils.get_grad_cam(ds, img, scores, label)
    for image in images:
        gradcam_utils.render_superimposition(input_directory, heatmap, image)
    print(scores)
    print(label)