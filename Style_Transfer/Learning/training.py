import os
import time

import torch

from matplotlib import pyplot as plt

from torchvision import models

import images
import objectives

from tqdm import tqdm

SIZE = 512

STYLE_LAYERS = [
    'conv_1',
    'conv_2',
    'conv_3',
    'conv_4',
    'conv_5'
]

def train(model, objective, param, transforms = None, epochs = 100, lr = 0.01, verbose = True, ultraverbose = False):
    """
    This function trains some parameters to minimize some objective.

    model : the model we are examining through feature visualization

    objective : the objective we are trying to minimize.
                e.g. the mean activation of a neuron, or the mean activation of a channel
                or darker stuff for demonic blend.
    
    param : the parameters we are trying to optimize.
            e.g. an image we want to train.
    
    transforms : a list of transforms to apply to the parameters before feeding them to the model.
                 when training an image this helps for resilience to noise.
                
    epochs : the number of epochs to train for.

    verbose : whether to print the loss over time.
    """

    param.train()

    if verbose:
        losses = []
    
    parameters = param.parameters()
    opt = torch.optim.Adam(parameters, lr=lr)

    for epoch in tqdm(range(epochs)) if verbose else range(epochs):
        opt.zero_grad()
        loss = objective(model, param, transforms)
        loss.backward()
        opt.step()
        if verbose:
            losses.append(loss.item())

        if epoch / epochs * 10 % 1 == 0 and ultraverbose:
            image = param()
            print(image.min(), image.max())
            images.imshow(image)
    
    if verbose:
        fig, ax = plt.subplots()
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Loss over time')
        ax.plot(losses)
        plt.show()
    
    return param

def visualise(model, objective, param, epochs=500, lr=10, verbose=True, ultraverbose=False, save=None):
    transforms = images.preprocess_image

    image = train(model, objective, param, transforms=transforms, epochs=epochs, lr = lr, verbose=verbose, ultraverbose=ultraverbose)
    
    images.imshow(image())
    if save is not None:
        images.save_image(image(), save)

def run_correlation(base_image_path, style_image_path):
    if isinstance(style_image_path, str):
        style_img = images.ExistingImage(style_image_path, resize=SIZE, train=False)().detach()
    else:
        style_img = []
        for path in style_image_path:
            style_img.append(images.ExistingImage(path, resize=SIZE, train=False)().detach())

    blend_img = images.ExistingImage(base_image_path, resize=SIZE)

    model = models.vgg19(pretrained=True).features.eval()
    model = objectives.get_truncated_model(model, STYLE_LAYERS).eval()
    model.requires_grad_(False)
    
    style_obj = objectives.stream_difference(model, STYLE_LAYERS, style_img, transform=objectives.feature_correlation_matrix)
    
    timestr = time.strftime("%Y%m%d-%H%M%S")
    output_dir = "outputs"
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    visualise(model, style_obj, blend_img, lr=1e-2, epochs=100, save = f"{output_dir}/{timestr}.jpg")

def run_feature(base_image_path, style_image_path):
    if isinstance(style_image_path, str):
        style_img = images.ExistingImage(style_image_path, resize=SIZE, train=False)().detach()
    else:
        style_img = []
        for path in style_image_path:
            style_img.append(images.ExistingImage(path, resize=SIZE, train=False)().detach())

    blend_img = images.ExistingImage(base_image_path, resize=SIZE)

    model = models.vgg19(pretrained=True).features.eval()
    model = objectives.get_truncated_model(model, STYLE_LAYERS).eval() #TODO : try trucate after the relu
    model.requires_grad_(False)
    
    style_obj = objectives.match_wct(model, STYLE_LAYERS, blend_img, style_img)
    
    timestr = time.strftime("%Y%m%d-%H%M%S")
    output_dir = "outputs"
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    visualise(model, style_obj, blend_img, lr=1e-2, epochs=100, save = f"{output_dir}/{timestr}.jpg")