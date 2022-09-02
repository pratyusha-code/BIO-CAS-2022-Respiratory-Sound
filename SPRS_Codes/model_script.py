import torch.nn as nn
from torchvision import models


def get_pretrained_model(model_name, n_classes, multi_gpu, train_on_gpu, device):
    """
    Used to retrieve pre trained VGG16 or Resnet50 with customc lassifier
    """

    if model_name == 'vgg16':
        model = models.vgg16(pretrained=True)

        # Freeze early layers
        for param in model.parameters():
            param.requires_grad = False
        n_inputs = model.classifier[0].in_features

        # Add on classifier
        model.classifier = nn.Sequential(
            nn.Linear(n_inputs, 256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, n_classes), nn.LogSoftmax(dim=1))

    elif model_name == 'resnet50':
        model = models.resnet50(pretrained=True)

        for param in model.parameters():
            param.requires_grad = False

        n_inputs = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Linear(n_inputs, 256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, n_classes), nn.LogSoftmax(dim=1))
        
    elif model_name == 'mobilenet':
        model = models.mobilenet_v2(pretrained=True)

        for param in model.parameters():
            param.requires_grad = False
        n_inputs = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(0.2), nn.Linear(n_inputs, 256), nn.ReLU(), 
            nn.Dropout(0.2), nn.Linear(256, n_classes), nn.LogSoftmax(dim=1))

    # Move to gpu and parallelize
    if multi_gpu:
        model = nn.DataParallel(model)
    if train_on_gpu:
        model = model.to(device)

    return model