import numpy as np
import torch.nn as nn
from torchvision import models

from sklearn.ensemble import RandomForestClassifier


def get_ensemble_model(cnn_model, n_classes, multi_gpu, train_on_gpu, device):

    rfc = RandomForestClassifier(n_estimators=100)
    rfc_final = RandomForestClassifier(n_estimators=100)
    models=[cnn_model, rfc, rfc_final]
    return models
