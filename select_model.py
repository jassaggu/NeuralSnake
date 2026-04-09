import torch

from train_transformer_model import TransformerWorldModel
from train_UNet_model import ResidualUNetWorldModel
from train_baseline_model import BaselineCNNModel


def load_model(model_name, device):
    if model_name == "transformer":
        model = TransformerWorldModel()
        model.load_state_dict(torch.load("transformer_world_model.pt", map_location=device))
    elif model_name == "unet":
        model = ResidualUNetWorldModel()
        model.load_state_dict(torch.load("unet_world_model.pt", map_location=device))
    elif model_name == "baseline":
        model = BaselineCNNModel()
        model.load_state_dict(torch.load("baseline_cnn_world_model.pt", map_location=device))
    else:
        raise Exception("Model incorrectly set")

    model.to(device)
    model.eval()

    print(f"{model_name} succesfully loaded on {device}.")
    return model
