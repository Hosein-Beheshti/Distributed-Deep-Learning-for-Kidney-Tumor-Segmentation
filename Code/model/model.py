import segmentation_models_pytorch as smp
import torch

class UNet:
    def __init__(self, classes, pretrained_weights_path=None):
        self.classes = classes
        self.model = smp.Unet("resnet50", encoder_weights=None, in_channels=1, classes=classes)

        if pretrained_weights_path is not None:
            self.load_pretrained_weights(pretrained_weights_path)

    def load_pretrained_weights(self, pretrained_weights_path):
        state_dict = torch.load(pretrained_weights_path)
        self.model.load_state_dict(state_dict, strict=False)

    def get_model(self):
        return self.model
