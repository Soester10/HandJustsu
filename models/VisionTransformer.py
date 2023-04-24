import torchvision
import torchvision.models.vit_b_16 as vit_b_16

##Test
from pytorch_pretrained_vit import ViT


# vit_b_16
def vit_model1():
    model = vit_b_16(weights=torchvision.models.ViT_B_16_Weights.IMAGENET1K_V1)
    return model


def vit_model2():
    model = ViT("B_16_imagenet1k", pretrained=True)
    return model
