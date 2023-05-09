import torchvision

##Test
from pytorch_pretrained_vit import ViT
from vit_pytorch.vivit import ViT as ViViT


##TODO: change model head to take input a smaller size
# vit_b_16
def vit_model1():
    model = torchvision.models.vit_b_16(
        weights=torchvision.models.ViT_B_16_Weights.IMAGENET1K_V1
    )
    return model


def vit_model2():
    model = ViT("B_16_imagenet1k", pretrained=True)
    return model


def vivit_model1():
    model = ViViT(
        image_size=200,  # image size
        frames=40,  # number of frames
        image_patch_size=20,  # image patch size
        frame_patch_size=2,  # frame patch size
        num_classes=2000,
        dim=1024,
        spatial_depth=6,  # depth of the spatial transformer
        temporal_depth=6,  # depth of the temporal transformer
        heads=8,
        mlp_dim=2048,
    )
    return model
