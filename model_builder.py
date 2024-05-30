import timm
import torch.nn as nn

class ResNet18(nn.Module):
    def __init__(self, num_classes):
        super(ResNet18, self).__init__()
        self.resnet18 = timm.create_model('resnet18', pretrained=True)
        self.resnet18.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.resnet18.fc = nn.Linear(in_features=512, out_features=num_classes, bias=True)

    def forward(self, x):
        return self.resnet18(x)

class ResNet50(nn.Module):
    def __init__(self, num_classes):
        super(ResNet18, self).__init__()
        self.resnet50 = timm.create_model('resnet50.a1_in1k', pretrained=True)
        self.resnet50.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.resnet50.fc = nn.Linear(in_features=2048, out_features=num_classes, bias=True)

    def forward(self, x):
        return self.resnet50(x)

class VITBase(nn.Module):
    def __init__(self, num_classes):
        super(VITBase, self).__init__()
        self.vit_base = timm.create_model('vit_base_patch16_224.augreg2_in21k_ft_in1k', pretrained=True)
        self.vit_base.patch_embed.proj = nn.Conv2d(1, 768, kernel_size=(16, 16), stride=(16, 16))
        self.vit_base.head = nn.Linear(in_features=768, out_features=num_classes, bias=True)

    def forward(self, x):
        return self.vit_base(x)

class Mixer(nn.Module):
    def __init__(self, num_classes):
        super(Mixer, self).__init__()
        self.mixer = timm.create_model('mixer_b16_224.goog_in21k_ft_in1k', pretrained=True)
        self.mixer.stem.proj = nn.Conv2d(1, 768, kernel_size=(16, 16), stride=(16, 16))
        self.mixer.head = nn.Linear(in_features=768, out_features=num_classes, bias=True)

    def forward(self, x):
        return self.vit_base(x)
