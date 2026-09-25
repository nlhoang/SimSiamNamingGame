import torch
import torch.nn as nn
import torch.nn.functional as F
from base.cifar_resnet import resnet18 as cifar_resnet18
from base.cifar_resnet import resnet34 as cifar_resnet34
from torchvision.models import (
    resnet18, resnet34, resnet50, resnet101,
    ResNet18_Weights, ResNet34_Weights, ResNet50_Weights, ResNet101_Weights
)


class Encoder(nn.Module):
    def __init__(
            self,
            backbone='cnn-img',
            pretrained=False,
            weights=None,           # torchvision weights OR None
            ckpt_path=None,         # SSL checkpoint OR None
            freeze_backbone=False,
    ):
        super(Encoder, self).__init__()

        if not pretrained:
            weights = None
            ckpt_path = None

        if pretrained and weights is None and ckpt_path is None:
            if backbone == "resnet18":
                weights = ResNet18_Weights.DEFAULT
            elif backbone == "resnet34":
                weights = ResNet34_Weights.DEFAULT
            elif backbone == "resnet50":
                weights = ResNet50_Weights.DEFAULT
            elif backbone == "resnet101":
                weights = ResNet101_Weights.DEFAULT

        if backbone == 'resnet18':
            self.resnet = resnet18(weights=weights, zero_init_residual=True)
            self.resnet.fc = nn.Identity()
            self.backbone = self.resnet
            if pretrained:
                print(f'Load ResNet18 with weights={weights}')

        elif backbone == 'resnet34':
            self.resnet = resnet34(weights=weights, zero_init_residual=True)
            self.resnet.fc = nn.Identity()
            self.backbone = self.resnet
            if pretrained:
                print(f'Load ResNet34 with weights={weights}')

        elif backbone == 'resnet50':
            self.resnet = resnet50(weights=weights, zero_init_residual=True)
            self.resnet.fc = nn.Identity()
            self.backbone = self.resnet
            if pretrained:
                print(f'Load ResNet50 with weights={weights}')

        elif backbone == 'resnet101':
            self.resnet = resnet101(weights=weights, zero_init_residual=True)
            self.resnet.fc = nn.Identity()
            self.backbone = self.resnet
            if pretrained:
                print(f'Load ResNet101 with weights={weights}')

        elif backbone == 'resnet_cifar18':
            self.resnet = cifar_resnet18(pretrained=pretrained, zero_init_residual=True)
            self.resnet.fc = nn.Identity()
            self.backbone = self.resnet
        elif backbone == 'resnet_cifar34':
            self.resnet = cifar_resnet34(pretrained=pretrained, zero_init_residual=True)
            self.resnet.fc = nn.Identity()
            self.backbone = self.resnet

        elif backbone == 'cnn-img':
            self.backbone = CNN_Img()
        elif backbone == 'cnn2-img':
            self.backbone = CNN2_Img()

        elif backbone == 'generic-mlp-MNIST-1':
            self.backbone = GenericMLP_Enc(input_dim=28*28, feature_dim=512, hidden_dim=1024, num_layers=2)
        elif backbone == 'generic-mlp-MNIST-2':
            self.backbone = GenericMLP_Enc(input_dim=28*28, feature_dim=512, hidden_dim=1024, num_layers=3)
        elif backbone == 'generic-cnn-MNIST-1':
            self.backbone = GenericCNN_MNIST(channels=[16, 32], output_dim=512)
        elif backbone == 'generic-cnn-MNIST-2':
            self.backbone = GenericCNN_MNIST(channels=[16, 32, 64], output_dim=512)

        elif backbone == 'generic-mlp-dSprites-1':
            self.backbone = GenericMLP_Enc(input_dim=64*64, feature_dim=512, hidden_dim=1024, num_layers=2)
        elif backbone == 'generic-mlp-dSprites-2':
            self.backbone = GenericMLP_Enc(input_dim=64*64, feature_dim=512, hidden_dim=1024, num_layers=3)
        elif backbone == 'generic-mlp-dSprites-3':
            self.backbone = GenericMLP_Enc(input_dim=64*64, feature_dim=512, hidden_dim=1024, num_layers=4)

        elif backbone == 'cnn-cifar-1':
            self.backbone = GenericCNN_CIFAR(d=512, in_channels=3, channels=(64, 128, 256),
                                             blocks_per_stage=(2, 2, 2),
                                             use_bn=True, l2_normalize=True, pool_type="avg")
        elif backbone == 'cnn-cifar-2':
            self.backbone = GenericCNN_CIFAR(d=256, in_channels=3, channels=(64, 128),
                                             blocks_per_stage=(1, 1),
                                             use_bn=True, l2_normalize=True, pool_type="avg")
        elif backbone == 'cnn-cifar-3':
            self.backbone = GenericCNN_CIFAR(d=512, in_channels=3, channels=(64, 128, 256, 512),
                                             blocks_per_stage=(2, 2, 2, 2),
                                             use_bn=True, l2_normalize=True, pool_type="avg")

        else:
            print('Backbone not supported')

        if ckpt_path is not None:
            ckpt = torch.load(ckpt_path, map_location="cpu")
            state_dict = ckpt.get("state_dict", ckpt)
            self.backbone.load_state_dict(state_dict, strict=False)
            print(f"Loaded ResNet weights from {ckpt_path}")

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def forward(self, x):
        x = self.backbone(x)
        return x

#--------------------------------------

class GenericMLP(nn.Module):
    def __init__(
            self,
            input_dim: int,             # input dimension
            hidden_dim: int = None,     # hidden dimension
            output_dim: int = None,     # output dim (or latent dim if output_gaussian=True)
            num_layers: int = 1,        # number of hidden layers
            last_relu: bool = False,    # apply ReLU on final output layer
            last_bn: bool = False,      # apply BN on final output layer
            output_gaussian: bool = False,  # output mu, logvar
    ):
        super(GenericMLP, self).__init__()
        self.output_gaussian = output_gaussian

        if hidden_dim is None:
            hidden_dim = input_dim // 2
        if output_dim is None:
            output_dim = hidden_dim

        # if gaussian: output_dim is interpreted as latent_dim
        self.latent_dim = output_dim if output_gaussian else None
        final_out_dim = 2 * output_dim if output_gaussian else output_dim

        layers = []
        in_dim = input_dim

        # ----- hidden layers -----
        for i in range(num_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())  # always ReLU for hidden layers
            in_dim = hidden_dim

        # ----- final output layer -----
        layers.append(nn.Linear(in_dim, final_out_dim))

        # For Gaussian mu/logvar, we don't want BN/ReLU on top of them
        if not output_gaussian:
            if last_bn:
                layers.append(nn.BatchNorm1d(final_out_dim))
            if last_relu:
                layers.append(nn.ReLU())

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        out = self.net(x)  # if gaussian: shape (B, 2 * latent_dim)

        if self.output_gaussian:
            mu, logvar = torch.chunk(out, 2, dim=-1)  # each (B, latent_dim)
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z = mu + eps * std
            return z, mu, logvar
        else:
            return out


class GenericMLP_Enc(nn.Module):
    def __init__(
            self,
            feature_dim: int,
            input_dim: int = 28 * 28,
            hidden_dim: int = None,
            num_layers: int = None,     # number of hidden layers
            last_relu: bool = False,    # apply ReLU on last hidden layer
            last_bn: bool = False       # apply BN on final output layer
    ):
        super().__init__()
        self.flatten = nn.Flatten()
        self.encoder = GenericMLP(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=feature_dim,
            num_layers=num_layers,
            last_relu=last_relu,
            last_bn=last_bn
        )

    def forward(self, x):
        x = self.flatten(x)
        return self.encoder(x)


class GenericCNN_MNIST(nn.Module):
    def __init__(
        self,
        in_channels=1,
        channels=[16, 32],      # number of conv layers = len(channels)
        kernel_size=4,
        stride=2,
        padding=1,
        output_dim=512          # final feature dimension
    ):
        super().__init__()

        layers = []
        prev_c = in_channels

        # ----- conv layers -----
        for c in channels:
            layers.append(nn.Conv2d(prev_c, c, kernel_size=kernel_size, stride=stride, padding=padding))
            layers.append(nn.ReLU())
            prev_c = c

        self.conv = nn.Sequential(*layers)

        x = torch.zeros(1, 1, 28, 28)
        with torch.no_grad():
            h = self.conv(x)
        conv_output_dim = h.numel()

        # ----- final linear layer -----
        self.fc = nn.Linear(conv_output_dim, output_dim)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class GenericCNN_CIFAR(nn.Module):
    def __init__(
        self,
        d=512,
        in_channels=3,
        channels=(64, 128, 256),      # channels per stage
        blocks_per_stage=(2, 2, 2),   # conv layers per stage
        use_bn=True,
        l2_normalize=True,
        pool_type="avg",              # "avg" or "max"
    ):
        super().__init__()

        assert len(channels) == len(blocks_per_stage), \
            "channels and blocks_per_stage must have same length"

        layers = []
        curr_in = in_channels

        for stage_idx, (c_out, n_blocks) in enumerate(zip(channels, blocks_per_stage)):
            for b in range(n_blocks):
                layers.append(nn.Conv2d(curr_in, c_out, kernel_size=3, padding=1))
                if use_bn:
                    layers.append(nn.BatchNorm2d(c_out))
                layers.append(nn.ReLU(inplace=True))
                curr_in = c_out

            # Add spatial downsampling between stages (except after last stage)
            if stage_idx < len(channels) - 1:
                layers.append(nn.MaxPool2d(kernel_size=2))

        self.features = nn.Sequential(*layers)

        if pool_type == "avg":
            self.pool = nn.AdaptiveAvgPool2d(1)
        elif pool_type == "max":
            self.pool = nn.AdaptiveMaxPool2d(1)
        else:
            raise ValueError("pool_type must be 'avg' or 'max'")

        self.fc = nn.Linear(channels[-1], d)
        self.l2_normalize = l2_normalize

        # Weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x, return_featmap=False):
        x = self.features(x)
        featmap = x
        x = self.pool(x).flatten(1)
        z = self.fc(x)
        if self.l2_normalize:
            z = F.normalize(z, dim=1)
        return (z, featmap) if return_featmap else z

#--------------------------------------

class CNN_Img(nn.Module):  # image of shape [3, 64, 64], output: 512
    def __init__(self):
        super(CNN_Img, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(512)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return x


class CNN2_Img(nn.Module):  # input [3, 64, 64], output: 512
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, stride=2, padding=1)    # 64x64 → 32x32
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, 3, stride=2, padding=1)  # 32x32 → 16x16
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, 3, stride=2, padding=1) # 16x16 → 8x8
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 512, 3, stride=2, padding=1) # 8x8 → 4x4
        self.bn4 = nn.BatchNorm2d(512)
        self.conv5 = nn.Conv2d(512, 512, 3, stride=1, padding=1) # keep 4x4
        self.bn5 = nn.BatchNorm2d(512)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return x

