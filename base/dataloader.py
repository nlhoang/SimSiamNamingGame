import os
import random

from PIL import Image, ImageFilter
from torch.utils.data import Dataset
from torchvision import transforms, datasets

#-----------------------------

class CropsTransform:
    """Take N random crops of one image as different views."""
    def __init__(self, base_transform, n_views=2):
        self.base_transform = base_transform
        self.n_views = n_views

    def __call__(self, x):
        return [self.base_transform(x) for _ in range(self.n_views)]


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""
    def __init__(self, sigma=None):
        if sigma is None:
            sigma = [.1, 2.]
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


class AgentViewsTransform:
    """
    Return one view per agent.
    The number of agents = n_views.
    Each agent has a fixed, different augmentation.
    """
    def __init__(self, mean, std, image_size=224, n_views=2):
        assert 2 <= n_views <= 5, "Supported number of agents: 2 to 5"
        self.n_views = n_views

        # ---------- shared parts ----------
        self.to_tensor_norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

        self.base_full = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(image_size),
        ])

        # ---------- agent-specific transforms ----------
        self.transforms_bank = []

        # 1) Grayscale first
        grayscale = transforms.Compose([
            self.base_full,
            transforms.RandomGrayscale(p=1.0),
            self.to_tensor_norm,
        ])

        # 5) Mild blur
        blur = transforms.Compose([
            self.base_full,
            GaussianBlur(sigma=[0.1, 1.0]),
            self.to_tensor_norm,
        ])

        # 4) Tone agent
        tone = transforms.Compose([
            self.base_full,
            transforms.RandomAutocontrast(p=1.0),
            self.to_tensor_norm,
        ])

        # 3) Color agent, but mild (so it adds value when introduced)
        color = transforms.Compose([
            self.base_full,
            transforms.ColorJitter(0.3, 0.3, 0.3, 0.1),
            self.to_tensor_norm,
        ])

        # 2) Crop but less informative than before
        viewpoint = transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.55, 0.85)),
            transforms.RandomHorizontalFlip(p=0.5),
            self.to_tensor_norm,
        ])

        self.transforms_bank = [grayscale, blur, tone, color, viewpoint]

    def __call__(self, x):
        return [self.transforms_bank[i](x) for i in range(self.n_views)]

#-----------------------------

class GenericMNISTDataset(Dataset):
    def __init__(self, path, train=True, augmented=True, n_views=2, name="MNIST"):
        self.train = train
        self.augmented = augmented
        if name == "MNIST":
            self.data = datasets.MNIST(root=path, train=train, download=True, transform=None)
        elif name == "FashionMNIST":
            self.data = datasets.FashionMNIST(root=path, train=train, download=True, transform=None)
        elif name == "KMNIST":
            self.data = datasets.KMNIST(root=path, train=train, download=True, transform=None)
        else:
            raise ValueError(f"Unknown dataset name: {name}")

        if self.augmented:
            self.transform = CropsTransform(transforms.Compose([
                transforms.RandomResizedCrop(size=28, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                transforms.ToTensor(),
                #transforms.Normalize(mean=[0.5], std=[0.5])
            ]), n_views=n_views)

        else:
            self.transform = transforms.Compose([
                transforms.Resize(28),
                transforms.CenterCrop(28),
                transforms.ToTensor(),
                #transforms.Normalize(mean=[0.5], std=[0.5])
            ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img, target = self.data[idx]

        if self.train and self.augmented:
            views = self.transform(img)
            return views
        else:
            img = self.transform(img)
            return img, target

#-----------------------------

class GenericCIFARDataset(Dataset):
    STATS = {
        "CIFAR10": {
            "mean": [0.4914, 0.4822, 0.4465],
            "std": [0.247, 0.243, 0.261],
            "cls": datasets.CIFAR10,
        },
        "CIFAR100": {
            "mean": [0.5071, 0.4867, 0.4408],
            "std": [0.2675, 0.2565, 0.2761],
            "cls": datasets.CIFAR100,
        },
    }

    def __init__(self, path, train=True, augmented=True, dif_augmented=True,
                 name="CIFAR10", image_size=224, n_views=2):
        self.train = train
        self.augmented = augmented

        info = self.STATS[name]
        mean, std = info["mean"], info["std"]
        ds_cls = info["cls"]
        self.data = ds_cls(root=path, train=train, download=True, transform=None)

        if self.augmented:
            if dif_augmented:
                self.transform = AgentViewsTransform(
                    mean=mean, std=std, image_size=image_size,
                    n_views=n_views
                )

            else:
                self.transform = CropsTransform(transforms.Compose([
                    transforms.RandomResizedCrop(size=image_size, scale=(0.2, 1.0)),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomApply([
                        transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
                    ], p=0.8),
                    transforms.RandomGrayscale(p=0.2),
                    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=mean, std=std)
                ]), n_views=n_views)

        else:
            self.transform = transforms.Compose([
                transforms.Resize(image_size),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)
            ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img, target = self.data[idx]

        if self.train and self.augmented:
            views = self.transform(img)
            return views
        else:
            img = self.transform(img)
            return img, target

# -----------------------------

class ImageNet100HFDataset(Dataset):
    def __init__(self, hf_dataset, train=True, augmented=True, n_views=2, selected_class=None):
        """
        hf_dataset: HuggingFace dataset split (train or val)
        train: True for training mode
        augmented: True for SimCLR-style augmentation
        selected_class: list of original label IDs to keep (e.g. 20 classes)
                        If None, keep all 100 classes.
        select20 = [95, 35, 4, 70, 20, 48, 10, 51, 54, 15, 29, 67, 37, 50, 12, 49, 83, 11, 42, 66]
        """
        self.train = train
        self.augmented = augmented

        # -------------------------------
        # 1. Filter dataset by selected class
        # -------------------------------
        if selected_class is not None:
            selected_class = sorted(selected_class)               # sort class IDs
            self.class_map = {orig: new for new, orig in enumerate(selected_class)}
            # Keep only samples whose label is in selected_class
            self.filtered_indices = [
                i for i, item in enumerate(hf_dataset)
                if item["label"] in self.class_map
            ]
        else:
            self.class_map = None
            self.filtered_indices = list(range(len(hf_dataset)))

        self.dataset = hf_dataset

        # -------------------------------
        # 2. Define transforms
        # -------------------------------
        if self.augmented:
            self.transform = CropsTransform(transforms.Compose([
                transforms.Resize(256),
                transforms.RandomResizedCrop(224, scale=(0.2, 1.0), interpolation=Image.BICUBIC),
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply([
                    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
                ], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                GaussianBlur(sigma=[0.1, 2.0]),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
            ]), n_views=n_views)
        else:
            self.transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
            ])

    def __len__(self):
        return len(self.filtered_indices)

    def __getitem__(self, idx):
        real_idx = self.filtered_indices[idx]
        item = self.dataset[real_idx]

        img = item['image']

        # Ensure PIL Image and convert to RGB
        if not isinstance(img, Image.Image):
            img = Image.fromarray(img)
        img = img.convert('RGB')

        # ------------------------------------
        # Remap label (old -> new)
        # ------------------------------------
        label = item["label"]
        if self.class_map is not None:
            label = self.class_map[label]   # new label 0..len(selected_class)-1

        if self.train and self.augmented:
            views = self.transform(img)
            return views
        else:
            img = self.transform(img)
            return img, label

#-----------------------------

if __name__ == "__main__":
    import pandas as pd
    from datasets import load_dataset

    df = pd.read_csv("imagenet100_order.csv")
    top20_ids = df.head(20)["ID"].tolist()
    print(top20_ids)

    full_dataset = load_dataset("clane9/imagenet-100")

    train_dataset = ImageNet100HFDataset(
        full_dataset["train"],
        train=True,
        augmented=True,
        selected_class=top20_ids,  # filter & remap
    )
    print(len(train_dataset))
    print(train_dataset[0])

    val_dataset = ImageNet100HFDataset(
        full_dataset["validation"],
        train=False,
        augmented=False,
        selected_class=top20_ids,
    )
    print(len(val_dataset))
    print(val_dataset[0])