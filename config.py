import random
import torch
import numpy as np
from torch.utils.data import DataLoader
from datasets import load_dataset
from base.dataloader import GenericMNISTDataset, GenericCIFARDataset, ImageNet100HFDataset


path = '/home/nlhoang/MachineLearning/data/'
ImgNet_select = list(range(1, 21))


def configurations(args):
    if args.dataset in ['FashionMNIST', 'MNIST']:
        args.image_size = 28
        args.backbone = 'generic-mlp-MNIST-1'
        args.backbone2 = 'generic-cnn-MNIST-1'
        args.backbone3 = 'generic-mlp-MNIST-2'
        args.backbone4 = 'generic-cnn-MNIST-2'
        args.feature_dim = 512
        args.latent_dim = 256
        args.num_classes = 10
        args.batch_size = 128
        args.learning_rate = 0.01
        args.epochs = 100
        args.alpha = 1.0
        args.beta = 1.0
        args.gamma = 1.0
        args.word_length = 10
        args.dictionary_size = 100
        args.pretrained_backbone = False
        args.freeze_backbone = False

    elif args.dataset in ['KMNIST']:
        args.image_size = 28
        args.backbone = 'generic-mlp-MNIST-2'
        args.backbone2 = 'generic-cnn-MNIST-2'
        args.feature_dim = 512
        args.latent_dim = 256
        args.num_classes = 10
        args.batch_size = 128
        args.learning_rate = 0.01
        args.epochs = 100
        args.alpha = 1.0
        args.beta = 1.0
        args.gamma = 1.0
        args.word_length = 10
        args.dictionary_size = 100
        args.pretrained_backbone = False
        args.freeze_backbone = False

    elif args.dataset in ['CIFAR', 'CIFAR100']:
        args.image_size = 224
        args.backbone = 'resnet18'
        args.backbone2 = 'resnet34'
        args.backbone3 = 'resnet18'
        args.backbone4 = 'resnet34'
        args.backbone5 = 'resnet18'
        args.feature_dim = 512
        args.latent_dim = 2048
        args.num_classes = 10 if args.dataset == 'CIFAR' else 100
        args.batch_size = 256
        args.learning_rate = 0.05
        args.epochs = 200
        args.alpha = 1.0
        args.beta = 1.0
        args.gamma = 1.0
        args.word_length = 10
        args.dictionary_size = 100
        args.pretrained_backbone = False
        args.freeze_backbone = False
        args.dif_augmented = False

    elif args.dataset == 'ImageNet100':
        args.image_size = 224
        args.backbone = 'resnet18'
        args.backbone2 = 'resnet34'
        args.backbone3 = 'resnet18'
        args.backbone4 = 'resnet34'
        args.backbone5 = 'resnet18'
        args.feature_dim = 512
        args.latent_dim = 2048
        args.num_classes = 20
        args.batch_size = 128
        args.learning_rate = 0.02
        args.alpha = 0.3
        args.beta = 0.3
        args.gamma = 0.4
        args.epochs = 200
        args.word_length = 10
        args.dictionary_size = 1000
        args.pretrained_backbone = True
        args.freeze_backbone = False

    else:
        print('Unknown dataset:', args.dataset)


def get_dataloader_eval(args):
    if args.dataset == 'FashionMNIST':
        eval_dataset_train = GenericMNISTDataset(path=path, train=True, augmented=False,
                                                 n_views=args.num_of_agent, name="FashionMNIST")
        eval_dataset_test = GenericMNISTDataset(path=path, train=False, augmented=False,
                                                n_views=args.num_of_agent, name="FashionMNIST")
    elif args.dataset == 'MNIST':
        eval_dataset_train = GenericMNISTDataset(path=path, train=True, augmented=False,
                                                 n_views=args.num_of_agent, name="MNIST")
        eval_dataset_test = GenericMNISTDataset(path=path, train=False, augmented=False,
                                                n_views=args.num_of_agent, name="MNIST")

    elif args.dataset == 'CIFAR':
        eval_dataset_train = GenericCIFARDataset(path=path, train=True, augmented=False, image_size=args.image_size,
                                                 n_views=args.num_of_agent, name="CIFAR10")
        eval_dataset_test = GenericCIFARDataset(path=path, train=False, augmented=False, image_size=args.image_size,
                                                n_views=args.num_of_agent, name="CIFAR10")
    elif args.dataset == 'CIFAR100':
        eval_dataset_train = GenericCIFARDataset(path=path, train=True, augmented=False, image_size=args.image_size,
                                                 n_views=args.num_of_agent, name="CIFAR100")
        eval_dataset_test = GenericCIFARDataset(path=path, train=False, augmented=False, image_size=args.image_size,
                                                n_views=args.num_of_agent, name="CIFAR100")

    elif args.dataset == 'ImageNet100':
        full_dataset = load_dataset("clane9/imagenet-100")
        eval_dataset_train = ImageNet100HFDataset(full_dataset['train'], train=False, augmented=False,
                                                  n_views=args.num_of_agent, selected_class=ImgNet_select)
        eval_dataset_test = ImageNet100HFDataset(full_dataset['validation'], train=False, augmented=False,
                                                 n_views=args.num_of_agent, selected_class=ImgNet_select)

    elif args.dataset == 'KMNIST':
        eval_dataset_train = GenericMNISTDataset(path=path, train=True, augmented=False,
                                                 n_views=args.num_of_agent, name="KMNIST")
        eval_dataset_test = GenericMNISTDataset(path=path, train=False, augmented=False,
                                                n_views=args.num_of_agent, name="KMNIST")

    else:
        print('Unknown dataset:', args.dataset)
        return None

    dataloader_train = DataLoader(eval_dataset_train, batch_size=args.batch_size, shuffle=False, pin_memory=True,
                                  num_workers=args.num_workers, persistent_workers=True)
    dataloader_test = DataLoader(eval_dataset_test, batch_size=args.batch_size, shuffle=False, pin_memory=True,
                                 num_workers=args.num_workers, persistent_workers=True)
    return dataloader_train, dataloader_test


def get_dataloader_train(args):
    if args.dataset == 'FashionMNIST':
        train_dataset = GenericMNISTDataset(path=path, train=True, n_views=args.num_of_agent, name="FashionMNIST")
        saved = args.run_path + args.eval_model + '_FashMNIST'
    elif args.dataset == 'MNIST':
        train_dataset = GenericMNISTDataset(path=path, train=True, n_views=args.num_of_agent, name="MNIST")
        saved = args.run_path + args.eval_model + '_MNIST'

    elif args.dataset == 'CIFAR':
        train_dataset = GenericCIFARDataset(path=path, train=True, image_size=args.image_size,
                                            augmented=True, dif_augmented=args.dif_augmented,
                                            n_views=args.num_of_agent, name="CIFAR10")
        saved = args.run_path + args.eval_model + '_CIFAR'
    elif args.dataset == 'CIFAR100':
        train_dataset = GenericCIFARDataset(path=path, train=True, image_size=args.image_size,
                                            augmented=True, dif_augmented=args.dif_augmented,
                                            n_views=args.num_of_agent, name="CIFAR100")
        saved = args.run_path + args.eval_model + '_CIFAR100'

    elif args.dataset == 'ImageNet100':
        full_dataset = load_dataset("clane9/imagenet-100")
        train_dataset = ImageNet100HFDataset(full_dataset['train'], train=True, augmented=True,
                                             n_views=args.num_of_agent, selected_class=ImgNet_select)
        saved = args.run_path + args.eval_model + '_ImageNetHF100'

    elif args.dataset == 'KMNIST':
        train_dataset = GenericMNISTDataset(path=path, train=True, n_views=args.num_of_agent, name="KMNIST")
        saved = args.run_path + args.eval_model + '_KMNIST'

    else:
        print('Unknown dataset:', args.dataset)
        return None

    dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True,
                            num_workers=args.num_workers, persistent_workers=True)
    return dataloader, saved


def set_seeds(seed):
    if seed == -1:
        seed = random.randint(1, 100)
    torch.manual_seed(seed)
    np.random.seed(seed)
    print('Seed: {:.2g}'.format(seed))
